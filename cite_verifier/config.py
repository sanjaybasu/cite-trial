"""
Central control surface for the ebm-verify retrospective validation study.

This is the single place that governs cost, models, seeds, and stratum sizes. The hard
budget ceiling here is the kill switch: no paid call is issued once cumulative spend reaches
BUDGET_USD_CEILING (enforced by cost_ledger.check_budget before every call).

Design goals (mirroring the ANCHOR paper-1 retrospective):
  - deterministic + seed-locked so the run is reproducible and pre-registrable
  - cheap by construction (Batch API, cheap verifier/judge, deterministic scoring where possible)
  - fail-safe: cannot silently overspend; can always resume without re-incurring cost
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Base models UNDER TEST (the deployed clinical LLM). Capability curve, not a single model,
# to avoid the single-strongest-model null trap and to satisfy model-agnostic framing.
# NOTE: claude-fable-5 requires 30-day org data retention (400 model_not_available otherwise).
# This study uses NO patient data, so retention is not a PHI concern; add "claude-fable-5" back
# to this list once retention is enabled — no other code change needed. Until then the frontier
# tier is Opus 4.8 (current top Opus-tier, $5/$25, works under the present retention config).
# Cross-vendor frontier grid (real model-agnostic claim + enables verifier independence).
# Add "claude-fable-5" once 30-day org data retention is enabled.
BASE_MODELS = ["claude-opus-4-8", "gpt-5.6", "gemini-3.1-pro-preview"]

# Cheap, NON-REASONING extractor/verifier per vendor. NB: gpt-5-mini is a reasoning model that can
# return empty text at low token budgets, so it is NOT used for extraction; Haiku and Gemini-flash
# are safe. Cross-vendor independence: the verifier is always a DIFFERENT vendor than the generator
# (no model grades its own output -> removes self-preference bias).
_CHEAP_NONREASONING = {"anthropic": "claude-haiku-4-5", "gemini": "gemini-2.5-flash"}
EXTRACTOR_MODEL = "claude-haiku-4-5"   # default when base vendor is unknown
VERIFIER_MODEL = "claude-haiku-4-5"


def _vendor(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o3", "o4")):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    return "anthropic"


def verifier_for(base_model: str) -> str:
    """A cheap non-reasoning extractor/verifier from a DIFFERENT vendor than base_model."""
    v = _vendor(base_model)
    if v == "anthropic":
        return _CHEAP_NONREASONING["gemini"]      # claude draft -> gemini verifier
    return _CHEAP_NONREASONING["anthropic"]        # gpt/gemini draft -> claude verifier

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens) — from the claude-api skill, cached 2026-06-04.
# ---------------------------------------------------------------------------
PRICING = {
    "claude-fable-5":   {"in": 10.0, "out": 50.0},
    "claude-opus-4-8":  {"in": 5.0,  "out": 25.0},
    "claude-sonnet-4-6":{"in": 3.0,  "out": 15.0},
    "claude-haiku-4-5": {"in": 1.0,  "out": 5.0},
    # cross-vendor — CONSERVATIVE upper bounds for the budget guard only. VERIFY exact
    # published prices before reporting any cost figure in the manuscript.
    "gpt-5.6":          {"in": 5.00, "out": 30.0},  # gpt-5.6-sol real price
    "gpt-5-mini":       {"in": 0.25, "out": 2.0},   # VERIFY
    "gpt-5":            {"in": 1.25, "out": 10.0},  # VERIFY
    "gpt-4.1-mini":     {"in": 0.40, "out": 1.60},  # VERIFY
    "gemini-3.1-pro":   {"in": 2.50, "out": 15.0},  # VERIFY (conservative)
    "gemini-3-pro":     {"in": 2.50, "out": 15.0},  # VERIFY
    "gemini-2.5-pro":   {"in": 1.25, "out": 10.0},  # VERIFY
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},  # VERIFY
}
BATCH_DISCOUNT = 0.5      # Message Batches API: 50% off input + output
CACHE_READ_MULT = 0.1    # cached prefix reads bill at ~0.1x input

# ---------------------------------------------------------------------------
# Cost controls (the "cut off if expense" requirement)
# ---------------------------------------------------------------------------
USE_BATCH = True                 # route non-interactive calls through the Batch API (50% off)
BUDGET_USD_CEILING = 200.0
BUDGET_SOFT_WARN_FRAC = 0.8      # log a loud warning at 80% of ceiling
MAX_TOKENS_DRAFT = 3000          # base-model output cap. Higher than Opus-tier because Fable 5's
                                 # always-on thinking is billed within max_tokens — too low a cap
                                 # truncates the answer mid-reasoning.
MAX_TOKENS_EXTRACT = 800
MAX_TOKENS_VERIFY = 500

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 20260707                  # locked; used for all stratified sampling and case ordering

# ---------------------------------------------------------------------------
# Strata (case counts). Pilot is deterministic and small; full is powered to the
# PRE-REGISTERED N=984 (OSF r6u5f / power.py: the binding constraint is subgroup
# precision on the widest equity axis — race/ethnicity ~6 levels at CI half-width
# 0.10 -> 164/level -> ~984). The earlier 655 total pre-dated the equity power calc
# and is SUPERSEDED. Enriched strata carry the power; `average` carries the honest
# base-rate denominator (H2 deployed-prevalence); true_negative + hard_negative carry
# SPECIFICITY / false-flag rate. Panel authors under the clinical category names
# (authoring_instructions.md); the coordinator files each case into its harness stratum
# via CATEGORY_TO_STRATUM below.
# ---------------------------------------------------------------------------
STRATA = {
    # harness stratum    (pilot_n, full_n)   axis it powers                         [panel category]
    "average":          (5, 120),   # honest base rate (low effect, big N)          [typical]
    "obscure_lit":      (5,  60),   # hallucination — Paperclip's regime            [obscure_evidence]
    "contraindication": (5, 120),   # commission — over-treatment edge cases        [contraindication]
    "guarded_omission": (5, 120),   # omission — guardrail-suppressed GDMT          [guideline_gap]
    # whole-person / social / longitudinal (Tier 1-2 deterministic; Tier 3 -> adjudication)
    "whole_person":     (5, 100),   # preventive + care-transition omissions (USPSTF/CMS-TCM/HEDIS)  [whole_person]
    "social_hrsn":      (5, 100),   # HRSN screen+refer omissions (CMS SDOH / USPSTF / NC-HOP / CalAIM)  [hrsn]
    "longitudinal":     (5, 100),   # safety-critical drops (allergy->Rx, dropped med/condition)     [longitudinal]
    # negative controls -> measure SPECIFICITY / false-flag rate (the adversarial-review fix)
    "true_negative":    (5, 164),   # correct plan has NO deficiency; layer must stay quiet          [true_negative]
    "hard_negative":    (5, 100),   # a naive rule would fire but acting is WRONG (contra/already-addressed)  [hard_negative]
}   # full-mode total = 984 (matches OSF prereg + power.py)

# Panel-facing clinical category  ->  harness stratum key above. The panel authors under the
# clinical names; the coordinator uses this to file each case into the pre-registered stratum.
# This is the single source of truth for the name divergence between the two documents.
CATEGORY_TO_STRATUM = {
    "typical":          "average",
    "guideline_gap":    "guarded_omission",
    "contraindication": "contraindication",
    "whole_person":     "whole_person",
    "hrsn":             "social_hrsn",
    "longitudinal":     "longitudinal",
    "obscure_evidence": "obscure_lit",
    "true_negative":    "true_negative",
    "hard_negative":    "hard_negative",
}

def stratum_n(name: str, mode: str) -> int:
    i = 0 if mode == "pilot" else 1
    return STRATA[name][i]

def total_n(mode: str) -> int:
    return sum(stratum_n(k, mode) for k in STRATA)

# ---------------------------------------------------------------------------
# Paths. On Modal these resolve inside a persistent Volume so a killed/rescheduled
# run resumes from the same ledger + partial results (no data loss, no re-spend).
# ---------------------------------------------------------------------------
import os
DATA_ROOT = os.environ.get("EBMV_DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))
CORPUS_SNAPSHOT = os.path.join(DATA_ROOT, "corpus_snapshot.json")   # frozen full-text evidence
# RUN_DIR holds all mutable run state (ledger, results, resume cursor). Override with
# EBMV_RUN_DIR to point at a Modal Volume so a killed/rescheduled run resumes with no data loss.
RUN_DIR = os.environ.get("EBMV_RUN_DIR", os.path.join(DATA_ROOT, "run"))
LEDGER_PATH = os.path.join(RUN_DIR, "cost_ledger.jsonl")   # atomic cost ledger
RESULTS_PATH = os.path.join(RUN_DIR, "results.jsonl")      # per-case results (append)
STATE_PATH = os.path.join(RUN_DIR, "state.json")           # resume cursor
