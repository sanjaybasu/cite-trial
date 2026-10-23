"""
Cost ledger + hard budget kill switch + resumable spend accounting.

Every paid LLM call is recorded here as one append-only JSONL line, written atomically. The
running total is derived by summing the ledger, so a killed/rescheduled/restarted run
reconstructs exact prior spend from disk and never double-counts or re-incurs cost.

Two guarantees this module enforces:
  1. cut off if expense  -> check_budget() raises BudgetExceeded before a call would push
     cumulative spend past config.BUDGET_USD_CEILING.
  2. don't lose data     -> record() uses atomic append; spent_so_far() re-reads the ledger,
     so progress survives process death, laptop sleep, and Modal rescheduling.

Nothing here is model- or provider-specific beyond the pricing table in config.
"""
from __future__ import annotations

import json
import os
import tempfile

from config import PRICING, BATCH_DISCOUNT, CACHE_READ_MULT, BUDGET_USD_CEILING, BUDGET_SOFT_WARN_FRAC


class BudgetExceeded(RuntimeError):
    """Raised to stop the run cleanly when the budget ceiling is reached."""


def _price(model: str) -> dict:
    """Look up pricing tolerant of dated full IDs (the API returns e.g.
    'claude-haiku-4-5-20251001' where our table keys on the alias 'claude-haiku-4-5')."""
    if model in PRICING:
        return PRICING[model]
    # longest matching prefix wins (so 'gpt-5-mini' matches 'gpt-5-mini', not 'gpt-5')
    for k in sorted(PRICING, key=len, reverse=True):
        if model.startswith(k):
            return PRICING[k]
    raise KeyError(f"no pricing for model {model!r}")


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cached_input_tokens: int = 0, batch: bool = False) -> float:
    """Compute USD cost for one call. Cached input tokens bill at CACHE_READ_MULT of input
    rate; batch applies BATCH_DISCOUNT to the whole call. Thinking tokens (Fable 5) are billed
    as output by the API and arrive already counted in output_tokens."""
    p = _price(model)
    fresh_input = max(0, input_tokens - cached_input_tokens)
    dollars = (
        fresh_input / 1e6 * p["in"]
        + cached_input_tokens / 1e6 * p["in"] * CACHE_READ_MULT
        + output_tokens / 1e6 * p["out"]
    )
    if batch:
        dollars *= BATCH_DISCOUNT
    return dollars


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def spent_so_far(ledger_path: str) -> float:
    """Reconstruct cumulative spend from the ledger on disk. Source of truth for resume."""
    if not os.path.exists(ledger_path):
        return 0.0
    total = 0.0
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line)["usd"])
            except (ValueError, KeyError, json.JSONDecodeError):
                continue  # tolerate a torn final line from an interrupted write
    return total


def record(ledger_path: str, *, stage: str, model: str, input_tokens: int,
           output_tokens: int, cached_input_tokens: int = 0, batch: bool = False,
           case_id: str = "", ts: str = "") -> float:
    """Atomically append one cost record. Returns the new cumulative total.
    `ts` is passed in (never generated here) so the module stays deterministic/resume-safe."""
    _ensure_dir(ledger_path)
    usd = cost_usd(model, input_tokens, output_tokens, cached_input_tokens, batch)
    rec = {
        "ts": ts, "stage": stage, "case_id": case_id, "model": model,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens, "batch": batch, "usd": round(usd, 6),
    }
    # append is atomic for small lines on POSIX; flush + fsync to survive power loss / sleep
    with open(ledger_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return spent_so_far(ledger_path)


def check_budget(ledger_path: str, next_call_estimate_usd: float = 0.0,
                 ceiling: float = BUDGET_USD_CEILING) -> float:
    """Call BEFORE issuing a paid request. Raises BudgetExceeded if the current spend plus the
    estimated next call would exceed the ceiling. Returns current spend otherwise."""
    spent = spent_so_far(ledger_path)
    if spent + next_call_estimate_usd > ceiling:
        raise BudgetExceeded(
            f"Budget ceiling ${ceiling:.2f} would be exceeded: "
            f"spent ${spent:.4f} + next ~${next_call_estimate_usd:.4f}. Halting cleanly."
        )
    if spent >= ceiling * BUDGET_SOFT_WARN_FRAC:
        print(f"[cost] WARNING: at ${spent:.4f} of ${ceiling:.2f} ceiling "
              f"({spent/ceiling*100:.0f}%).")
    return spent


def summary(ledger_path: str) -> dict:
    """Aggregate ledger for the tracker / paper cost table."""
    by_stage: dict[str, float] = {}
    by_model: dict[str, float] = {}
    n_calls = 0
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_calls += 1
                by_stage[r["stage"]] = by_stage.get(r["stage"], 0.0) + r["usd"]
                by_model[r["model"]] = by_model.get(r["model"], 0.0) + r["usd"]
    return {
        "total_usd": round(spent_so_far(ledger_path), 4),
        "n_calls": n_calls,
        "by_stage": {k: round(v, 4) for k, v in by_stage.items()},
        "by_model": {k: round(v, 4) for k, v in by_model.items()},
    }
