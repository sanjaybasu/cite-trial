"""Figures for the CITE npj Digital Medicine paper. Every plotted value is read from
audit/CANONICAL_NUMBERS.json, except the per-encounter flag counts in Fig. 2b, which are read from the
locked trial dataset (cite_n_flags). Writes figures/Fig{1,2,3}.png and .pdf.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DATA = "/Users/sanjaybasu/waymark-local/notebooks/anchor/rct_analysis/derived/analysis_dataset_wave11_locked.csv"
C = json.load(open(os.path.join(HERE, "CANONICAL_NUMBERS.json")))
BLUE, ORANGE, AQUA, GRAY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def kn(s):
    k, n = re.match(r"(\d+)/(\d+)", str(s)).groups()
    return int(k), int(n)


def rng(s):
    a, b = [float(x) for x in re.findall(r"-?\d+\.\d+", s)]
    return a, b


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#f0efec"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=INK2, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=INK, wrap=True)


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.0); ax.axis("off")
    ax.text(0.0, 0.99, "a  Randomized trial in live Medicaid primary care (NCT07718893)", fontsize=8.5, weight="bold", va="top")
    box(ax, 0.30, 0.82, 0.40, 0.11, f"{C['n_total']} patients randomized 1:1:1\n{C['enroll_first']} to {C['enroll_last']}")
    xs = (0.0, 0.345, 0.69)
    labs = (f"Arm 1: unassisted clinician\nallocated n = {C['n_arm1']}",
            f"Arm 2: Gemini 3.1 Pro draft\n+ safety guardrails\nallocated n = {C['n_arm2']}",
            f"Arm 3: Arm 2 + CITE verifier\nallocated n = {C['n_arm3']}\n(CITE ran in {C['cite_ran_n']})")
    linked = (C["linked_arm1"], C["linked_arm2"], C["linked_arm3"])
    for x, lab, lk, n in zip(xs, labs, linked, (C["n_arm1"], C["n_arm2"], C["n_arm3"])):
        box(ax, x, 0.58, 0.30, 0.16, lab)
        ax.annotate("", xy=(x + 0.15, 0.74), xytext=(0.5, 0.82), arrowprops=dict(arrowstyle="->", color=INK2))
        box(ax, x, 0.40, 0.30, 0.12, f"Analysed (ITT): {n}\n30-day data linked: {lk}\nnot linked: {n - lk}", fc="#ffffff")
        ax.annotate("", xy=(x + 0.15, 0.52), xytext=(x + 0.15, 0.58), arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(0.0, 0.27, "b  Retrospective rating of language-model care plans for authored cases", fontsize=8.5, weight="bold", va="top")
    box(ax, 0.0, 0.0, 0.29, 0.20, f"{C['retro_n_plans']:,} care plans\n{C['retro_n_models']} models x {C['retro_per_model']}\nauthored Medicaid cases")
    box(ax, 0.34, 0.0, 0.29, 0.20, "Rubric ratings drafted by a\nlanguage model, each reviewed\nand approved by clinicians")
    box(ax, 0.68, 0.0, 0.31, 0.20, f"Deficient plans: {C['retro_deficient'].replace('/1341', '/1,341')}\nGemini 3.1 Pro, codifiable:\n{C['retro_cod_gemini']}")
    for x in (0.29, 0.63):
        ax.annotate("", xy=(x + 0.04, 0.10), xytext=(x, 0.10), arrowprops=dict(arrowstyle="->", color=INK2))
    save(fig, "Fig1")


def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axs[0]
    rows = [("Live trial drafts:\nGemini 3.1 Pro", C["haz_draft_total"], BLUE), ("Authored cases:\nGemini 3.1 Pro", C["retro_cod_gemini"], GRAY),
            ("Authored cases:\nGPT-5.6", C["retro_cod_gpt"], GRAY), ("Authored cases:\nClaude Opus 4.8", C["retro_cod_claude"], GRAY)]
    for i, (lab, s, col) in enumerate(rows):
        k, n = kn(s)
        lo, hi = proportion_confint(k, n, method="beta")
        ax.errorbar(100 * k / n, i, xerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], fmt="o", color=col, ms=5, capsize=2, lw=1.5)
        ax.text(100 * hi + 0.3, i, f"{k}/{n}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel("Plans with a codifiable hazard, % (95% CI)"); ax.set_xlim(0, 7.5)
    ax.set_title("a", loc="left", weight="bold")
    ax = axs[1]
    d = pd.read_csv(DATA, low_memory=False)
    f = pd.to_numeric(d[(d.arm == "gemini_guardrails_anchor") & (d.cite_used == "t")].cite_n_flags, errors="coerce").fillna(0).astype(int)
    vc = f.value_counts().sort_index()
    assert int(f.sum()) == int(C["cite_flags_total"]) and len(f) == int(C["cite_ran_n"])
    ax.bar(vc.index, vc.values, color=BLUE, width=0.7)
    for x, y in vc.items():
        ax.text(x, y + 1, str(y), ha="center", fontsize=6.5, color=INK2)
    ax.axvline(C["alert_ceiling"], color=INK2, ls="--", lw=1)
    ax.text(C["alert_ceiling"] + 0.1, vc.max() * 0.8, "pre-registered\nceiling (median 3)", fontsize=6, color=INK2)
    ax.set_xlabel("CITE flags per encounter"); ax.set_ylabel("Encounters"); ax.set_xlim(-0.6, 4.5); ax.set_xticks(range(0, 5))
    ax.set_title("b", loc="left", weight="bold")
    ax = axs[2]
    om_v, om_a = C["flag_omission_valid"], C["valid_omission_acted"]
    cf_v, cf_a = C["flag_confabulation_valid"], C["valid_confab_removed"]
    ax.barh([0, 1], [om_a, cf_a], color=BLUE, height=0.55, label="acted on in signed note")
    ax.barh([0, 1], [om_v - om_a, cf_v - cf_a], left=[om_a, cf_a], color="#cde2fb", height=0.55, label="not acted on")
    for y, (a_, v_) in enumerate(((om_a, om_v), (cf_a, cf_v))):
        ax.text(v_ + 0.15, y, f"{a_}/{v_}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Valid omission flags\n(action added?)", "Valid confabulation flags\n(statement removed?)"], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlabel("Valid CITE flags (n)"); ax.set_xlim(0, 6.5)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    ax.set_title("c", loc="left", weight="bold")
    fig.tight_layout()
    save(fig, "Fig2")


def fig3():
    rows = [("Safety composite (exploratory)", "composite", "sens_composite"),
            ("  Missed do-not-miss diagnosis (exploratory)", "missed_dx", None),
            ("  Under-triage (exploratory)", "undertriage", None),
            ("  Missed escalation (exploratory)", "missed_escalation", None),
            ("Over-escalation (protocol, 5-point margin)", "over_esc", "sens_over_esc"),
            ("Appropriate escalation (protocol)", "approp_esc", None),
            ("30-day ED visit (exploratory)", "ed", "sens_ed"),
            ("30-day hospitalization (exploratory)", "hosp", "sens_hosp")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (lab, k, _) in enumerate(rows):
        for off, cmp, col in ((-0.18, "32", BLUE), (0.18, "21", ORANGE)):
            e = float(C[f"{k}_rd{cmp}"]); lo, hi = rng(C[f"{k}_rd{cmp}_ci"])
            ax.errorbar(e, i + off, xerr=[[e - lo], [hi - e]], fmt="o", color=col, ms=4, capsize=2, lw=1.4)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.plot([C["over_esc_margin_pp"]] * 2, [4 - 0.45, 4 - 0.05], color=INK, ls=":", lw=1.2)
    ax.text(C["over_esc_margin_pp"] + 0.4, 4.0, "5-point margin", fontsize=6, color=INK, va="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Risk difference, percentage points (95% CI)")
    ax.plot([], [], "o", color=BLUE, label="Arm 3 (CITE) minus Arm 2 (guardrails)")
    ax.plot([], [], "o", color=ORANGE, label="Arm 2 (guardrails) minus Arm 1 (unassisted)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "Fig3")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig1(); fig2(); fig3()
    print("figures written")
