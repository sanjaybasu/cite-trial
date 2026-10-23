"""Additional analyses for the CITE (NCT07718893) npj Digital Medicine paper.

Writes results/wave11_locked/cite_paper_extras.json (aggregate only):
  - composite components by arm; appropriate escalation; over-escalation (protocol safety margin 5 pp)
  - time to decision (median, IQR; Mann-Whitney U Arm 3 vs Arm 2; t-test on log seconds)
  - minimum detectable difference for the composite at the achieved sample (80% power, two-sided 0.05)
  - design-effect scenarios for unmodelled clinician clustering
  - insufficient-content rate by arm (Packet A and Packet B consensus)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

HERE = os.path.dirname(os.path.abspath(__file__))
RCT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(RCT, "results", "wave11_locked")
PK = os.path.join(RCT, "adjudication_packets_cite_outcomes")
A1, A2, A3 = "unassisted_md_soap", "gemini_guardrails", "gemini_guardrails_anchor"
sys.path.insert(0, HERE)
from compute_exploratory_and_fairness import newcombe, wilson  # noqa: E402


def prop(df, col):
    v = pd.to_numeric(df[col], errors="coerce")
    k, n = int(v.sum()), int(v.notna().sum())
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "lo": lo, "hi": hi}


def rd(d, col, a, b):
    x, y = prop(d[d.arm == a], col), prop(d[d.arm == b], col)
    e, lo, hi = newcombe(x["k"], x["n"], y["k"], y["n"])
    return {"rd": e, "lo": lo, "hi": hi, "fisher_p": float(stats.fisher_exact([[x["k"], x["n"] - x["k"]], [y["k"], y["n"] - y["k"]]])[1])}


def main():
    d = pd.read_csv(os.path.join(RCT, "derived", "analysis_dataset_wave11_locked.csv"), low_memory=False)
    out = {"n_by_arm": d.arm.value_counts().to_dict()}
    for col in ["primary_event_final", "adj_missed_dx", "adj_undertriage", "adj_contraindicated_med",
                "adj_missed_escalation", "over_escalation_adjudicated", "appropriate_escalation_adjudicated"]:
        out[col] = {"by_arm": {a: prop(d[d.arm == a], col) for a in (A1, A2, A3)},
                    "arm3_vs_arm2": rd(d, col, A3, A2), "arm2_vs_arm1": rd(d, col, A2, A1)}
    out["over_escalation_margin_pp"] = 5.0
    out["over_escalation_upper_bound_below_margin"] = bool(out["over_escalation_adjudicated"]["arm3_vs_arm2"]["hi"] * 100 < 5.0)

    t = {a: pd.to_numeric(d[d.arm == a].time_to_decision_seconds, errors="coerce").dropna() for a in (A1, A2, A3)}
    out["time_to_decision"] = {a: {"n": int(len(v)), "median": float(v.median()), "q1": float(v.quantile(.25)),
                                   "q3": float(v.quantile(.75))} for a, v in t.items()}
    out["time_to_decision"]["mannwhitney_p_arm3_vs_arm2"] = float(stats.mannwhitneyu(t[A3], t[A2]).pvalue)
    out["time_to_decision"]["log_ttest_p_arm3_vs_arm2"] = float(stats.ttest_ind(np.log(t[A3].clip(lower=1)), np.log(t[A2].clip(lower=1))).pvalue)

    p0 = out["primary_event_final"]["by_arm"][A2]["rate"]
    n3, n2 = out["primary_event_final"]["by_arm"][A3]["n"], out["primary_event_final"]["by_arm"][A2]["n"]
    pw = NormalIndPower()
    mdd = None
    for delta in np.arange(0.005, 0.5, 0.0025):
        if pw.power(abs(proportion_effectsize(p0 + delta, p0)), nobs1=n3, ratio=n2 / n3, alpha=0.05) >= 0.8:
            mdd = float(delta)
            break
    out["mdd_composite_increase_80pct_power"] = {"baseline_rate_arm2": p0, "mdd": mdd, "n3": n3, "n2": n2}

    e = out["primary_event_final"]["arm3_vs_arm2"]
    half = (e["hi"] - e["lo"]) / 2
    out["design_effect_scenarios"] = [
        {"cluster_size": m, "icc": icc, "design_effect": 1 + (m - 1) * icc,
         "ci_lo": e["rd"] - half * np.sqrt(1 + (m - 1) * icc), "ci_hi": e["rd"] + half * np.sqrt(1 + (m - 1) * icc)}
        for m in (10, 20, 40) for icc in (0.01, 0.05, 0.10)]

    key = json.load(open(os.path.join(PK, "PRIVATE_key_map.json")))
    for P, name in (("A", "packetA_ai_draft"), ("B", "packetB_final_plan")):
        L = pd.concat([pd.read_csv(os.path.join(PK, r, f"{name}_{r}.csv"), keep_default_na=False, dtype=str)
                       for r in ("adj1", "adj2", "adj3")])
        ins = L.assign(i=L["insufficient_content(yes|no)"].str.strip().str.lower().eq("yes")).groupby("blinded_id").i.sum() >= 2
        k = pd.DataFrame(key[P]).set_index("blinded_id").arm
        out[f"insufficient_content_packet{P}"] = {a: {"k": int(ins[k[ins.index] == a].sum()), "n": int((k == a).sum())}
                                                  for a in k.unique()}
    with open(os.path.join(OUT, "cite_paper_extras.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("mdd_composite_increase_80pct_power", "time_to_decision",
                                          "insufficient_content_packetA", "insufficient_content_packetB")}, indent=1, default=float))
    print("over-esc", out["over_escalation_adjudicated"]["arm3_vs_arm2"], "appropriate", out["appropriate_escalation_adjudicated"]["by_arm"])


if __name__ == "__main__":
    main()
