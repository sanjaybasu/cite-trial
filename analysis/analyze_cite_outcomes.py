"""CITE trial (NCT07718893) — registered outcomes from the returned adjudication packets.

Reads adj{1,2,3}/packet{A,B,D}_*.csv (and PRIVATE_packetC_30day_chart_review.csv if present),
unblinds with PRIVATE_key_map.json, and writes results/wave11_locked/cite_registered_outcomes.json.

Pre-specified (OSF addendum §3, §7):
  Co-primary 1  CITE sens/spec/PPV/NPV vs majority-of-three hazard_present on the AI draft (Packet A),
                Arm 3 encounters where CITE ran; exact (Clopper-Pearson) 95% CIs.
                ANY-flag (pre-registered) = commission or omission flag; every-flag is a post hoc expansion. ESCALATE-only = a commission flag (pre-registered commission/allergy/refuted numeric).
                Primary operating point = ESCALATE-only (commission flags in cite-1.0); ANY-flag is the sensitivity analysis.
  Co-primary 2  flags per encounter (median vs ceiling 3), acceptance (flag content in final note,
                Packet D), alert PPV (flag_valid, Packet D).
  Descriptive   unresolved hazards per checkpoint in the final plan by arm (Packet B), Arm 3 - Arm 2
                with an encounter-level cluster bootstrap CI.
  Secondary     30-day hazard correction and referral completion (Packet C).
Sensitivity: every estimate is repeated excluding the 15 participants younger than 18.

Usage: python analyze_cite_outcomes.py [--packets DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
RCT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(RCT, "results", "wave11_locked")
RATERS = ("adj1", "adj2", "adj3")
A2, A3 = "gemini_guardrails", "gemini_guardrails_anchor"
sys.path.insert(0, HERE)
from analyze_rct import fleiss_kappa  # noqa: E402

HAZ = "hazard_present(yes|no)"
INSUF = "insufficient_content(yes|no)"
NCHK = "n_checkpoints"


def col(df, prefix):
    return next(c for c in df.columns if c.startswith(prefix))


def yes(s):
    return s.astype(str).str.strip().str.lower().eq("yes")


def cp(k, n):
    if n == 0:
        return {"k": 0, "n": 0, "est": None, "lo": None, "hi": None}
    lo, hi = proportion_confint(k, n, method="beta")
    return {"k": int(k), "n": int(n), "est": k / n, "lo": float(lo), "hi": float(hi)}


def load(pdir, name):
    frames = []
    for r in RATERS:
        f = pd.read_csv(os.path.join(pdir, r, f"{name}_{r}.csv"), keep_default_na=False)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def gwet_ac1(items):
    """Gwet AC1 for binary ratings; items = (n, 2) counts per category, constant raters per item."""
    r = items.sum(axis=1)[0]
    pa = ((items * (items - 1)).sum(axis=1) / (r * (r - 1))).mean()
    pi = items.sum(axis=0) / items.sum()
    pe = (pi * (1 - pi)).sum()  # q = 2 categories: pe = sum pi(1-pi)/(q-1)
    return float((pa - pe) / (1 - pe))


def kappa(long, flag_col):
    v = long.assign(y=yes(long[flag_col])).groupby("blinded_id").y.agg(["sum", "count"])
    v = v[v["count"] == 3]
    items = np.column_stack([3 - v["sum"].values, v["sum"].values])
    return {"fleiss_kappa": fleiss_kappa(items), "gwet_ac1": gwet_ac1(items),
            "pct_unanimous": float(((v["sum"] == 0) | (v["sum"] == 3)).mean()),
            "positives_any_rater": int((v["sum"] > 0).sum()), "n_items": int(len(v))}


def n_sev2(fam, sev):
    """Number of listed hazards with severity >= 2 for one rater's row."""
    s = [x.strip() for x in str(sev).split(";") if x.strip()]
    return sum(1 for x in s if x.isdigit() and int(x) >= 2)


def consensus(long):
    long = long.copy()
    long["haz"] = yes(long[HAZ])
    long["insuf"] = yes(long[INSUF])
    long["nchk"] = pd.to_numeric(long[NCHK], errors="coerce")
    fam, sev = col(long, "hazard_family"), col(long, "hazard_severity")
    long["nhaz"] = [n_sev2(f, s) for f, s in zip(long[fam], long[sev])]
    long["fams"] = long[fam].astype(str)
    g = long.groupby("blinded_id")
    c = pd.DataFrame({
        "haz": g.haz.sum() >= 2,
        "insuf": g.insuf.sum() >= 2,
        "nchk": g.nchk.median(),
        "nhaz": g.nhaz.median(),
        "fams": g.fams.apply(lambda v: ";".join(v)),
    })
    c.loc[~c.haz, "nhaz"] = 0
    return c.reset_index()


def diag(df, test_col):
    tp = int((df.ref & df[test_col]).sum()); fn = int((df.ref & ~df[test_col]).sum())
    tn = int((~df.ref & ~df[test_col]).sum()); fp = int((~df.ref & df[test_col]).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": cp(tp, tp + fn), "specificity": cp(tn, tn + fp),
            "ppv": cp(tp, tp + fp), "npv": cp(tn, tn + fn)}


def boot_rate_diff(b, n=2000, seed=20261019):
    rng = np.random.default_rng(seed)
    x3, x2 = b[b.arm == A3], b[b.arm == A2]
    def rate(x):
        return x.nhaz.sum() / x.nchk.sum() if x.nchk.sum() > 0 else np.nan
    est = rate(x3) - rate(x2)
    bs = [rate(x3.sample(len(x3), replace=True, random_state=int(rng.integers(1e9))))
          - rate(x2.sample(len(x2), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(n)]
    return {"rd_per_checkpoint": est, "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}


def boot_prop_diff(x, flag, n=2000, seed=20261019):
    """Arm 3 minus Arm 2 difference in a referral-level proportion; bootstrap resamples encounters within arm."""
    rng = np.random.default_rng(seed)
    g = {arm: [grp[flag].to_numpy() for _, grp in x[x.arm == arm].groupby("encounter_id")] for arm in (A3, A2)}
    def prop(groups):
        v = np.concatenate(groups)
        return v.mean() if len(v) else np.nan
    est = prop(g[A3]) - prop(g[A2])
    bs = [prop([g[A3][i] for i in rng.integers(0, len(g[A3]), len(g[A3]))])
          - prop([g[A2][i] for i in rng.integers(0, len(g[A2]), len(g[A2]))]) for _ in range(n)]
    return {"rd": float(est), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
            "n_encounters_arm3": len(g[A3]), "n_encounters_arm2": len(g[A2])}


def analyze(d, pdir, keep):
    """keep: set of encounter_ids in the analysis population."""
    key = json.load(open(os.path.join(pdir, "PRIVATE_key_map.json")))
    res = {}
    dd = d.set_index("encounter_id")

    # Co-primary 1: CITE diagnostic accuracy (Packet A, Arm 3, CITE ran)
    A = load(pdir, "packetA_ai_draft")
    res["reliability_packetA_hazard"] = kappa(A, HAZ)
    res["reliability_packetA_insufficient"] = kappa(A, INSUF)
    ca = consensus(A).merge(pd.DataFrame(key["A"]), on="blinded_id")
    ca = ca[ca.encounter_id.isin(keep)]
    a3 = ca[(ca.arm == A3)].copy()
    a3 = a3[a3.encounter_id.map(dd.cite_used).eq("t")]
    a3["ref"] = a3.haz
    flags = a3.encounter_id.map(dd.cite_flags).apply(lambda s: json.loads(s) if isinstance(s, str) and s.strip() else [])
    a3["any_flag"] = flags.apply(len) > 0
    # Pre-registered ESCALATE-only (OSF addendum section 3): commission, allergy conflict, or refuted numeric claim.
    # cite-1.0 has no allergy-conflict or numeric-claim type, so this point is operationalized as any commission flag. Commission or confabulation is a post hoc threshold.
    a3["escalate"] = flags.apply(lambda fl: any(f.get("type") == "commission" for f in fl))
    a3["posthoc_cc"] = flags.apply(lambda fl: any(f.get("type") in ("commission", "confabulation") for f in fl))
    res["copri1_escalate_only"] = diag(a3, "escalate")
    res["copri1_any_flag"] = diag(a3, "any_flag")
    # Pre-registered ANY-flag (OSF addendum section 3): ESCALATE plus omissions (commission or omission flags).
    a3["any_prereg"] = flags.apply(lambda fl: any(f.get("type") in ("commission", "omission") for f in fl))
    res["copri1_any_flag_prereg"] = diag(a3, "any_prereg")
    res["copri1_posthoc_commission_or_confabulation"] = diag(a3, "posthoc_cc")
    res["copri1_excl_insufficient_posthoc_commission_or_confabulation"] = diag(a3[~a3.insuf], "posthoc_cc")
    fam_list = ["contraindication", "drug_allergy", "gdmt_omission", "high_risk_med_dropped", "numeric_refuted"]
    res["copri1_by_family_any_flag_sensitivity"] = {
        f: cp(int((a3.fams.str.contains(f) & a3.ref & a3.any_flag).sum()), int((a3.fams.str.contains(f) & a3.ref).sum()))
        for f in fam_list}
    res["hazard_prevalence_ai_draft"] = {arm: cp(int(x.haz.sum()), len(x)) for arm, x in ca.groupby("arm")}

    # Co-primary 2: alert burden, acceptance, PPV
    ran = d[(d.arm == A3) & d.cite_used.eq("t") & d.encounter_id.isin(keep)]
    nf = pd.to_numeric(ran.cite_n_flags, errors="coerce").fillna(0)
    res["copri2_flags_per_encounter"] = {"n_encounters": int(len(ran)), "total_flags": int(nf.sum()),
                                          "median": float(nf.median()), "mean": float(nf.mean()),
                                          "p90": float(nf.quantile(0.9)), "ceiling": 3,
                                          "meets_ceiling": bool(nf.median() <= 3)}
    D = load(pdir, "packetD_cite_flags")
    res["reliability_packetD_valid"] = kappa(D, "flag_valid(yes|no|uncertain)")
    D["valid"] = yes(D["flag_valid(yes|no|uncertain)"])
    D["incorp"] = D["flag_content_in_final_note(yes|partial|no)"].astype(str).str.strip().str.lower().isin(["yes", "partial"])
    cd = D.groupby("blinded_id").agg(valid=("valid", "sum"), incorp=("incorp", "sum")).reset_index()
    cd = cd.merge(pd.DataFrame(key["D"]), on="blinded_id")
    cd = cd[cd.encounter_id.isin(keep)]
    res["copri2_alert_ppv"] = cp(int((cd.valid >= 2).sum()), len(cd))
    res["copri2_acceptance_content_in_final_note"] = cp(int((cd.incorp >= 2).sum()), len(cd))
    res["copri2_override_recorded"] = {"overrides": 0, "note": "per-flag override counter not written (deviation log #7)"}

    # Descriptive: unresolved hazards per checkpoint in the final plan (Packet B)
    B = load(pdir, "packetB_final_plan")
    res["reliability_packetB_hazard"] = kappa(B, HAZ)
    res["reliability_packetB_insufficient"] = kappa(B, INSUF)
    cb = consensus(B).merge(pd.DataFrame(key["B"]), on="blinded_id")
    cb = cb[cb.encounter_id.isin(keep) & ~cb.insuf]
    res["unresolved_hazard_by_arm"] = {
        arm: {"encounters": int(len(x)), "encounters_with_hazard": cp(int(x.haz.sum()), len(x)),
              "hazards": float(x.nhaz.sum()), "checkpoints": float(x.nchk.sum()),
              "per_checkpoint": float(x.nhaz.sum() / x.nchk.sum()) if x.nchk.sum() else None}
        for arm, x in cb.groupby("arm")}
    res["unresolved_hazard_arm3_minus_arm2"] = boot_rate_diff(cb)

    # Secondary: 30-day hazard correction and referral completion (Packet C v2, referral-level)
    r1p = os.path.join(pdir, "PRIVATE_packetC_v2_referrals_reviewer1.csv")
    h1p = os.path.join(pdir, "PRIVATE_packetC_v2_hazards_reviewer1.csv")
    STAT = next((c for c in pd.read_csv(r1p, nrows=0).columns if c.startswith("status(")), None) if os.path.exists(r1p) else None
    def scored(df):
        return df[df.reviewer_initials.astype(str).str.strip() != ""]
    if STAT and len(scored(pd.read_csv(r1p, keep_default_na=False, dtype=str))):
        R1 = scored(pd.read_csv(r1p, keep_default_na=False, dtype=str))
        R1 = R1[R1.encounter_id.isin(keep)].merge(d[["encounter_id", "arm"]], on="encounter_id")
        inplan = R1[R1["referral_in_final_plan(yes|no)"].str.strip().str.lower().eq("yes")]
        setting = next(c for c in R1.columns if c.startswith("service_setting"))
        src = next(c for c in R1.columns if c.startswith("evidence_source"))
        elig = inplan.copy()
        st = elig[STAT].str.strip()
        # Primary (PI definition, 2026-10-20): the referral was acted on = a visit with the referred service,
        # a visit with a different service, or a scheduled appointment. Strict: a visit with the referred service.
        elig["done"] = st.isin(["completed", "visit_other_service", "scheduled_not_completed"])
        elig["done_strict"] = st.eq("completed")
        elig["done_documented"] = elig.done_strict & ~elig[src].str.strip().eq("patient_report_in_note")
        res["referrals_completed_30d_by_arm"] = {
            arm: {"referrals": int(len(x)), "encounters": int(x.encounter_id.nunique()),
                  "acted_on_primary": cp(int(x.done.sum()), len(x)),
                  "completed_referred_service_strict": cp(int(x.done_strict.sum()), len(x)),
                  "completed_strict_documented_only": cp(int(x.done_documented.sum()), len(x)),
                  "status_counts": x[STAT].str.strip().value_counts().to_dict()}
            for arm, x in elig.groupby("arm")}
        res["referrals_by_setting"] = elig.groupby([setting, "arm"]).done.agg(["sum", "count"]).reset_index().to_dict("records")
        # Sensitivity: referrals scored cannot_determine (late windows with claims lag) left out of the denominator.
        det = elig[~st.eq("cannot_determine")]
        res["referrals_acted_on_excl_cannot_determine_by_arm"] = {
            arm: cp(int(x.done.sum()), len(x)) for arm, x in det.groupby("arm")}
        res["referrals_acted_on_total"] = {"all": cp(int(elig.done.sum()), len(elig)),
                                           "excl_cannot_determine": cp(int(det.done.sum()), len(det))}
        res["referrals_acted_on_arm3_minus_arm2"] = {
            "all": boot_prop_diff(elig, "done"), "excl_cannot_determine": boot_prop_diff(det, "done")}
        if os.path.exists(h1p):
            H1 = scored(pd.read_csv(h1p, keep_default_na=False, dtype=str))
            H1 = H1[H1.encounter_id.isin(keep)].merge(d[["encounter_id", "arm"]], on="encounter_id")
            res["hazard_corrected_30d_by_arm"] = {arm: cp(int(yes(x["hazard_corrected_30d(yes|no)"]).sum()), len(x))
                                                  for arm, x in H1.groupby("arm")}
        r2p = os.path.join(pdir, "PRIVATE_packetC_v2_referrals_reviewer2.csv")
        if os.path.exists(r2p):
            R2 = scored(pd.read_csv(r2p, keep_default_na=False, dtype=str))
            j2 = R1.merge(R2, on="row_id", suffixes=("_1", "_2"))
            if len(j2):
                acted = ["completed", "visit_other_service", "scheduled_not_completed"]
                a1 = j2[STAT + "_1"].str.strip().isin(acted).astype(int)
                a2 = j2[STAT + "_2"].str.strip().isin(acted).astype(int)
                po = float((a1 == a2).mean())
                pe = float(a1.mean() * a2.mean() + (1 - a1.mean()) * (1 - a2.mean()))
                both = j2[j2["referral_in_final_plan(yes|no)_1"].str.lower().str.strip().eq("yes")
                          & j2["referral_in_final_plan(yes|no)_2"].str.lower().str.strip().eq("yes")]
                b1 = both[STAT + "_1"].str.strip().isin(acted).astype(int)
                b2 = both[STAT + "_2"].str.strip().isin(acted).astype(int)
                bpo = float((b1 == b2).mean())
                bpe = float(b1.mean() * b2.mean() + (1 - b1.mean()) * (1 - b2.mean()))
                res["packetC_interrater_confirmed_referrals"] = {
                    "n_referrals_both_in_plan": int(len(both)),
                    "n_encounters": int(both.encounter_id_1.nunique()),
                    "acted_on_pct_agreement": bpo,
                    "acted_on_cohen_kappa": (bpo - bpe) / (1 - bpe) if bpe < 1 else None,
                    "status_exact_agreement": float((both[STAT + "_1"].str.strip() == both[STAT + "_2"].str.strip()).mean())}
                res["packetC_interrater"] = {
                    "n_double_reviewed_rows": int(len(j2)),
                    "status_exact_agreement": float((j2[STAT + "_1"].str.strip() == j2[STAT + "_2"].str.strip()).mean()),
                    "in_plan_agreement": float((j2["referral_in_final_plan(yes|no)_1"].str.lower().str.strip()
                                                == j2["referral_in_final_plan(yes|no)_2"].str.lower().str.strip()).mean()),
                    "completed_cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else None,
                    "completed_pct_agreement": po}
    else:
        res["packetC"] = "not yet returned"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", default=os.path.join(RCT, "adjudication_packets_cite_outcomes"))
    ap.add_argument("--out", default=os.path.join(OUT, "cite_registered_outcomes.json"))
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(RCT, "derived", "analysis_dataset_wave11_locked.csv"), low_memory=False)
    link = pd.read_csv(os.path.join(OUT, "PRIVATE_wave11_30day_linkage.csv"))
    minors = set(link.loc[link.age_at_encounter < 18, "encounter_id"])
    allx = set(d.encounter_id)
    out = {"as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
           "primary_population_itt": analyze(d, a.packets, allx),
           "sensitivity_excluding_minors": analyze(d, a.packets, allx - minors),
           "n_minors_excluded": len(minors)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    p = out["primary_population_itt"]
    print(json.dumps({k: p[k] for k in ("copri1_escalate_only", "copri2_flags_per_encounter",
                                        "unresolved_hazard_arm3_minus_arm2")}, indent=1, default=float))


if __name__ == "__main__":
    main()
