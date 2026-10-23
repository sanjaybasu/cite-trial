"""Build CANONICAL_NUMBERS.json / .md for the CITE npj Digital Medicine paper.

Every value is loaded from an upstream result file; nothing is hard-coded except protocol constants
(the 5-point over-escalation margin and the median-3 alert ceiling, both fixed in the protocol).
Upstream sources (absolute paths below):
  trial  rct_analysis/results/wave11_locked/{canonical.json, cite_paper_extras.json,
         cite_registered_outcomes.json, exploratory_30day_wave11.json, ctgov_results_aggregates.json,
         sensitivity_excluding_minors.json}; rct_analysis/derived/analysis_dataset_wave11_locked.csv
  retro  ebm-verify/study/run/adjudication_powered/{consensus_completed.csv, UNBLINDING_KEY.json}
Run:  python build_canonical.py   (writes alongside this file)
"""
import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
W = "/Users/sanjaybasu/waymark-local/notebooks/anchor/rct_analysis/results/wave11_locked"
D = "/Users/sanjaybasu/waymark-local/notebooks/anchor/rct_analysis/derived/analysis_dataset_wave11_locked.csv"
R = "/Users/sanjaybasu/waymark-local/notebooks/ebm-verify/study/run/adjudication_powered"
A1, A2, A3 = "unassisted_md_soap", "gemini_guardrails", "gemini_guardrails_anchor"
OVER_ESC_MARGIN_PP = 5.0     # protocol (WCG modification of 20261776)
ALERT_CEILING = 3            # pre-registered co-primary 2 ceiling (median flags per encounter)


def J(name):
    return json.load(open(os.path.join(W, name)))


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def frac(k, n):
    return f"{k}/{n} ({pct(k / n)}%)" if n else f"{k}/{n}"


def ci(lo, hi, d=1):
    if lo is None or hi is None:
        return "not estimable"
    return f"{100 * lo:.{d}f} to {100 * hi:.{d}f}"


def main():
    C = {}
    canon, ex, reg, e30, agg, sens = (J("canonical.json"), J("cite_paper_extras.json"), J("cite_registered_outcomes.json"),
                                      J("exploratory_30day_wave11.json"), J("ctgov_results_aggregates.json"),
                                      J("sensitivity_excluding_minors.json"))
    d = pd.read_csv(D, low_memory=False)
    t = pd.to_datetime(d.encounter_timestamp, format="mixed")
    C["enroll_first"], C["enroll_last"] = str(t.min().date()), str(t.max().date())
    C["n_total"] = int(len(d))
    for a, k in ((A1, "n_arm1"), (A2, "n_arm2"), (A3, "n_arm3")):
        C[k] = int((d.arm == a).sum())
    C["n_patients"] = int(d.patient_key.nunique())

    # primary composite (exploratory in NCT07718893) and components
    for col, key in (("primary_event_final", "composite"), ("adj_missed_dx", "missed_dx"), ("adj_undertriage", "undertriage"),
                     ("adj_contraindicated_med", "contra_med"), ("adj_missed_escalation", "missed_escalation"),
                     ("over_escalation_adjudicated", "over_esc"), ("appropriate_escalation_adjudicated", "approp_esc")):
        o = ex[col]
        for a, s in ((A1, "1"), (A2, "2"), (A3, "3")):
            b = o["by_arm"][a]
            C[f"{key}_arm{s}"] = frac(b["k"], b["n"])
            C[f"{key}_arm{s}_k"], C[f"{key}_arm{s}_n"] = b["k"], b["n"]
        for cmp, lab in (("arm3_vs_arm2", "32"), ("arm2_vs_arm1", "21")):
            r = o[cmp]
            C[f"{key}_rd{lab}"] = f"{100 * r['rd']:.1f}"
            C[f"{key}_rd{lab}_ci"] = ci(r["lo"], r["hi"])
            C[f"{key}_rd{lab}_p"] = round(r["fisher_p"], 2)
    C["over_esc_margin_pp"] = OVER_ESC_MARGIN_PP
    C["over_esc_upper_below_margin"] = ex["over_escalation_upper_bound_below_margin"]
    C["composite_kappa"] = f"{canon['adjudicator_reliability']['fleiss_kappa']:.3f}"
    C["composite_kappa_ci"] = f"{canon['adjudicator_reliability']['ci95_low']:.3f} to {canon['adjudicator_reliability']['ci95_high']:.3f}"
    C["mdd_composite_pp"] = f"{100 * ex['mdd_composite_increase_80pct_power']['mdd']:.1f}"
    de = [s for s in ex["design_effect_scenarios"] if s["cluster_size"] == 20 and s["icc"] == 0.05][0]
    C["de_m20_icc05_ci"] = ci(de["ci_lo"], de["ci_hi"])
    de40 = [s for s in ex["design_effect_scenarios"] if s["cluster_size"] == 40 and s["icc"] == 0.10][0]
    C["de_m40_icc10_ci"] = ci(de40["ci_lo"], de40["ci_hi"])

    # time to decision (protocol-specified secondary)
    tt = ex["time_to_decision"]
    for a, s in ((A1, "1"), (A2, "2"), (A3, "3")):
        C[f"ttd_arm{s}"] = f"{tt[a]['median']:.1f} s (IQR {tt[a]['q1']:.1f}-{tt[a]['q3']:.1f})"
    C["ttd_mw_p"] = round(tt["mannwhitney_p_arm3_vs_arm2"], 2)
    C["ttd_logt_p"] = round(tt["log_ttest_p_arm3_vs_arm2"], 2)
    tsec = pd.to_numeric(d.time_to_decision_seconds, errors="coerce")
    C["ttd_zero_n"] = int((tsec == 0).sum())
    for a, s in ((A1, "1"), (A2, "2"), (A3, "3")):
        v = tsec[(d.arm == a) & (tsec > 0)]
        C[f"ttd_nonzero_arm{s}"] = f"{v.median():.1f} s"

    # registered co-primaries
    P = reg["primary_population_itt"]
    for op, lab in (("copri1_escalate_only", "esc"), ("copri1_any_flag", "any")):
        q = P[op]
        C[f"cp1_{lab}_tp"], C[f"cp1_{lab}_fp"], C[f"cp1_{lab}_tn"], C[f"cp1_{lab}_fn"] = q["tp"], q["fp"], q["tn"], q["fn"]
        sp = q["specificity"]
        C[f"cp1_{lab}_spec"] = f"{sp['k']}/{sp['n']} ({pct(sp['est'])}%; 95% CI {ci(sp['lo'], sp['hi'])})"
        C[f"cp1_{lab}_flagged"] = q["tp"] + q["fp"]
        C[f"cp1_{lab}_ppv"] = f"{q['ppv']['k']}/{q['ppv']['n']}"
    prev = P["hazard_prevalence_ai_draft"]
    C["haz_draft_arm2"] = f"{prev[A2]['k']}/{prev[A2]['n']}"
    C["haz_draft_arm3"] = f"{prev[A3]['k']}/{prev[A3]['n']}"
    kd = prev[A2]["k"] + prev[A3]["k"]
    nd = prev[A2]["n"] + prev[A3]["n"]
    C["haz_draft_total"] = frac(kd, nd)
    lo, hi = proportion_confint(kd, nd, method="beta")
    C["haz_draft_total_ci"] = ci(lo, hi)
    for k in ("reliability_packetA_hazard", "reliability_packetB_hazard", "reliability_packetD_valid",
              "reliability_packetA_insufficient", "reliability_packetB_insufficient"):
        C[k + "_ac1"] = f"{P[k]['gwet_ac1']:.3f}"
        C[k + "_kappa"] = f"{P[k]['fleiss_kappa']:.3f}"
        C[k + "_unanimous"] = f"{pct(P[k]['pct_unanimous'])}%"
    f = P["copri2_flags_per_encounter"]
    C["cite_ran_n"], C["cite_flags_total"] = f["n_encounters"], f["total_flags"]
    C["cite_flags_median"], C["cite_flags_mean"] = f"{f['median']:.0f}", f"{f['mean']:.2f}"
    C["alert_ceiling"], C["alert_ceiling_met"] = ALERT_CEILING, f["median"] <= ALERT_CEILING
    C["cite_flagged_enc"] = int((d[d.arm == A3].cite_flagged == "t").sum())
    v, inc = P["copri2_alert_ppv"], P["copri2_acceptance_content_in_final_note"]
    C["flag_valid"] = f"{v['k']}/{v['n']} ({pct(v['est'])}%; 95% CI {ci(v['lo'], v['hi'])})"
    C["flag_incorporated"] = f"{inc['k']}/{inc['n']} ({pct(inc['est'])}%; 95% CI {ci(inc['lo'], inc['hi'])})"
    types = {}
    for s in d[d.cite_flagged == "t"].cite_flags:
        for fl in json.loads(s):
            types[fl["type"]] = types.get(fl["type"], 0) + 1
    C["flag_types"] = types
    PK = "/Users/sanjaybasu/waymark-local/notebooks/anchor/rct_analysis/adjudication_packets_cite_outcomes"
    Dd = pd.concat([pd.read_csv(os.path.join(PK, r, f"packetD_cite_flags_{r}.csv"), keep_default_na=False, dtype=str)
                    for r in ("adj1", "adj2", "adj3")])
    Dd["v"] = Dd["flag_valid(yes|no|uncertain)"].str.strip().str.lower().eq("yes")
    Dd["i"] = Dd["flag_content_in_final_note(yes|partial|no)"].str.strip().str.lower().isin(["yes", "partial"])
    g = Dd.groupby(["blinded_id", "flag_type"]).agg(v=("v", "sum"), i=("i", "sum")).reset_index()
    for ft, x in g.groupby("flag_type"):
        C[f"flag_{ft}_n"], C[f"flag_{ft}_valid"], C[f"flag_{ft}_incorp"] = int(len(x)), int((x.v >= 2).sum()), int((x.i >= 2).sum())
    # flag-level action: omission flags are acted on when the flagged action is added to the signed note;
    # confabulation flags are acted on when the flagged statement is removed (i.e., NOT retained).
    vv = g.assign(V=g.v >= 2)
    inc_maj = Dd.groupby("blinded_id")["flag_content_in_final_note(yes|partial|no)"].agg(
        lambda x: max(set(x.str.strip().str.lower()), key=list(x.str.strip().str.lower()).count))
    vv["inc"] = vv.blinded_id.map(inc_maj)
    om = vv[(vv.flag_type == "omission") & vv.V]
    cf = vv[(vv.flag_type == "confabulation") & vv.V]
    C["valid_omission_acted"] = int(om.inc.isin(["yes", "partial"]).sum())
    C["valid_omission_acted_full"] = int((om.inc == "yes").sum())
    C["valid_confab_removed"] = int((cf.inc == "no").sum())
    C["valid_flags_acted"] = C["valid_omission_acted"] + C["valid_confab_removed"]
    # pre-registered ESCALATE-only = commission (or allergy conflict / refuted numeric; not separable types in cite-1.0)
    a3r = d[(d.arm == A3) & (d.cite_used == "t")]
    comm = a3r.cite_flags.fillna("[]").map(lambda s: any(f["type"] == "commission" for f in json.loads(s)) if s.strip() else False)
    lo, hi = proportion_confint(int((~comm).sum()), len(a3r), method="beta")
    C["cp1_prereg_spec"] = f"{int((~comm).sum())}/{len(a3r)} ({pct((~comm).mean())}%; 95% CI {ci(lo, hi)})"
    C["cp1_commission_flags"] = int(comm.sum())
    for op, lab in (("copri1_escalate_only", "esc"), ("copri1_any_flag", "any")):
        q = P[op]["npv"]
        C[f"cp1_{lab}_npv"] = f"{q['k']}/{q['n']} ({pct(q['est'])}%; 95% CI {ci(q['lo'], q['hi'])})"
        q = P[op]["ppv"]
        C[f"cp1_{lab}_ppv_ci"] = f"{q['k']}/{q['n']} (95% CI {ci(q['lo'], q['hi'])})"
    q = P["copri1_posthoc_commission_or_confabulation"]["specificity"]
    C["cp1_ph_spec"] = f"{q['k']}/{q['n']} ({pct(q['est'])}%; 95% CI {ci(q['lo'], q['hi'])})"
    q = P["copri1_excl_insufficient_posthoc_commission_or_confabulation"]["specificity"]
    C["cp1_ph_spec_excl_insuf"] = f"{q['k']}/{q['n']} ({pct(q['est'])}%; 95% CI {ci(q['lo'], q['hi'])})"
    q = P["copri1_any_flag_prereg"]
    C["cp1_anyp_flagged"] = q["tp"] + q["fp"]
    C["cp1_anyp_spec"] = f"{q['specificity']['k']}/{q['specificity']['n']} ({pct(q['specificity']['est'])}%; 95% CI {ci(q['specificity']['lo'], q['specificity']['hi'])})"
    C["cp1_anyp_ppv"] = f"{q['ppv']['k']}/{q['ppv']['n']} (95% CI {ci(q['ppv']['lo'], q['ppv']['hi'])})"
    C["cp1_anyp_npv"] = f"{q['npv']['k']}/{q['npv']['n']}"
    C["cp1_anyp_npv_ci"] = f"{pct(q['npv']['est'])}%; 95% CI {ci(q['npv']['lo'], q['npv']['hi'])}"
    # encounter-level clustering of flags
    keyD = pd.DataFrame(json.load(open(os.path.join(PK, "PRIVATE_key_map.json")))["D"])
    vv2 = vv.merge(keyD, on="blinded_id")
    C["valid_flag_encounters"] = int(vv2[vv2.V].encounter_id.nunique())
    C["confab_encounters"] = int(vv2[vv2.flag_type == "confabulation"].encounter_id.nunique())
    C["max_flags_one_encounter_confab"] = int(vv2[vv2.flag_type == "confabulation"].groupby("encounter_id").size().max())
    tch = pd.to_numeric(d.transcription_chars, errors="coerce").fillna(0)
    C["no_transcript_n"] = int((tch == 0).sum())
    C["no_transcript_by_arm"] = {a: int(((tch == 0) & (d.arm == a)).sum()) for a in (A1, A2, A3)}
    C["no_transcript_cite_ran"] = int(((tch == 0) & (d.arm == A3) & (d.cite_used == "t")).sum())
    KA = pd.DataFrame(json.load(open(os.path.join(PK, "PRIVATE_key_map.json")))["A"])
    LA = pd.concat([pd.read_csv(os.path.join(PK, r, f"packetA_ai_draft_{r}.csv"), keep_default_na=False, dtype=str) for r in ("adj1", "adj2", "adj3")])
    ha = LA.assign(h=LA["hazard_present(yes|no)"].str.strip().str.lower().eq("yes")).groupby("blinded_id").h.sum() >= 2
    KA["haz"] = KA.blinded_id.map(ha)
    KA["has_tx"] = KA.encounter_id.map(dict(zip(d.encounter_id, tch > 0)))
    C["haz_draft_with_transcript"] = f"{int(KA[KA.has_tx].haz.sum())}/{int(KA.has_tx.sum())}"
    C["escalation_field_set"] = int((d.escalation_flag == "t").sum())
    C["cite_status_counts"] = a3r.cite_status.value_counts().to_dict()
    C["drafts_over_6000"] = int((a3r.gemini_response.fillna("").str.len() > 6000).sum())
    C["cite_enc_any_flag"] = frac(int((a3r.cite_flagged == "t").sum()), len(a3r))
    C["cite_flags_max"] = int(pd.to_numeric(a3r.cite_n_flags, errors="coerce").max())
    # ED among flagged vs unflagged Arm 3 encounters (descriptive)
    L = pd.read_csv(os.path.join(W, "PRIVATE_wave11_30day_linkage.csv"))
    a3l = a3r.merge(L[["encounter_id", "ed_30d"]], on="encounter_id")
    fl_ = a3l[a3l.cite_flagged == "t"].ed_30d.dropna(); nf_ = a3l[a3l.cite_flagged != "t"].ed_30d.dropna()
    C["ed_arm3_flagged"] = f"{int(fl_.sum())}/{len(fl_)}"
    C["ed_arm3_unflagged"] = f"{int(nf_.sum())}/{len(nf_)}"
    tz = pd.to_numeric(d.time_to_decision_seconds, errors="coerce")
    C["ttd_zero_by_arm"] = {a: int(((tz == 0) & (d.arm == a)).sum()) for a in (A1, A2, A3)}
    st = pd.read_csv("/Users/sanjaybasu/waymark-local/notebooks/anchor/rct_analysis/derived/PRIVATE_v2_final_240_as_stored.csv",
                     usecols=["cite_certificate_stored"], low_memory=False).cite_certificate_stored.dropna().map(json.loads)
    lat = st.map(lambda x: x.get("latency_ms")).dropna()
    C["cite_latency_median_s"], C["cite_latency_n"] = f"{lat.median() / 1000:.1f}", int(len(lat))
    nins = ex["insufficient_content_packetA"]
    C["haz_draft_excl_insuf"] = f"{kd}/{nd - nins[A2]['k'] - nins[A3]['k']}"
    xb = ex["insufficient_content_packetB"]
    C["insuf_B"] = {a: f"{xb[a]['k']}/{xb[a]['n']}" for a in (A1, A2, A3)}
    u = P["unresolved_hazard_by_arm"]
    for a, s in ((A1, "1"), (A2, "2"), (A3, "3")):
        C[f"final_haz_arm{s}"] = f"{u[a]['encounters_with_hazard']['k']}/{u[a]['encounters']}"
        C[f"final_chk_arm{s}"] = int(u[a]["checkpoints"])
    r = P["unresolved_hazard_arm3_minus_arm2"]
    C["final_haz_rd32"] = f"{100 * r['rd_per_checkpoint']:.1f}"
    C["final_haz_rd32_ci"] = ci(r["lo"], r["hi"])
    C["packetC_status"] = P.get("packetC", "returned")
    # 30-day referral follow-through and hazard correction (secondary; Packet C v2 chart review)
    rc = P["referrals_completed_30d_by_arm"]
    rx = P["referrals_acted_on_excl_cannot_determine_by_arm"]
    for a, s in ((A1, "1"), (A2, "2"), (A3, "3")):
        C[f"ref_n_arm{s}"] = rc[a]["referrals"]
        C[f"ref_enc_arm{s}"] = rc[a]["encounters"]
        C[f"ref_acted_arm{s}"] = frac(rc[a]["acted_on_primary"]["k"], rc[a]["acted_on_primary"]["n"])
        C[f"ref_completed_arm{s}"] = frac(rc[a]["completed_referred_service_strict"]["k"], rc[a]["completed_referred_service_strict"]["n"])
        C[f"ref_cd_arm{s}"] = int(rc[a]["status_counts"].get("cannot_determine", 0))
        C[f"ref_acted_excl_cd_arm{s}"] = frac(rx[a]["k"], rx[a]["n"])
    t = P["referrals_acted_on_total"]
    C["ref_acted_total"] = frac(t["all"]["k"], t["all"]["n"])
    C["ref_acted_total_excl_cd"] = frac(t["excl_cannot_determine"]["k"], t["excl_cannot_determine"]["n"])
    C["ref_cd_total"] = t["all"]["n"] - t["excl_cannot_determine"]["n"]
    C["ref_enc_total"] = sum(rc[a]["encounters"] for a in (A1, A2, A3))
    for v in ("all", "excl_cannot_determine"):
        r = P["referrals_acted_on_arm3_minus_arm2"][v]
        C[f"ref_rd32_{v}"] = f"{100 * r['rd']:.1f}"
        C[f"ref_rd32_{v}_ci"] = ci(r["lo"], r["hi"])
    hc = P["hazard_corrected_30d_by_arm"]
    C["haz_corrected"] = {a: f"{hc[a]['k']}/{hc[a]['n']}" for a in hc}
    ir = P["packetC_interrater_confirmed_referrals"]
    C["refC_irr_n"] = ir["n_referrals_both_in_plan"]
    C["refC_irr_enc"] = ir["n_encounters"]
    C["refC_irr_acted_pct"] = pct(ir["acted_on_pct_agreement"])
    C["refC_irr_kappa"] = f"{ir['acted_on_cohen_kappa']:.3f}"
    C["refC_irr_status_pct"] = pct(ir["status_exact_agreement"])
    C["refC_irr_inplan_pct"] = pct(P["packetC_interrater"]["in_plan_agreement"])
    C["refC_r2_rows"] = P["packetC_interrater"]["n_double_reviewed_rows"]
    sm = reg["sensitivity_excluding_minors"]["referrals_acted_on_arm3_minus_arm2"]["all"]
    C["sens_ref_rd32"] = f"{100 * sm['rd']:.1f}"
    C["sens_ref_rd32_ci"] = ci(sm["lo"], sm["hi"])

    # 30-day acute care (exploratory) and adverse events
    for oc, key in (("ed_30d", "ed"), ("hosp_30d", "hosp")):
        o = e30["outcomes"][oc]
        for a, s in ((A1, "1"), (A2, "2"), (A3, "3")):
            b = o["by_arm"][a]
            C[f"{key}_arm{s}"] = frac(b["events"], b["n"])
        C[f"{key}_rd32"] = f"{100 * o['arm3_vs_arm2']['rd']:.1f}"
        C[f"{key}_rd32_ci"] = ci(o["arm3_vs_arm2"]["ci_low"], o["arm3_vs_arm2"]["ci_high"])
        C[f"{key}_rd21"] = f"{100 * o['arm2_vs_arm1']['rd']:.1f}"
        C[f"{key}_rd21_ci"] = ci(o["arm2_vs_arm1"]["ci_low"], o["arm2_vs_arm1"]["ci_high"])
    C["linked_n"] = e30["linked_n"]
    for a, s_ in ((A1, "1"), (A2, "2"), (A3, "3")):
        C[f"linked_arm{s_}"] = e30["outcomes"]["ed_30d"]["by_arm"][a]["n"]
    C["deaths_30d_total"] = agg["ae"]["total"]["deaths_30d"]

    # baseline
    for a, s in ((A1, "1"), (A2, "2"), (A3, "3"), ("total", "T")):
        b = agg["baseline"][a]
        C[f"age_arm{s}"] = f"{b['age_mean']:.1f} ({b['age_sd']:.1f})"
        C[f"female_arm{s}"] = frac(b["sex"].get("Female", 0), b["n"])
        C[f"minors_arm{s}"] = b["age_cat"].get("<18", 0)
        C[f"black_arm{s}"] = frac(b["race"].get("Black or African American", 0), b["n"])
        for st in ("OH", "VA", "WA", "IL"):
            C[f"{st}_arm{s}"] = b["state"].get(st, 0)
    C["minors_total"] = agg["baseline"]["total"]["age_cat"].get("<18", 0)

    # sensitivity excluding minors
    for oc, key in (("primary_event_final", "composite"), ("over_escalation_adjudicated", "over_esc"),
                    ("ed_30d", "ed"), ("hosp_30d", "hosp")):
        x = sens[oc]["arm3_minus_arm2"]
        C[f"sens_{key}_rd32"] = f"{x[0]:.1f}"
        C[f"sens_{key}_rd32_ci"] = f"{x[1]:.1f} to {x[2]:.1f}"
    sc = J("sensitivity_excluding_coenrolled.json")
    C["coenrolled_n"] = sc["excluded_coenrolled"]
    for oc, key in (("primary_event_final", "composite"), ("over_escalation_adjudicated", "over_esc"), ("ed_30d", "ed"), ("hosp_30d", "hosp")):
        C[f"sensco_{key}_rd32"] = f"{sc[oc][0]:.1f}"
        C[f"sensco_{key}_rd32_ci"] = f"{sc[oc][1]:.1f} to {sc[oc][2]:.1f}"
    C["insuf_A_arm2"] = f"{ex['insufficient_content_packetA'][A2]['k']}/{ex['insufficient_content_packetA'][A2]['n']}"
    C["insuf_A_arm3"] = f"{ex['insufficient_content_packetA'][A3]['k']}/{ex['insufficient_content_packetA'][A3]['n']}"

    # retrospective study (AI-drafted ratings, each reviewed and approved by clinicians)
    c = pd.read_csv(os.path.join(R, "consensus_completed.csv"), keep_default_na=False, dtype=str)
    key = json.load(open(os.path.join(R, "UNBLINDING_KEY.json")))
    c["model"] = c.plan_id.map(lambda p: key[p]["base_model"])
    c["def"] = c.primary_deficient.astype(int)
    c["cod"] = (c["def"] == 1) & c.consensus_codifiable.str.contains("yes")
    C["retro_n_plans"], C["retro_n_models"] = int(len(c)), int(c.model.nunique())
    C["retro_per_model"] = int(c.groupby("model").size().iloc[0])
    C["retro_deficient"] = frac(int(c["def"].sum()), len(c))
    for m, lab in (("gemini-3.1-pro-preview", "gemini"), ("gpt-5.6", "gpt"), ("claude-opus-4-8", "claude")):
        s = c[c.model == m]
        C[f"retro_def_{lab}"] = frac(int(s["def"].sum()), len(s))
        C[f"retro_cod_{lab}"] = frac(int(s["cod"].sum()), len(s))
    C["retro_cod_among_def"] = frac(int(c["cod"].sum()), int(c["def"].sum()))
    C["retro_strata"] = c.plan_id.map(lambda p: key[p].get("stratum")).value_counts().to_dict()

    with open(os.path.join(HERE, "CANONICAL_NUMBERS.json"), "w") as fh:
        json.dump(C, fh, indent=1, default=str)
    with open(os.path.join(HERE, "CANONICAL_NUMBERS.md"), "w") as fh:
        fh.write("# CANONICAL_NUMBERS (generated by audit/build_canonical.py; do not edit by hand)\n\n| key | value |\n|---|---|\n")
        for k, v in C.items():
            fh.write(f"| {k} | {v} |\n")
    print(f"{len(C)} canonical values written")


if __name__ == "__main__":
    main()
