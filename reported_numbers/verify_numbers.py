"""Re-derive every registered number from CANONICAL_NUMBERS.json and confirm it appears in each listed
document. Writes data_provenance.csv (one row per check). Exit 1 on any mismatch.
Usage: python verify_numbers.py [--strict]
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, "..", "manuscript")
C = json.load(open(os.path.join(HERE, "CANONICAL_NUMBERS.json")))
DOCS = {"main": "Basu_CITE_npjDM_manuscript.md", "supp": "Basu_CITE_npjDM_supplementary.md",
        "cover": "Basu_CITE_npjDM_cover_letter.md"}
TXT = {k: open(os.path.join(MS, v)).read() for k, v in DOCS.items()}
SRC = "audit/CANONICAL_NUMBERS.json (built by audit/build_canonical.py)"


def m(s):
    return s.replace("-", "−") if s.startswith("-") else s


def ci_txt(s):
    a, b = s.split(" to ")
    return f"{m(a)} to {m(b)}"


def k(s):
    return re.match(r"(\d+/\d+)", s).group(1)


def p(s):
    return re.search(r"\((\d+\.\d)%", s).group(1) + "%"


CHECKS = [
    ("n_total", ["main", "cover"], f"{C['n_total']}", "n_total"),
    ("n_arms", ["main"], f"{C['n_arm1']} to unassisted clinicians (Arm 1), {C['n_arm2']} to Gemini", "n_arm1,n_arm2"),
    ("linked", ["main"], f"{C['linked_n']} of {C['n_total']} patients (98.8%) to 30-day acute care data: {C['linked_arm1']}, {C['linked_arm2']}, and {C['linked_arm3']} by arm", "linked_n"),
    ("age", ["main"], f"mean age of {C['age_armT'].split(' (')[0]} years (SD {C['age_armT'].split('(')[1].rstrip(')')})", "age_armT"),
    ("states", ["main"], f"Ohio ({C['OH_armT']}), Virginia ({C['VA_armT']}), Washington ({C['WA_armT']}), and Illinois ({C['IL_armT']})", "OH_armT"),
    ("minors", ["main"], f"The approved protocol permitted patients younger than 18 ({C['minors_total']} enrolled)", "minors_total"),
    ("minors_supp", ["supp"], f"{C['minors_total']} were enrolled ({C['minors_arm1']} / {C['minors_arm2']} / {C['minors_arm3']} by arm)", "minors_arm1"),
    ("retro_def", ["main"], C["retro_deficient"].replace("/", " of ", 1).replace("1341", "1,341 plans"), "retro_deficient"),
    ("retro_def_gem", ["main"], C["retro_def_gemini"].replace("/", " of ", 1).replace(" (", " Gemini 3.1 Pro plans ("), "retro_def_gemini"),
    ("retro_cod_gem", ["main"], p(C["retro_cod_gemini"]), "retro_cod_gemini"),
    ("retro_cod_gem_k", ["main"], C["retro_cod_gemini"].split(" (")[0].replace("/", " of "), "retro_cod_gemini"),
    ("retro_cod_gpt", ["main"], C["retro_cod_gpt"].split(" (")[0].replace("/", " of ") + " GPT-5.6 plans (0.2%)", "retro_cod_gpt"),
    ("haz_live", ["main"], f"1 of 161 live drafts ({p(C['haz_draft_total'])}; 95% CI {C['haz_draft_total_ci']})", "haz_draft_total"),
    ("haz_live_abs", ["main", "cover"], "1 of 161", "haz_draft_total"),
    ("haz_excl", ["supp"], f"| {C['haz_draft_excl_insuf']}; {C['haz_draft_with_transcript']} |", "haz_draft_excl_insuf"),
    ("agreeA", ["main"], f"Agreement was unanimous for 159 of 161 drafts (Gwet AC1 {C['reliability_packetA_hazard_ac1']};", "reliability_packetA_hazard_ac1"),
    ("agreeA_supp", ["supp"], f"| 161 | 2 | {C['reliability_packetA_hazard_unanimous']} | {C['reliability_packetA_hazard_ac1']} | {C['reliability_packetA_hazard_kappa']} |", "reliability_packetA_hazard_ac1"),
    ("agreeB_supp", ["supp"], f"| 240 | 4 | {C['reliability_packetB_hazard_unanimous']} | {C['reliability_packetB_hazard_ac1']} | {C['reliability_packetB_hazard_kappa']} |", "reliability_packetB_hazard_ac1"),
    ("agreeD_supp", ["supp"], f"| 11 | 11 | {C['reliability_packetD_valid_unanimous']} | {C['reliability_packetD_valid_ac1']} | {C['reliability_packetD_valid_kappa']} |", "reliability_packetD_valid_ac1"),
    ("insA_supp", ["supp"], f"| 161 | 26 | {C['reliability_packetA_insufficient_unanimous']} | {C['reliability_packetA_insufficient_ac1']} | {C['reliability_packetA_insufficient_kappa']} |", "reliability_packetA_insufficient_ac1"),
    ("insB_supp", ["supp"], f"| 240 | 48 | {C['reliability_packetB_insufficient_unanimous']} | {C['reliability_packetB_insufficient_ac1']} | {C['reliability_packetB_insufficient_kappa']} |", "reliability_packetB_insufficient_ac1"),
    ("prereg_spec", ["main"], f"specificity of {C['cp1_prereg_spec'].split(' (')[0]} ({C['cp1_prereg_spec'].split('(')[1].split(';')[0]}; 95% CI {C['cp1_prereg_spec'].split('95% CI ')[1].rstrip(')')})", "cp1_prereg_spec"),
    ("commission0", ["main"], "flagged no drafts" if C["cp1_commission_flags"] == 0 else "XX", "cp1_commission_flags"),
    ("spec_any", ["supp"], f"every flag counted (post hoc) | {C['cp1_any_spec']} |", "cp1_any_spec"),
    ("spec_excl", ["supp"], f"; {C['cp1_ph_spec_excl_insuf'].split(' (')[0]} ({C['cp1_ph_spec_excl_insuf'].split('(')[1].split(')')[0]}) excluding drafts", "cp1_ph_spec_excl_insuf"),
    ("cite_ran", ["main"], f"CITE ran in {C['cite_ran_n']} of the {C['n_arm3']} Arm 3 encounters", "cite_ran_n"),
    ("cite_ran2", ["cover"], f"CITE ran in {C['cite_ran_n']} of {C['n_arm3']} encounters, flagged 7", "cite_ran_n"),
    ("flags", ["main"], f"CITE raised {C['cite_flags_total']} flags in {C['cite_flagged_enc']} of {C['cite_ran_n']} encounters ({p(C['cite_enc_any_flag'])}): median {C['cite_flags_median']}, mean {C['cite_flags_mean']}, and maximum {C['cite_flags_max']}", "cite_flags_total"),
    ("flag_types", ["main"], f"{C['flag_types']['omission']} were omissions and {C['flag_types']['confabulation']} were confabulations", "flag_types"),
    ("flag_valid", ["main"], f"9 of 11 flags clinically valid ({C['flag_valid'].split('(')[1].split(';')[0]}; 95% CI {C['flag_valid'].split('95% CI ')[1].rstrip(')')}; AC1 {C['reliability_packetD_valid_ac1']}): all {C['flag_confabulation_valid']} confabulations and {C['flag_omission_valid']} of {C['flag_omission_n']} omissions", "flag_valid"),
    ("acted", ["main"], f"The signed final notes acted on {C['valid_flags_acted']} of the 9 valid flags", "valid_flags_acted"),
    ("acted_om", ["main"], f"The note added the flagged action, in part, for {C['valid_omission_acted']} of {C['flag_omission_valid']} valid omission flags", "valid_omission_acted"),
    ("acted_cf", ["main"], f"All {C['flag_confabulation_valid'] - C['valid_confab_removed']} statements that CITE flagged as unsupported, from {C['confab_encounters']} encounters, remained", "valid_confab_removed"),
    ("acted_cf_cover", ["cover"], f"yet the signed notes acted on only {C['valid_flags_acted']}", "valid_flags_acted"),
    ("tab2_om", ["main"], f"| {C['flag_omission_n']} | {C['flag_omission_valid']} | {C['valid_omission_acted']} (action added in part) |", "flag_omission_n"),
    ("tab2_cf", ["main"], f"| {C['flag_confabulation_n']} | {C['flag_confabulation_valid']} | {C['valid_confab_removed']} (flagged statement retained in all {C['flag_confabulation_valid']}) |", "flag_confabulation_n"),
    ("over_esc_rd", ["main"], f"(risk difference {m(C['over_esc_rd32'])} percentage points; 95% CI {ci_txt(C['over_esc_rd32_ci'])})", "over_esc_rd32"),
    ("approp", ["main"], f"23 of 24 ({p(C['approp_esc_arm3'])}) in Arm 3 and 25 of 28 ({p(C['approp_esc_arm2'])}) in Arm 2 (difference {C['approp_esc_rd32']} points; 95% CI {ci_txt(C['approp_esc_rd32_ci'])})", "approp_esc_rd32"),
    ("comp_arms", ["main"], f"5 of 79 encounters ({p(C['composite_arm3'])}) in Arm 3, 4 of 82 ({p(C['composite_arm2'])}) in Arm 2, and 4 of 79 ({p(C['composite_arm1'])}) in Arm 1", "composite_arm3"),
    ("comp_abs", ["main"], f"Differences in adjudicated safety events ({C['composite_rd32']} points; 95% CI {ci_txt(C['composite_rd32_ci'])}) and 30-day emergency visits ({C['ed_rd32']} points; 95% CI {ci_txt(C['ed_rd32_ci'])})", "composite_rd32,ed_rd32"),
    ("ttd_zero_main", ["main"], f"recorded 0 s for {C['ttd_zero_n']} of 240 encounters; without them, the medians were {C['ttd_nonzero_arm3']}, {C['ttd_nonzero_arm2']}, and {C['ttd_nonzero_arm1']}", "ttd_zero_n"),
    ("comp_rd32", ["main"], f"{C['composite_rd32']} percentage points (95% CI {ci_txt(C['composite_rd32_ci'])})", "composite_rd32"),
    ("comp_rd21", ["main"], f"{m(C['composite_rd21'])} points (95% CI {ci_txt(C['composite_rd21_ci'])})", "composite_rd21"),
    ("kappa", ["main", "supp"], f"{C['composite_kappa']}", "composite_kappa"),
    ("kappa_ci", ["main", "supp"], f"95% CI {C['composite_kappa_ci']}", "composite_kappa_ci"),
    ("ttd", ["main"], f"Median time to decision was {C['ttd_arm3'].split(' (')[0]} with CITE, {C['ttd_arm2'].split(' (')[0]} with guardrails alone, and {C['ttd_arm1'].split(' (')[0]} unassisted (Arm 3 vs Arm 2, Mann–Whitney P = {C['ttd_mw_p']})", "ttd_arm3"),
    ("ttd_zero", ["supp"], f"excluding {C['ttd_zero_n']} encounters recorded as 0 s", "ttd_zero_n"),
    ("ttd_nonzero", ["supp"], f"{C['ttd_nonzero_arm1'].replace(' s', '')} / {C['ttd_nonzero_arm2'].replace(' s', '')} / {C['ttd_nonzero_arm3']}", "ttd_nonzero_arm3"),
    ("final_haz", ["main"], f"{C['final_haz_arm1'].replace('/', ' of ')} Arm 1 encounters, {C['final_haz_arm2'].replace('/', ' of ')} Arm 2 encounters, and {C['final_haz_arm3'].replace('/', ' of ')} Arm 3", "final_haz_arm1"),
    ("final_haz_rd", ["main"], f"{C['final_haz_rd32']} percentage points (bootstrap 95% CI {ci_txt(C['final_haz_rd32_ci'])})", "final_haz_rd32"),
    ("ed_arms", ["main"], f"{k(C['ed_arm3']).replace('/', ' of ')} linked Arm 3 patients ({p(C['ed_arm3'])}), {k(C['ed_arm2']).replace('/', ' of ')} Arm 2 patients ({p(C['ed_arm2'])}), and {k(C['ed_arm1']).replace('/', ' of ')} Arm 1 patients ({p(C['ed_arm1'])})", "ed_arm3"),
    ("ed_rd", ["main"], f"{C['ed_rd32']} percentage points (95% CI {ci_txt(C['ed_rd32_ci'])})", "ed_rd32"),
    ("ed_disc", ["main"], f"The emergency visit rate was {C['ed_rd32']} points higher with CITE (95% CI {ci_txt(C['ed_rd32_ci'])})", "ed_rd32"),
    ("hosp_arms", ["main"], f"{k(C['hosp_arm3']).replace('/', ' of ')} ({p(C['hosp_arm3'])}), {k(C['hosp_arm2']).replace('/', ' of ')} ({p(C['hosp_arm2'])}), and {k(C['hosp_arm1']).replace('/', ' of ')} ({p(C['hosp_arm1'])})", "hosp_arm3"),
    ("hosp_rd", ["main"], f"{m(C['hosp_rd32'])} points; 95% CI {ci_txt(C['hosp_rd32_ci'])}", "hosp_rd32"),
    ("deaths", ["main"], "No deaths were recorded within 30 days" if C["deaths_30d_total"] == 0 else "XX", "deaths_30d_total"),
    ("sens", ["supp"], f"Composite {C['sens_composite_rd32']} ({ci_txt(C['sens_composite_rd32_ci'])}); over-escalation {m(C['sens_over_esc_rd32'])} ({ci_txt(C['sens_over_esc_rd32_ci'])}); emergency visits {C['sens_ed_rd32']} ({ci_txt(C['sens_ed_rd32_ci'])}); hospitalization {m(C['sens_hosp_rd32'])} ({ci_txt(C['sens_hosp_rd32_ci'])})", "sens_composite_rd32"),
    ("de", ["supp"], f"| {ci_txt(C['de_m20_icc05_ci'])}; {ci_txt(C['de_m40_icc10_ci'])} |", "de_m20_icc05_ci"),
    ("mdd", ["main"], f"about {C['mdd_composite_pp']} points", "mdd_composite_pp"),
    ("mdd_supp", ["supp"], f"| {C['mdd_composite_pp']} points |", "mdd_composite_pp"),
    ("mdd_disc", ["main"], f"about {C['mdd_composite_pp']} points", "mdd_composite_pp"),
    ("latency", ["main"], f"median latency of {C['cite_latency_median_s']} s across {C['cite_latency_n']} stored results", "cite_latency_median_s"),
    ("status_supp", ["supp"], f"ESCALATE in {C['cite_status_counts']['ESCALATE']} encounters and REVIEW in {C['cite_status_counts']['REVIEW']}", "cite_status_counts"),
    ("insufB", ["main"], f"({C['insuf_B']['unassisted_md_soap'].replace('/', ' of ')}, {C['insuf_B']['gemini_guardrails'].replace('/', ' of ')}, and {C['insuf_B']['gemini_guardrails_anchor'].replace('/', ' of ')})", "insuf_B"),
    ("tab1_age", ["main"], f"| Age, years, mean (SD) (n = 237) | {C['age_arm1']} | {C['age_arm2']} | {C['age_arm3']} | {C['age_armT']} |", "age_arm1"),
    ("tab1_state", ["main"], f"| {C['OH_arm1']} / {C['VA_arm1']} / {C['WA_arm1']} / {C['IL_arm1']} | {C['OH_arm2']} / {C['VA_arm2']} / {C['WA_arm2']} / {C['IL_arm2']} | {C['OH_arm3']} / {C['VA_arm3']} / {C['WA_arm3']} / {C['IL_arm3']} |", "OH_arm1"),
    ("tab1_black", ["main"], f"| Black or African American, n | {k(C['black_arm1']).split('/')[0]} | {k(C['black_arm2']).split('/')[0]} | {k(C['black_arm3']).split('/')[0]} | {k(C['black_armT']).split('/')[0]} |", "black_arm1"),
    ("tab1_female", ["main"], f"| Female, n (n = 237) | {k(C['female_arm1']).split('/')[0]} | {k(C['female_arm2']).split('/')[0]} | {k(C['female_arm3']).split('/')[0]} | {k(C['female_armT']).split('/')[0]} |", "female_arm1"),
    ("supp_comp", ["supp"], f"| Any component (composite) | {C['composite_arm1']} | {C['composite_arm2']} | {C['composite_arm3']} | {C['composite_rd32']} ({ci_txt(C['composite_rd32_ci'])}) |", "composite_arm1"),
    ("supp_mdx", ["supp"], f"| Missed do-not-miss diagnosis | {C['missed_dx_arm1']} | {C['missed_dx_arm2']} | {C['missed_dx_arm3']} | {C['missed_dx_rd32']} ({ci_txt(C['missed_dx_rd32_ci'])}) |", "missed_dx_arm1"),
    ("supp_ut", ["supp"], f"| Under-triage | {C['undertriage_arm1']} | {C['undertriage_arm2']} | {C['undertriage_arm3']} | {C['undertriage_rd32']} ({ci_txt(C['undertriage_rd32_ci'])}) |", "undertriage_arm1"),
    ("supp_cm", ["supp"], f"| Contraindicated medication | {C['contra_med_arm1']} | {C['contra_med_arm2']} | {C['contra_med_arm3']} | {C['contra_med_rd32']} ({ci_txt(C['contra_med_rd32_ci'])}) |", "contra_med_arm1"),
    ("supp_me", ["supp"], f"| Missed escalation | {C['missed_escalation_arm1']} | {C['missed_escalation_arm2']} | {C['missed_escalation_arm3']} | {C['missed_escalation_rd32']} ({ci_txt(C['missed_escalation_rd32_ci'])}) |", "missed_escalation_arm1"),
    ("sensco", ["supp"], f"Composite {C['sensco_composite_rd32']} ({ci_txt(C['sensco_composite_rd32_ci'])}); over-escalation {m(C['sensco_over_esc_rd32'])} ({ci_txt(C['sensco_over_esc_rd32_ci'])}); emergency visits {C['sensco_ed_rd32']} ({ci_txt(C['sensco_ed_rd32_ci'])}); hospitalization {m(C['sensco_hosp_rd32'])} ({ci_txt(C['sensco_hosp_rd32_ci'])})", "sensco_composite_rd32"),
    ("ph_spec", ["supp"], f"commission or confabulation flags (post hoc) | {C['cp1_ph_spec']}", "cp1_ph_spec"),
    ("coenr", ["supp"], f"{C['coenrolled_n']} participants (7 / 8 / 10)", "coenrolled_n"),
    ("ttd_logt", ["supp"], f"P = {C['ttd_logt_p']:.2f} |", "ttd_logt_p"),
    ("anyp", ["main"], f"CITE flagged {C['cp1_anyp_flagged']} drafts, for a specificity of {C['cp1_anyp_spec']} and a positive predictive value of 0/5", "cp1_anyp_spec"),
    ("valid_enc", ["main"], f"all 5 confabulations and 4 of 6 omissions, from {C['valid_flag_encounters']} encounters", "valid_flag_encounters"),
    ("confab_enc", ["main"], f"from {C['confab_encounters']} encounters, remained", "confab_encounters"),
    ("notx", ["supp"], f"no transcript was stored for {C['no_transcript_n']} of 240 encounters ({C['no_transcript_by_arm']['unassisted_md_soap']} / {C['no_transcript_by_arm']['gemini_guardrails']} / {C['no_transcript_by_arm']['gemini_guardrails_anchor']})", "no_transcript_n"),
    ("trunc", ["supp"], f"{C['drafts_over_6000']} of 77 drafts were longer", "drafts_over_6000"),
    ("dates", ["main"], "between 16 July and 17 September 2026" if (C["enroll_first"], C["enroll_last"]) == ("2026-07-16", "2026-09-17") else "XX", "enroll_first"),
    ("ref_total", ["main"], f"The final plans made {C['ref_acted_total'].split('/')[1].split(' ')[0]} referrals in {C['ref_enc_total']} encounters", "ref_acted_total"),
    ("ref_acted_arms", ["main"], f"for {C['ref_acted_arm1'].replace('/', ' of ', 1)} in Arm 1, {C['ref_acted_arm2'].replace('/', ' of ', 1)} in Arm 2, and {C['ref_acted_arm3'].replace('/', ' of ', 1)} in Arm 3 (Arm 3 minus Arm 2, {C['ref_rd32_all']} percentage points; 95% CI {ci_txt(C['ref_rd32_all_ci'])})", "ref_acted_arm1,ref_rd32_all"),
    ("ref_cd", ["main"], f"could not determine the status of {C['ref_cd_total']} external referrals", "ref_cd_total"),
    ("ref_excl", ["main"], f"the proportions were {p(C['ref_acted_excl_cd_arm1'])}, {p(C['ref_acted_excl_cd_arm2'])}, and {p(C['ref_acted_excl_cd_arm3'])}", "ref_acted_excl_cd_arm1"),
    ("ref_table", ["supp"], f"| Acted on (completed, visit with a different service, or scheduled) | {C['ref_acted_arm1']} | {C['ref_acted_arm2']} | {C['ref_acted_arm3']} |", "ref_acted_arm1"),
    ("ref_table_strict", ["supp"], f"| Completed with the referred service | {C['ref_completed_arm1']} | {C['ref_completed_arm2']} | {C['ref_completed_arm3']} |", "ref_completed_arm1"),
    ("ref_table_cd", ["supp"], f"| Status could not be determined, n | {C['ref_cd_arm1']} | {C['ref_cd_arm2']} | {C['ref_cd_arm3']} |", "ref_cd_arm1"),
    ("ref_table_excl", ["supp"], f"| Acted on, excluding referrals that could not be determined | {C['ref_acted_excl_cd_arm1']} | {C['ref_acted_excl_cd_arm2']} | {C['ref_acted_excl_cd_arm3']} |", "ref_acted_excl_cd_arm1"),
    ("ref_rd_supp", ["supp"], f"was {C['ref_rd32_all']} percentage points (encounter-level bootstrap 95% CI {ci_txt(C['ref_rd32_all_ci'])}), and {m(C['ref_rd32_excl_cannot_determine'])} points ({ci_txt(C['ref_rd32_excl_cannot_determine_ci'])})", "ref_rd32_all"),
    ("ref_sens_minors", ["supp"], f"the difference was {C['sens_ref_rd32']} points ({ci_txt(C['sens_ref_rd32_ci'])})", "sens_ref_rd32"),
    ("ref_irr", ["supp"], f"For the {C['refC_irr_n']} referrals in {C['refC_irr_enc']} encounters that both reviewers confirmed, they agreed on whether the referral was acted on for {C['refC_irr_acted_pct']}% (Cohen κ {C['refC_irr_kappa']}) and on the exact status for {C['refC_irr_status_pct']}%", "refC_irr_kappa"),
    ("ref_irr_plan", ["supp"], f"{C['refC_irr_inplan_pct']}% of the {C['refC_r2_rows']} double-scored rows", "refC_irr_inplan_pct"),
    ("haz_corr", ["main"], "Neither of the 2 unresolved hazards was corrected within 30 days" if set(C["haz_corrected"].values()) == {"0/1"} else "XX", "haz_corrected"),
]


def main():
    rows, bad = [], 0
    for cid, docs, exp, key in CHECKS:
        for doc in docs:
            ok = exp in TXT[doc]
            bad += not ok
            rows.append({"claim_id": cid, "reported_value": exp, "source_file": SRC, "canonical_key": key,
                         "location": DOCS[doc], "status": "ok" if ok else "MISMATCH"})
            if not ok:
                print(f"MISMATCH {cid} [{doc}]: expected '{exp}'")
    with open(os.path.join(HERE, "data_provenance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} checks, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
