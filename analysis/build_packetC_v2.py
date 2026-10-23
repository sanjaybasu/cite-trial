"""Packet C v2: referral-level 30-day chart review sheets (replaces the encounter-level v1 sheet).

One row per candidate referral (from the three adjudicators' Packet B referral types, normalized and
deduplicated) plus one blank 'other' row per encounter. Hazard-correction rows are on a separate sheet.
The second reviewer gets every row for the same 16 sampled encounters. Writes PRIVATE files only.
"""
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.join(HERE, "..", "adjudication_packets_cite_outcomes")
RATERS = ("adj1", "adj2", "adj3")
SYN = [
    (r"physical therap|\bpt\b", "Physical therapy"),
    (r"occupational therap|\bot\b", "Occupational therapy"),
    (r"speech", "Speech therapy"),
    (r"pharm", "Pharmacist (internal)"),
    (r"therap|counsel|psychotherap", "Therapist (internal)"),
    (r"social work|case manag|care manag|chw|community health", "Social work / case management"),
    (r"behavio|mental|psychiatr", "Behavioral health / psychiatry"),
    (r"primary care|pcp", "Primary care"),
]
REV_COLS = [
    "referral_in_final_plan(yes|no)",
    "service_setting(internal_pharmacist|internal_therapist|internal_waymark_chw|external)",
    "status(completed|visit_other_service|scheduled_not_completed|not_scheduled|declined_or_unreachable|cannot_determine|not_applicable)",
    "evidence_source(claims|note_by_referred_service|consult_report|scheduling_record|patient_report_in_note|none)",
    "evidence_date(YYYY-MM-DD)",
    "evidence_text(short quote or claim line)",
    "reviewer_initials",
    "notes",
]


def norm(t):
    t = re.sub(r"^\s*adj\d\s*:\s*", "", t.strip(), flags=re.I).strip(" .")
    for pat, lab in SYN:
        if re.search(pat, t, re.I):
            return lab
    return t[:1].upper() + t[1:] if t else ""


def main():
    key = pd.DataFrame(json.load(open(os.path.join(PK, "PRIVATE_key_map.json")))["B"])
    B = pd.concat([pd.read_csv(os.path.join(PK, r, f"packetB_final_plan_{r}.csv"), keep_default_na=False, dtype=str)
                   for r in RATERS])
    tcol = next(c for c in B.columns if c.startswith("referral_types"))
    B = B.merge(key[["blinded_id", "encounter_id"]], on="blinded_id")
    cand = {}
    for e, v in zip(B.encounter_id, B[tcol]):
        for part in re.split(r"[;|]", v):
            lab = norm(part)
            if lab:
                cand.setdefault(e, {}).setdefault(lab.lower(), [lab, 0])[1] += 1
    c = pd.read_csv(os.path.join(PK, "PRIVATE_packetC_30day_chart_review.csv"), keep_default_na=False, dtype=str)
    rows = []
    for _, r in c.iterrows():
        base = {"encounter_id": r.encounter_id, "patient_key": r.patient_key, "chart_extract_row": r.chart_extract_row,
                "encounter_date": r.encounter_timestamp[:10], "window_end": r.window_end}
        for lab, n in sorted(cand.get(r.encounter_id, {}).values(), key=lambda x: -x[1]):
            rows.append({**base, "candidate_referral": lab, "listed_by_n_adjudicators": n})
        rows.append({**base, "candidate_referral": "OTHER (add only if the final plan starts a referral not listed above)",
                     "listed_by_n_adjudicators": 0})
    ref = pd.DataFrame(rows)
    ref.insert(0, "row_id", [f"R{i + 1:03d}" for i in range(len(ref))])
    for col in REV_COLS:
        ref[col] = ""
    hz = c[c.has_final_plan_hazard.astype(str).str.lower().eq("true")][
        ["encounter_id", "patient_key", "chart_extract_row", "encounter_timestamp", "window_end",
         "hazard_quotes_by_adjudicator"]].copy()
    for col in ["hazard_corrected_30d(yes|no)", "evidence_date(YYYY-MM-DD)", "evidence_text", "reviewer_initials", "notes"]:
        hz[col] = ""
    s2 = set(pd.read_csv(os.path.join(PK, "PRIVATE_packetC_second_reviewer_sample.csv"), dtype=str).encounter_id)
    out = {
        "PRIVATE_packetC_v2_referrals_reviewer1.csv": ref,
        "PRIVATE_packetC_v2_referrals_reviewer2.csv": ref[ref.encounter_id.isin(s2)],
        "PRIVATE_packetC_v2_hazards_reviewer1.csv": hz,
        "PRIVATE_packetC_v2_hazards_reviewer2.csv": hz[hz.encounter_id.isin(s2)],
    }
    for fn, df in out.items():
        p = os.path.join(PK, fn)
        df.to_csv(p, index=False)
        os.chmod(p, 0o600)
        print(fn, len(df), "rows,", df.encounter_id.nunique(), "encounters")
    print("candidate referral labels:", ref.candidate_referral.value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
