"""CITE trial (NCT07718893) — blinded adjudication packets for the registered outcomes.

Packets (one copy per adjudicator, identical rows, independent shuffles of a shared blinded ID):
  A  AI-draft hazard review (Arms 2+3, n=161): reference standard for CITE diagnostic accuracy.
     Blind to arm and to CITE output (both arms carry a Gemini draft).
  B  Final-plan review (all arms, n=240): unresolved codifiable hazards and referrals in the final note.
  D  CITE flag review (Arm 3 flags only): flag validity. Unblinds Arm 3 membership; do it LAST.
  C  30-day chart review is generated after A/B consensus: `--stage c`.

Writes PRIVATE key maps; never writes WAY IDs or arm into a rater-facing file.
Usage:
  python generate_cite_outcome_packets.py            # A, B, D
  python generate_cite_outcome_packets.py --stage c  # C, after scoring A/B
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RCT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(RCT, "adjudication_packets_cite_outcomes")
SEED = 20261019
RATERS = ("adj1", "adj2", "adj3")
sys.path.insert(0, HERE)
from generate_adjudication_packets import redact_text  # noqa: E402

HAZARD_COLS = [
    "n_checkpoints",                        # patient-specific codifiable checkpoints in this encounter (integer)
    "hazard_present(yes|no)",
    "hazard_family(contraindication|drug_allergy|gdmt_omission|high_risk_med_dropped|numeric_refuted; ';' separated)",
    "hazard_severity(1-4; ';' aligned)",
    "hazard_quote(verbatim from plan or 'ABSENT: <expected action>'; ' || ' separated)",
    "insufficient_content(yes|no)",
    "notes",
]
FINAL_EXTRA = [
    "referrals_in_plan(n)",
    "referral_types(';' separated: specialty or service)",
]
FLAG_COLS = [
    "flag_valid(yes|no|uncertain)",
    "flag_severity_if_valid(1-4)",
    "flag_content_in_final_note(yes|partial|no)",
    "notes",
]


ARM_TAG = re.compile(r"\[Auto-captured[^\]]*\]|gemini_guardrails_anchor|gemini_guardrails|unassisted_md_soap|\bANCHOR\b|\bCITE\b")
WAY_ID = re.compile(r"\bWAY\d{4,}\b", re.I)


NAMES_SRC = os.path.join(RCT, "derived", "PRIVATE_v2_final_240_as_stored.csv")


def _load_names():
    """Cohort patient names from the as-stored Replit export (PRIVATE). Returns
    (per-encounter own-name regex, cohort full-name regex, non-dictionary name-token regex)."""
    if not os.path.exists(NAMES_SRC):
        return {}, None, None
    u = pd.read_csv(NAMES_SRC, low_memory=False, usecols=["encounter_id", "patient_first_name", "patient_last_name"])
    words = set()
    if os.path.exists("/usr/share/dict/words"):
        words = {w.strip() for w in open("/usr/share/dict/words") if w.strip().islower()}
    own, full, tokens = {}, set(), set()
    for e, f, l in zip(u.encounter_id, u.patient_first_name.fillna(""), u.patient_last_name.fillna("")):
        parts = [p for p in re.split(r"[\s\-]+", f"{f} {l}".strip()) if len(p) >= 2]
        if parts:
            # Name parts that are also English words match only when capitalized (e.g. a patient named
            # "Hope" must not erase "hope"); other parts match case-insensitively.
            alts = [re.escape(p) if p.lower() in words else "(?i:" + re.escape(p) + ")" for p in parts]
            own[e] = re.compile(r"\b(" + "|".join(alts) + r")\b")
        if f.strip() and l.strip():
            full.add(re.escape(f"{f.strip()} {l.strip()}"))
        tokens |= {p for p in parts if len(p) >= 3 and p.lower() not in words}
    full_re = re.compile(r"\b(" + "|".join(sorted(full, key=len, reverse=True)) + r")\b", re.I) if full else None
    tok_re = re.compile(r"\b(" + "|".join(map(re.escape, sorted(tokens, key=len, reverse=True))) + r")\b") if tokens else None
    return own, full_re, tok_re


OWN, FULL, TOK = _load_names()


def red(t, enc=None):
    t = redact_text("" if pd.isna(t) else str(t))[0]
    t = WAY_ID.sub("[ID]", ARM_TAG.sub("[REDACTED]", t))
    if FULL is not None:
        t = FULL.sub("[NAME]", t)
    if enc in OWN:
        t = OWN[enc].sub("[NAME]", t)
    if TOK is not None:
        t = TOK.sub("[NAME]", t)
    return t


def write_copies(df, name, cols, key_cols):
    rng = np.random.default_rng(SEED + sum(map(ord, name)))
    for r in RATERS:
        sheet = df[key_cols].copy()
        for c in cols:
            sheet[c] = ""
        sheet.insert(0, "adjudicator_id", r)
        sheet = sheet.sample(frac=1, random_state=int(rng.integers(1e9)))
        os.makedirs(os.path.join(OUT, r), exist_ok=True)
        sheet.to_csv(os.path.join(OUT, r, f"{name}_{r}.csv"), index=False)


def stage_abd(d):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    keys = {}

    a = d[d.arm.isin(["gemini_guardrails", "gemini_guardrails_anchor"])].copy()
    a["blinded_id"] = [f"A{i:03d}" for i in rng.permutation(len(a)) + 1]
    a["presenting"] = [red(t, e) for t, e in zip(a.presenting_complaint_text, a.encounter_id)]
    a["transcript"] = [red(t, e) for t, e in zip(a.transcription_text, a.encounter_id)]
    a["ai_draft_plan"] = [red(t, e) for t, e in zip(a.gemini_response, a.encounter_id)]
    write_copies(a, "packetA_ai_draft", HAZARD_COLS, ["blinded_id", "presenting", "transcript", "ai_draft_plan"])
    keys["A"] = a[["blinded_id", "encounter_id", "arm"]].to_dict("records")

    b = d.copy()
    b["blinded_id"] = [f"B{i:03d}" for i in rng.permutation(len(b)) + 1]
    b["presenting"] = [red(t, e) for t, e in zip(b.presenting_complaint_text, b.encounter_id)]
    b["transcript"] = [red(t, e) for t, e in zip(b.transcription_text, b.encounter_id)]
    b["final_plan"] = [red(t, e) for t, e in zip(b.final_soap_note, b.encounter_id)]
    write_copies(b, "packetB_final_plan", HAZARD_COLS[:-1] + FINAL_EXTRA + ["notes"],
                 ["blinded_id", "presenting", "transcript", "final_plan"])
    keys["B"] = b[["blinded_id", "encounter_id", "arm"]].to_dict("records")

    rows = []
    for _, r in d[d.cite_flagged == "t"].iterrows():
        for j, f in enumerate(json.loads(r.cite_flags)):
            rows.append({"encounter_id": r.encounter_id, "flag_seq": j,
                         "flag_type": f.get("type"), "flag_text": red(f.get("quote"), r.encounter_id),
                         "flag_evidence": red(f.get("evidence_citation"), r.encounter_id), "flag_severity_cite": f.get("severity"),
                         "presenting": red(r.presenting_complaint_text, r.encounter_id),
                         "transcript": red(r.transcription_text, r.encounter_id),
                         "ai_draft_plan": red(r.gemini_response, r.encounter_id),
                         "final_plan": red(r.final_soap_note, r.encounter_id)})
    fl = pd.DataFrame(rows)
    fl["blinded_id"] = [f"D{i:03d}" for i in rng.permutation(len(fl)) + 1]
    write_copies(fl, "packetD_cite_flags", FLAG_COLS,
                 ["blinded_id", "flag_type", "flag_text", "flag_evidence", "flag_severity_cite",
                  "presenting", "transcript", "ai_draft_plan", "final_plan"])
    keys["D"] = fl[["blinded_id", "encounter_id", "flag_seq"]].to_dict("records")

    with open(os.path.join(OUT, "PRIVATE_key_map.json"), "w") as f:
        json.dump({"seed": SEED, "generated": pd.Timestamp.now(tz="UTC").isoformat(), **keys}, f, indent=1)
    print(f"A={len(a)} B={len(b)} D={len(fl)} -> {OUT}")


def consensus_yes(name, col):
    """Majority-of-three 'yes' on `col` per blinded_id."""
    s = pd.concat([pd.read_csv(os.path.join(OUT, r, f"{name}_{r}.csv"), keep_default_na=False) for r in RATERS])
    v = s.assign(y=s[col].str.strip().str.lower().eq("yes")).groupby("blinded_id").y.sum()
    return set(v[v >= 2].index), s


def stage_c(d):
    """30-day chart-review sheet: final-plan hazards (B consensus) and referrals (any rater listed one).
    Chart review needs patient identity, so this sheet is PRIVATE and is completed in lighthouse."""
    key = pd.DataFrame(json.load(open(os.path.join(OUT, "PRIVATE_key_map.json")))["B"])
    haz, s = consensus_yes("packetB_final_plan", "hazard_present(yes|no)")
    ref_col = "referrals_in_plan(n)"
    ref = set(s[pd.to_numeric(s[ref_col], errors="coerce").fillna(0) > 0].blinded_id)
    ids = sorted(haz | ref)
    c = key[key.blinded_id.isin(ids)].merge(d[["encounter_id", "patient_key", "encounter_timestamp"]], on="encounter_id")
    qcol = [k for k in s.columns if k.startswith("hazard_quote")][0]
    tcol = [k for k in s.columns if k.startswith("referral_types")][0]
    agg = lambda col: s.groupby("blinded_id")[col].apply(lambda v: " | ".join(f"{r}: {t}" for r, t in zip(s.loc[v.index, "adjudicator_id"], v) if str(t).strip()))
    c["hazard_quotes_by_adjudicator"] = c.blinded_id.map(agg(qcol))
    c["referral_types_by_adjudicator"] = c.blinded_id.map(agg(tcol))
    c["has_final_plan_hazard"] = c.blinded_id.isin(haz)
    c["has_referral"] = c.blinded_id.isin(ref)
    c["window_end"] = (pd.to_datetime(c.encounter_timestamp, format="mixed") + pd.Timedelta(days=30)).dt.date
    for col in ["hazard_corrected_30d(yes|no|na)", "correction_evidence(note date + type)",
                "referrals_completed_30d(n)", "referral_completion_evidence", "reviewer", "notes"]:
        c[col] = ""
    c.drop(columns=["arm"]).to_csv(os.path.join(OUT, "PRIVATE_packetC_30day_chart_review.csv"), index=False)
    print(f"C rows={len(c)} (hazard {len(haz)}, referral {len(ref)})")


def stage_reredact(d):
    """Rewrite text columns of the issued per-rater packets in place (same blinded IDs and row order),
    keeping every scoring column, so returned calibration scores carry over."""
    key = json.load(open(os.path.join(OUT, "PRIVATE_key_map.json")))
    src = {"A": ("packetA_ai_draft", {"presenting": "presenting_complaint_text", "transcript": "transcription_text",
                                      "ai_draft_plan": "gemini_response"}),
           "B": ("packetB_final_plan", {"presenting": "presenting_complaint_text", "transcript": "transcription_text",
                                        "final_plan": "final_soap_note"})}
    dd = d.set_index("encounter_id")
    for k, (name, cols) in src.items():
        enc = pd.DataFrame(key[k]).set_index("blinded_id").encounter_id
        for r in RATERS:
            f = os.path.join(OUT, r, f"{name}_{r}.csv")
            sh = pd.read_csv(f, keep_default_na=False, dtype=str)
            for out_col, src_col in cols.items():
                sh[out_col] = [red(dd.at[enc[b], src_col], enc[b]) for b in sh.blinded_id]
            sh.to_csv(f, index=False)
    # Packet D: rebuild text from flags (same blinded IDs)
    dk = pd.DataFrame(key["D"])
    for r in RATERS:
        f = os.path.join(OUT, r, f"packetD_cite_flags_{r}.csv")
        sh = pd.read_csv(f, keep_default_na=False, dtype=str)
        for i, b in enumerate(sh.blinded_id):
            row = dk[dk.blinded_id == b].iloc[0]
            e = row.encounter_id; fl = json.loads(dd.at[e, "cite_flags"])[int(row.flag_seq)]
            sh.at[i, "flag_text"] = red(fl.get("quote"), e)
            sh.at[i, "flag_evidence"] = red(fl.get("evidence_citation"), e)
            for oc, sc in (("presenting", "presenting_complaint_text"), ("transcript", "transcription_text"),
                           ("ai_draft_plan", "gemini_response"), ("final_plan", "final_soap_note")):
                sh.at[i, oc] = red(dd.at[e, sc], e)
        sh.to_csv(f, index=False)
    print("re-redacted all issued packets in place")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="abd", choices=["abd", "c", "reredact"])
    d = pd.read_csv(os.path.join(RCT, "derived", "analysis_dataset_wave11_locked.csv"), low_memory=False)
    {"abd": stage_abd, "c": stage_c, "reredact": stage_reredact}[ap.parse_args().stage](d)
