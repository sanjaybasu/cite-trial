"""30-day ED and hospitalization outcomes, site, and demographics for the wave-11 (CITE, NCT07718893) cohort.

Source of truth for acute care: lighthouse public."AdmissionDischargeTransfer".
ED visit = patientClassCode 'E' episode; hospitalization = patientClassCode 'I' episode.
Episodes are deduplicated on (patientId, visitNumber), falling back to (patientId, admitDate::date).
Window: admitDate in (encounter_ts, encounter_ts + 30 days].
Writes PRIVATE linkage (patient-level) and an aggregate JSON only.
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import lighthouse, query  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RCT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(RCT, "results", "wave11_locked")
ARMS = ("unassisted_md_soap", "gemini_guardrails", "gemini_guardrails_anchor")
sys.path.insert(0, HERE)
from compute_exploratory_and_fairness import wilson, newcombe, fisher_p  # noqa: E402


def main():
    d = pd.read_csv(os.path.join(RCT, "derived", "analysis_dataset_wave11_locked.csv"), low_memory=False)
    d["encounter_ts"] = pd.to_datetime(d["encounter_timestamp"], format="mixed")
    d["window_end"] = d["encounter_ts"] + pd.Timedelta(days=30)


    # Link key: the captured WAY ID, or the WAY ID recovered from the HMAC hashes
    # (derived/PRIVATE_hash_to_wayid.csv; ';' separates duplicate lighthouse records of one person).
    fix = pd.read_csv(os.path.join(RCT, "derived", "PRIVATE_hash_to_wayid.csv")).set_index("encounter_id")["wayid"]
    d["link_way"] = d["encounter_id"].map(fix).fillna(d["patient_key"].where(d["patient_key"].str.startswith("WAY")))
    links = d[["encounter_id", "link_way"]].dropna().assign(link_way=lambda x: x.link_way.str.split(";")).explode("link_way")

    eng = lighthouse("prod")
    ids = links["link_way"].unique().tolist()
    assert all(i.isalnum() for i in ids)
    arr = "ARRAY[" + ",".join(f"'{i}'" for i in ids) + "]"
    pt = query(eng, f'''SELECT id AS lh_id, "waymarkPatientNumber" AS link_way, "birthDate", gender,
                               race, ethnicity, languages, "assignedMco"
                        FROM public."Patient" WHERE "waymarkPatientNumber" = ANY({arr})''')
    links = links.merge(pt, on="link_way", how="inner")
    print(f"linked encounters: {links.encounter_id.nunique()}/{len(d)}")
    d = d.merge(links.drop_duplicates("encounter_id").drop(columns=["link_way"]), on="encounter_id", how="left")
    lh_by_enc = links.groupby("encounter_id")["lh_id"].apply(set)

    lh_ids = links["lh_id"].unique().tolist()
    arr2 = "ARRAY[" + ",".join(f"'{i}'" for i in lh_ids) + "]"
    adt = query(eng, f'''SELECT "patientId" AS lh_id, "visitNumber", "admitDate", "patientClassCode"
                         FROM public."AdmissionDischargeTransfer"
                         WHERE "patientId" = ANY({arr2}) AND "admitDate" >= '2026-07-01' ''')
    adt["admitDate"] = pd.to_datetime(adt["admitDate"])
    adt["ep_key"] = adt["visitNumber"].where(adt["visitNumber"].notna() & (adt["visitNumber"] != ""),
                                             adt["admitDate"].dt.date.astype(str))
    ep = adt.sort_values("admitDate").drop_duplicates(["lh_id", "ep_key", "patientClassCode"])
    print(f"ADT rows {len(adt)} -> episodes {len(ep)}")

    def outcome(row, cls):
        if pd.isna(row["lh_id"]):
            return np.nan
        s = ep[ep.lh_id.isin(lh_by_enc[row["encounter_id"]]) & (ep.patientClassCode == cls)
               & (ep.admitDate > row["encounter_ts"]) & (ep.admitDate <= row["window_end"])]
        return int(len(s) > 0)

    d["ed_30d"] = d.apply(outcome, axis=1, cls="E")
    d["hosp_30d"] = d.apply(outcome, axis=1, cls="I")
    d["age_at_encounter"] = (d["encounter_ts"] - pd.to_datetime(d["birthDate"])).dt.days / 365.25

    d[["patient_key", "link_way", "encounter_id", "arm", "encounter_ts", "site_id", "site_id_state", "site_id_mco",
       "assignedMco", "age_at_encounter", "gender", "race", "ethnicity", "languages", "ed_30d", "hosp_30d"]
      ].to_csv(os.path.join(OUT, "PRIVATE_wave11_30day_linkage.csv"), index=False)

    res = {"as_of": pd.Timestamp.now().strftime("%Y-%m-%d"), "source": 'lighthouse public."AdmissionDischargeTransfer"',
           "last_window_close": str(d["window_end"].max().date()),
           "linked_n": int(d["lh_id"].notna().sum()), "n": int(len(d)), "outcomes": {}}
    for oc in ("ed_30d", "hosp_30d"):
        by = {}
        for a in ARMS:
            s = d[(d.arm == a) & d[oc].notna()]
            k, n = int(s[oc].sum()), int(len(s))
            p, lo, hi = wilson(k, n)
            by[a] = {"events": k, "n": n, "rate": p, "ci_low": lo, "ci_high": hi}
        c32 = newcombe(by[ARMS[2]]["events"], by[ARMS[2]]["n"], by[ARMS[1]]["events"], by[ARMS[1]]["n"])
        c21 = newcombe(by[ARMS[1]]["events"], by[ARMS[1]]["n"], by[ARMS[0]]["events"], by[ARMS[0]]["n"])
        res["outcomes"][oc] = {"by_arm": by,
                               "arm3_vs_arm2": {"rd": c32[0], "ci_low": c32[1], "ci_high": c32[2]},
                               "arm2_vs_arm1": {"rd": c21[0], "ci_low": c21[1], "ci_high": c21[2]}}
    with open(os.path.join(OUT, "exploratory_30day_wave11.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
