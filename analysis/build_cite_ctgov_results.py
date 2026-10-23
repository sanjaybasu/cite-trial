"""Aggregate tables for the NCT07718893 (CITE) results record: baseline, 30-day acute care, adverse events.

Inputs: wave-11 locked dataset; PRIVATE_wave11_30day_linkage.csv (from pull_30day_outcomes_wave11.py).
Deaths: dbt.eligibility.death_date (via medicaidNumber) plus ADT discharge disposition containing 'expir'.
Serious adverse event (SAE) definition used here: all-cause hospitalization or death within 30 days.
Writes aggregate-only JSON (no patient-level rows).
"""
import json
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, lighthouse, query  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RCT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(RCT, "results", "wave11_locked")
ARMS = ("unassisted_md_soap", "gemini_guardrails", "gemini_guardrails_anchor")
MCO_TO_STATE = {"CSOH": "OH", "UHCOH": "OH", "ABHVA": "VA", "SHPVA": "VA", "BCBSIL": "IL", "CHPW": "WA", "UHCWA": "WA"}


def arr(ids):
    assert all(str(i).replace("-", "").isalnum() for i in ids)
    return "ARRAY[" + ",".join(f"'{i}'" for i in ids) + "]"


def main():
    d = pd.read_csv(os.path.join(OUT, "PRIVATE_wave11_30day_linkage.csv"), parse_dates=["encounter_ts"])
    lh = lighthouse("prod")
    d = d.drop(columns=["lh_id"], errors="ignore")
    d["link_first"] = d.link_way.str.split(";").str[0]
    way = d.link_first.dropna().unique().tolist()
    pt = query(lh, f'''SELECT id lh_id, "waymarkPatientNumber" link_first, "medicaidNumber" mcd
                       FROM public."Patient" WHERE "waymarkPatientNumber" = ANY({arr(way)})''')
    d = d.merge(pt, on="link_first", how="left")

    cx = coredb("prod")
    el = query(cx, f"""SELECT medicaid_number mcd, max(death_date) death_date FROM dbt.eligibility
                       WHERE medicaid_number = ANY({arr(d.mcd.dropna().unique())}) AND death_date IS NOT NULL
                       GROUP BY 1""")
    adt = query(lh, f'''SELECT "patientId" lh_id, min("dischargedDate") expired_at FROM public."AdmissionDischargeTransfer"
                        WHERE "patientId" = ANY({arr(d.lh_id.dropna().unique())})
                          AND "dischargeDisposition" ILIKE '%%expir%%' AND "admitDate" >= '2026-07-01' GROUP BY 1''')
    d = d.merge(el, on="mcd", how="left").merge(adt, on="lh_id", how="left")
    d["death_ts"] = pd.to_datetime(d.death_date).combine_first(pd.to_datetime(d.expired_at))
    d["death_30d"] = ((d.death_ts > d.encounter_ts.dt.normalize() - pd.Timedelta(days=0))
                      & (d.death_ts <= d.encounter_ts + pd.Timedelta(days=30))).astype(int)
    d.loc[d.lh_id.isna(), "death_30d"] = pd.NA
    d["state"] = d.assignedMco.map(MCO_TO_STATE)

    def race(r):
        r = str(r)
        for k in ("Black or African American", "White", "American Indian or Alaska Native", "Hispanic or Latinx", "Asian", "Other"):
            if k in r:
                return k
        return "Unknown or not reported"

    d["race_cat"] = d.race.map(race)
    d.loc[d.lh_id.isna(), "race_cat"] = "Unknown or not reported"
    d["sex"] = d.gender.fillna("Unknown or not reported")
    d["age_cat"] = pd.cut(d.age_at_encounter, [-1, 17.999, 64.999, 200], labels=["<18", "18-64", ">=65"]).astype(str)
    d.loc[d.age_at_encounter.isna(), "age_cat"] = "Unknown"

    res = {"as_of": pd.Timestamp.now().strftime("%Y-%m-%d"), "n_by_arm": d.arm.value_counts().to_dict(), "baseline": {}, "ae": {}}
    for arm in ARMS + ("total",):
        s = d if arm == "total" else d[d.arm == arm]
        a = s.age_at_encounter.dropna()
        res["baseline"][arm] = {
            "n": int(len(s)), "age_n": int(len(a)), "age_mean": round(a.mean(), 1), "age_sd": round(a.std(), 1),
            "age_cat": s.age_cat.value_counts().to_dict(), "sex": s.sex.value_counts().to_dict(),
            "race": s.race_cat.value_counts().to_dict(), "state": s.state.fillna("Unknown").value_counts().to_dict(),
        }
        lk = s[s.lh_id.notna()]
        sae = ((lk.hosp_30d == 1) | (lk.death_30d == 1))
        res["ae"][arm] = {"at_risk": int(len(lk)), "deaths_30d": int(lk.death_30d.sum()),
                          "hosp_30d": int(lk.hosp_30d.sum()), "sae_any_30d": int(sae.sum()),
                          "ed_30d": int(lk.ed_30d.sum())}
    with open(os.path.join(OUT, "ctgov_results_aggregates.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
