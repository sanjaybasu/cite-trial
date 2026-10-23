"""ANCHOR RCT — assemble the database-locked, unblinded analysis dataset.

Merges three real inputs, all produced this session:
  1. The three physicians' completed primary scoresheets
     (rct_analysis/adjudication_packets_wave11_real/adjudicator_{A,B,C}_scoresheet.csv),
     scored per SAP §9 after the calibration gate cleared (Fleiss kappa 0.728
     at n=240, see rct_analysis/calibration/).
  2. adjudication_id_map_PRIVATE.json, which unblinds adjudication_id back to
     encounter_id/patient_key/arm. This is the trial's database-lock /
     unblinding step -- run once, per explicit authorization (2026-09-18).
  3. rct_analysis/derived/wave11_with_site_id_mco.csv, the wave11 export with
     the real wayid->MCO->state site covariate already resolved.

Per-component adjudicated labels are majority-of-three (>=2 of 3 raters);
primary_event_final is the disjunction of the four primary components, per
SAP: missed do-not-miss diagnosis, undertriage, contraindicated medication,
missed escalation. adj_over_escalation and adj_appropriate_escalation are
majority-voted the same way (appropriate_escalation is NA when fewer than 2
raters gave a non-NA vote, since it is only applicable when an escalation
occurred).

Output: rct_analysis/derived/analysis_dataset_wave11_locked.csv, ready for
rct_analysis/scripts/analyze_rct.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RCT_ANALYSIS = HERE.parent

COMPOSITE_ITEMS = ["adj_missed_dx", "adj_undertriage", "adj_contraindicated_med", "adj_missed_escalation"]
ALL_ITEMS = COMPOSITE_ITEMS + ["adj_over_escalation", "adj_appropriate_escalation"]


def load_rater(letter: str) -> pd.DataFrame:
    df = pd.read_csv(
        RCT_ANALYSIS / "adjudication_packets_wave11_real" / f"adjudicator_{letter}_scoresheet.csv"
    )
    df = df.sort_values("adjudication_id").reset_index(drop=True)
    for c in ALL_ITEMS:
        df[c] = df[c].map({"Yes": 1, "No": 0})  # "Not applicable" / blank -> NaN
    return df[["adjudication_id"] + ALL_ITEMS]


def majority_vote(row_a, row_b, row_c) -> float:
    votes = [v for v in (row_a, row_b, row_c) if not pd.isna(v)]
    if len(votes) < 2:
        return np.nan
    return float(sum(votes) >= 2)


def main() -> None:
    A, B, C = load_rater("A"), load_rater("B"), load_rater("C")
    assert (A["adjudication_id"].to_numpy() == B["adjudication_id"].to_numpy()).all()
    assert (A["adjudication_id"].to_numpy() == C["adjudication_id"].to_numpy()).all()

    adjudicated = pd.DataFrame({"adjudication_id": A["adjudication_id"]})
    for item in ALL_ITEMS:
        adjudicated[item] = [
            majority_vote(a, b, c) for a, b, c in zip(A[item], B[item], C[item])
        ]

    adjudicated["primary_event_final"] = (
        adjudicated[COMPOSITE_ITEMS].sum(axis=1, skipna=False) > 0
    ).astype(float)
    adjudicated["appropriate_escalation_adjudicated"] = adjudicated["adj_appropriate_escalation"]
    adjudicated["over_escalation_adjudicated"] = adjudicated["adj_over_escalation"]
    # per-rater raw composite (for compute_adjudicator_reliability in analyze_rct.py)
    for letter, df in [("1", A), ("2", B), ("3", C)]:
        adjudicated[f"adj{letter}_missed_dx"] = (
            df[COMPOSITE_ITEMS].sum(axis=1) > 0
        ).astype(int)

    id_map = json.load(
        open(RCT_ANALYSIS / "adjudication_packets_wave11_real" / "adjudication_id_map_PRIVATE.json")
    )["mapping"]
    id_map_df = pd.DataFrame(id_map)  # adjudication_id, encounter_id, patient_key, arm

    unblinded = adjudicated.merge(id_map_df, on="adjudication_id", how="left", validate="one_to_one")
    assert unblinded["encounter_id"].notna().all(), "unmapped adjudication_id(s)"

    base = pd.read_csv(RCT_ANALYSIS / "derived" / "wave11_with_site_id_mco.csv", dtype=str)
    drop_cols = [
        "adj1_missed_dx", "adj2_missed_dx", "adj3_missed_dx", "primary_event_final",
        "appropriate_escalation_adjudicated", "over_escalation_adjudicated",
    ]
    base = base.drop(columns=[c for c in drop_cols if c in base.columns])

    # arm and patient_key are present in both base (from the export) and
    # unblinded (from the id_map); cross-check agreement, then drop the
    # id_map's copies so the merge does not suffix duplicate column names.
    check = base[["encounter_id", "arm", "patient_key"]].merge(
        unblinded[["encounter_id", "arm", "patient_key"]], on="encounter_id", suffixes=("_base", "_idmap")
    )
    assert (check["arm_base"] == check["arm_idmap"]).all(), "arm mismatch between base export and id_map"
    assert (check["patient_key_base"] == check["patient_key_idmap"]).all(), "patient_key mismatch between base export and id_map"

    merged = base.merge(
        unblinded.drop(columns=["adjudication_id", "arm", "patient_key"]),
        on="encounter_id",
        how="left",
        validate="one_to_one",
    )
    assert merged["arm"].notna().all(), "encounter(s) missing arm after merge"
    assert merged["primary_event_final"].notna().all(), "encounter(s) missing adjudication after merge"

    merged["site_id"] = merged["site_id_state"]  # real state covariate, not the 'waymark_default' placeholder

    out_path = RCT_ANALYSIS / "derived" / "analysis_dataset_wave11_locked.csv"
    merged.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(merged)} rows)")
    print()
    print("primary_event_final by arm:")
    print(merged.groupby("arm")["primary_event_final"].agg(["count", "sum", "mean"]).to_string())
    print()
    print("site_id (state) coverage:", merged["site_id"].notna().sum(), "/", len(merged))
    print("appropriate_escalation_adjudicated non-null:", merged["appropriate_escalation_adjudicated"].notna().sum())


if __name__ == "__main__":
    main()
