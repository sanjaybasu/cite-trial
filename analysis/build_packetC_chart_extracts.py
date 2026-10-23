"""Packet C chart extracts: one PRIVATE Markdown file per row with the patient's records from the
index encounter date through window_end (inclusive).

Sources (Waymark prod, read-only, via waymark-data-access):
  lighthouse  EncounterNote (noteV2 + scribe summary; the human-edited Yjs summary when present), author role
              Task (+ TaskComment), Goal, AdmissionDischargeTransfer (deduplicated to episodes)
  coredb      dbt_tuva_core.encounter (claims; 60-90 day lag, so the latest windows are incomplete)
Internal pharmacist and therapist referrals appear as notes, so every note carries the author's role.
Adds `chart_extract_file` to both Packet C files. Patient-identifying: keep on Waymark devices only.
"""
import html
import json
import os
import pathlib
import re
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, lighthouse, query  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RCT = os.path.abspath(os.path.join(HERE, ".."))
PK = os.path.join(RCT, "adjudication_packets_cite_outcomes")
OUTD = os.path.join(PK, "PRIVATE_packetC_chart_extracts")
ARM_TAG = re.compile(r"gemini_guardrails_anchor|gemini_guardrails|unassisted_md_soap|\bANCHOR\b|\bCITE\b")


def arr(ids):
    ids = [str(i) for i in ids]
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids)
    return "ARRAY[" + ",".join(f"'{i}'" for i in ids) + "]"


def strip_html(t):
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ""
    t = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", "\n", str(t))
    t = re.sub(r"<li[^>]*>", "- ", t)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    return ARM_TAG.sub("[REDACTED]", re.sub(r"\n{3,}", "\n\n", t)).strip()


def yjs_text(b):
    try:
        import pycrdt
        doc = pycrdt.Doc()
        doc.apply_update(bytes(b))
        return strip_html(str(doc.get("root", type=pycrdt.XmlFragment)))
    except Exception:
        return ""


def main():
    c = pd.read_csv(os.path.join(PK, "PRIVATE_packetC_30day_chart_review.csv"), keep_default_na=False, dtype=str)
    fix = pd.read_csv(os.path.join(RCT, "derived", "PRIVATE_hash_to_wayid.csv")).set_index("encounter_id").wayid
    c["link_way"] = c.encounter_id.map(fix).fillna(c.patient_key.where(c.patient_key.str.startswith("WAY")))
    c["start"] = pd.to_datetime(c.encounter_timestamp, format="mixed").dt.normalize()
    c["end"] = pd.to_datetime(c.window_end) + pd.Timedelta(days=1)  # inclusive of window_end
    ways = sorted({w for v in c.link_way.dropna() for w in v.split(";")})

    dd = pd.read_csv(os.path.join(RCT, "derived", "analysis_dataset_wave11_locked.csv"), low_memory=False,
                     usecols=["encounter_id", "final_soap_note"])
    dfinal = dict(zip(dd.encounter_id, dd.final_soap_note.fillna("")))
    lh = lighthouse("prod")
    pt = query(lh, f'''SELECT id lh_id, "waymarkPatientNumber" way FROM public."Patient"
                       WHERE "waymarkPatientNumber" = ANY({arr(ways)})''')
    lhid = pt.groupby("way").lh_id.apply(list).to_dict()
    ids = pt.lh_id.tolist()
    lo, hi = c.start.min(), c.end.max()
    p = {"lo": lo.to_pydatetime(), "hi": hi.to_pydatetime()}

    notes = query(lh, f'''SELECT n.id, n."patientId" lh_id, n."startTime", n."dateOfEncounter", n."encounterOccurred",
                                 n."contactType"::text ct, n."encounterType"::text et, n."noteV2", w.title::text author_role,
                                 s.summary, y.data yjs
                          FROM public."EncounterNote" n
                          LEFT JOIN public."Waymarker" w ON w.id = n."createdByWaymarkerId"
                          LEFT JOIN public.encounter_summaries s ON s.encounter_note_id = n.id
                          LEFT JOIN public.yjs_document y ON y.id = s.yjs_document_id
                          WHERE n."patientId" = ANY({arr(ids)}) AND COALESCE(n.deleted, false) = false
                            AND COALESCE(n."startTime", n."dateOfEncounter") >= :lo
                            AND COALESCE(n."startTime", n."dateOfEncounter") < :hi''', **p)
    tasks = query(lh, f'''SELECT t.id, t."patientId" lh_id, t."taskTitle", t."descriptionV2", t.description, t.completed,
                                 t."dueDate", t."createdAt", t."updatedAt", t.type::text ttype
                          FROM public."Task" t WHERE t."patientId" = ANY({arr(ids)}) AND COALESCE(t.deleted,false)=false
                            AND (t."createdAt" BETWEEN :lo AND :hi OR t."updatedAt" BETWEEN :lo AND :hi
                                 OR t."dueDate" BETWEEN :lo AND :hi)''', **p)
    tc = query(lh, f'''SELECT "taskId" id, "commentText", "createdAt" FROM public."TaskComment"
                       WHERE "taskId" = ANY({arr(tasks.id.tolist() or ['x'])})''')
    goals = query(lh, f'''SELECT "patientId" lh_id, title, description, status::text status, "statusNote",
                                 category::text category, "createdAt", "updatedAt"
                          FROM public."Goal" WHERE "patientId" = ANY({arr(ids)}) AND COALESCE(deleted,false)=false
                            AND ("createdAt" BETWEEN :lo AND :hi OR "updatedAt" BETWEEN :lo AND :hi)''', **p)
    adt = query(lh, f'''SELECT "patientId" lh_id, "visitNumber", "admitDate", "dischargedDate", "patientClassCode" cls,
                               "hospitalName", "dischargeDisposition", "hospitalServiceDescription"
                        FROM public."AdmissionDischargeTransfer"
                        WHERE "patientId" = ANY({arr(ids)}) AND "admitDate" >= :lo AND "admitDate" < :hi''', **p)
    adt["k"] = adt.visitNumber.where(adt.visitNumber.fillna("") != "", adt.admitDate.astype(str).str[:10])
    adt = adt.sort_values("admitDate").drop_duplicates(["lh_id", "k", "cls"])
    cx = coredb("prod")
    clm = query(cx, f"""SELECT person_id way, encounter_type, encounter_start_date, facility_name,
                               attending_provider_name, primary_diagnosis_description
                        FROM dbt_tuva_core.encounter WHERE person_id = ANY({arr(ways)})
                          AND encounter_start_date >= :lo AND encounter_start_date < :hi""",
                lo=lo.date(), hi=hi.date())

    os.makedirs(OUTD, exist_ok=True)
    os.chmod(OUTD, 0o700)
    names = []
    for i, r in c.iterrows():
        mine = [x for w in str(r.link_way).split(";") for x in lhid.get(w, [])]
        fn = f"row{i + 1:03d}_{r.encounter_id[:8]}.md"
        names.append(fn)
        L = [f"# Chart extract, Packet C row {i + 1}", "",
             f"Patient: {r.link_way if r.link_way else r.patient_key}. Window: {r.start.date()} to {r.window_end} (inclusive).",
             "Read-only extract from lighthouse and the claims mart. Use it for the review; open lighthouse if something is unclear.",
             "Claims lag 60 to 90 days, so a missing claim near the end of the window is not evidence that a visit did not happen.", ""]
        fp = dfinal.get(r.encounter_id, "")
        L += ["## Final plan from the index encounter (as stored; [PATIENT] marks words removed before storage)", "",
              strip_html(fp) or "(no final plan text)", ""]
        if not mine:
            L += ["**This patient could not be linked to lighthouse. Review the chart by name and DOB.**", ""]
        def inw(df, col):
            return df[df.lh_id.isin(mine) & (pd.to_datetime(df[col]) >= r.start) & (pd.to_datetime(df[col]) < r.end)]
        n = notes[notes.lh_id.isin(mine)].copy()
        n["t"] = pd.to_datetime(n.startTime.fillna(n.dateOfEncounter))
        n = n[(n.t >= r.start) & (n.t < r.end)].sort_values("t")
        L += [f"## Care-team notes ({len(n)})", ""]
        for _, x in n.iterrows():
            # noteV2 first; for AI-scribe encounters the human-edited summary (Yjs), then the original draft
            body = strip_html(x.noteV2) or (yjs_text(x.yjs) if x.yjs is not None else "") or strip_html(x.summary)
            L += [f"### {x.t:%Y-%m-%d %H:%M} | author: {x.author_role or 'unknown'} | {x.ct or ''} {x.et or ''} | occurred: {x.encounterOccurred}",
                  "", body or "(no note text)", ""]
        t = tasks[tasks.lh_id.isin(mine)].copy()
        t = t.loc[pd.Series([any(pd.notna(v) and r.start <= pd.Timestamp(v) < r.end for v in (a, b, d))
                             for a, b, d in zip(t.createdAt, t.updatedAt, t.dueDate)], index=t.index, dtype=bool)]
        L += [f"## Tasks ({len(t)})", ""]
        for _, x in t.sort_values("createdAt").iterrows():
            L += [f"- **{x.taskTitle}** (created {str(x.createdAt)[:10]}, due {str(x.dueDate)[:10]}, completed: {x.completed}, type: {x.ttype})"]
            d = strip_html(x.descriptionV2) or strip_html(x.description)
            if d:
                L += ["  " + d.replace("\n", "\n  ")]
            for _, cm in tc[tc.id == x.id].sort_values("createdAt").iterrows():
                L += [f"  - comment {str(cm.createdAt)[:10]}: {strip_html(cm.commentText)}"]
        L += [""]
        g = goals[goals.lh_id.isin(mine)]
        g = g.loc[pd.Series([(r.start <= pd.Timestamp(a) < r.end) or (r.start <= pd.Timestamp(b) < r.end)
                             for a, b in zip(g.createdAt, g.updatedAt)], index=g.index, dtype=bool)]
        L += [f"## Goals created or updated ({len(g)})", ""]
        for _, x in g.iterrows():
            L += [f"- **{x.title}** [{x.category}] status {x.status} (updated {str(x.updatedAt)[:10]}): {strip_html(x.statusNote) or strip_html(x.description)}"]
        L += [""]
        a = inw(adt, "admitDate")
        L += [f"## ADT episodes ({len(a)})", ""]
        for _, x in a.iterrows():
            L += [f"- {str(x.admitDate)[:16]} class {x.cls} at {x.hospitalName}; discharged {str(x.dischargedDate)[:16]}; "
                  f"{x.dischargeDisposition or ''}; {x.hospitalServiceDescription or ''}"]
        L += [""]
        ws = str(r.link_way).split(";")
        k = clm[clm.way.isin(ws) & (pd.to_datetime(clm.encounter_start_date) >= r.start) & (pd.to_datetime(clm.encounter_start_date) < r.end)]
        L += [f"## Claims encounters ({len(k)}; lagged)", ""]
        for _, x in k.sort_values("encounter_start_date").iterrows():
            L += [f"- {x.encounter_start_date} {x.encounter_type}: {x.facility_name or ''} / {x.attending_provider_name or ''}; {x.primary_diagnosis_description or ''}"]
        with open(os.path.join(OUTD, fn), "w") as f:
            f.write("\n".join(L) + "\n")
        os.chmod(os.path.join(OUTD, fn), 0o600)

    # Combine into one file per reviewer; the per-row files are temporary.
    c["chart_extract_row"] = [f"row {i + 1}" for i in range(len(c))]
    by_enc = dict(zip(c.encounter_id, names))
    for f, out in (("PRIVATE_packetC_30day_chart_review.csv", "PRIVATE_packetC_chart_extracts_all78.md"),
                   ("PRIVATE_packetC_second_reviewer_sample.csv", "PRIVATE_packetC_chart_extracts_second_reviewer16.md")):
        path = os.path.join(PK, f)
        s = pd.read_csv(path, keep_default_na=False, dtype=str).drop(columns=["chart_extract_file"], errors="ignore")
        s["chart_extract_row"] = s.encounter_id.map(dict(zip(c.encounter_id, c.chart_extract_row)))
        s.to_csv(path, index=False)
        order = sorted(s.encounter_id, key=lambda e: names.index(by_enc[e]))
        parts = [f"# Packet C chart extracts ({len(order)} rows)", "",
                 "Each section matches the `chart_extract_row` value in your review sheet. Patient-identifying: Waymark devices only.", ""]
        parts += [f"- {c.set_index('encounter_id').at[e, 'chart_extract_row']}: {c.set_index('encounter_id').at[e, 'link_way']}" for e in order]
        for e in order:
            parts += ["", "---", "", open(os.path.join(OUTD, by_enc[e])).read()]
        op = os.path.join(PK, out)
        with open(op, "w") as fh:
            fh.write("\n".join(parts))
        os.chmod(op, 0o600)
    import shutil
    shutil.rmtree(OUTD)
    print(f"wrote combined extracts for {len(names)} rows; notes {len(notes)}, tasks {len(tasks)}, goals {len(goals)}, "
          f"adt episodes {len(adt)}, claims encounters {len(clm)}")


if __name__ == "__main__":
    main()
