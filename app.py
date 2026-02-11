import json
import os
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
    from gspread.exceptions import WorksheetNotFound
except Exception:
    gspread = None
    Credentials = None
    WorksheetNotFound = Exception

pwd = os.environ.get("APP_PASSWORD", "")
if pwd:
    p = st.text_input("Mot de passe", type="password")
    if p != pwd:
        st.stop()

st.set_page_config(page_title="AVC Hyperaigu - Délais Thrombolyse", layout="centered")

DB_PATH = os.environ.get("APP_DB_PATH", "data/avc_delais.db")
DEFAULT_GSHEET_ID = "1ZQMo6j6zJ9G0Pl-rZGh6Oa1wPyV7OSYNb_joSjRdRec"
GSHEET_ID = os.environ.get("GOOGLE_SHEET_ID", DEFAULT_GSHEET_ID)
GSHEET_WORKSHEET_EXTRA = os.environ.get("GOOGLE_SHEET_WORKSHEET_EXTRA", "avc_extra_hospitaliers")
GSHEET_WORKSHEET_INTRA = os.environ.get("GOOGLE_SHEET_WORKSHEET_INTRA", "avc_intra_hospitaliers")
GSHEET_WORKSHEET_OTHER_HOSPITAL = os.environ.get("GOOGLE_SHEET_WORKSHEET_OTHER_HOSPITAL", "avc_appel_autre_hopital")

PATHWAY_SAMU = "Appel du SAMU"
PATHWAY_INTRA = "Appel intra-hospitalier"
PATHWAY_OTHER = "Appel d'un autre hôpital"

GSHEET_COLUMNS: List[Tuple[str, str]] = [
    ("id", "ID enregistrement"),
    ("created_at", "Date enregistrement"),
    ("care_pathway", "Parcours AVC"),
    ("onset_source", "Source heure début"),
    ("ts_onset_known", "Heure début connue"),
    ("ts_onset_unknown", "Heure dernière fois vue normale"),
    ("ts_onset_reference", "Heure début retenue pour le calcul"),
    ("ts_samu_call", "Heure appel SAMU"),
    ("ts_other_hospital_call", "Heure appel d'un autre hôpital"),
    ("ts_ed_arrival", "Heure arrivée urgences"),
    ("ts_neuro_call", "Heure appel neurologue de garde"),
    ("ts_irm_arrival", "Heure arrivée IRM"),
    ("ts_imaging_start", "Heure début IRM/imagerie"),
    ("ts_imaging_end", "Heure fin IRM/imagerie"),
    ("ts_needle", "Heure bolus rtPA"),
    ("dtn_min", "Délai arrivée->bolus (DTN) (min)"),
    ("door_to_imaging_min", "Délai arrivée->début imagerie (min)"),
    ("imaging_to_needle_min", "Délai début imagerie->bolus (min)"),
    ("samu_to_arrival_min", "Délai appel SAMU->arrivée (min)"),
    ("samu_to_needle_min", "Délai appel SAMU->bolus (min)"),
    ("other_hospital_to_arrival_min", "Délai appel autre hôpital->arrivée (min)"),
    ("other_hospital_to_needle_min", "Délai appel autre hôpital->bolus (min)"),
    ("internal_alert_to_arrival_min", "Délai alerte interne->arrivée (min)"),
    ("internal_alert_to_needle_min", "Délai alerte interne->bolus (min)"),
    ("onset_to_needle_min", "Délai début retenu->bolus (min)"),
    ("commentaire", "Commentaire"),
    ("auto_fix_enabled", "Correction automatique activée"),
    ("auto_correction_applied", "Correction automatique appliquée"),
    ("notes", "Notes"),
    ("exported_at", "Date export"),
]

EVENT_LABELS = {
    "onset_reference": "Début AVC/dernière fois vue normale retenu",
    "samu_call": "Appel SAMU",
    "other_hospital_call": "Appel d'un autre hôpital",
    "ed_arrival": "Arrivée urgences",
    "neuro_call": "Appel neurologue de garde",
    "irm_arrival": "Arrivée IRM",
    "imaging_start": "Début IRM/imagerie",
    "imaging_end": "Fin IRM/imagerie",
    "needle": "Bolus rtPA",
}


def parse_time_hhmm(value: str) -> Optional[time]:
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%H:%M").time()
    except ValueError:
        return None


def build_datetimes_with_rollover(
    base_date: date,
    time_inputs: Dict[str, str],
    event_order: List[str],
) -> Tuple[Dict[str, Optional[datetime]], List[str], List[str]]:
    parsed: Dict[str, Optional[time]] = {}
    errors: List[str] = []

    for key, value in time_inputs.items():
        t = parse_time_hhmm(value)
        if value.strip() and t is None:
            errors.append(f"Format invalide pour '{EVENT_LABELS.get(key, key)}' (attendu HH:MM).")
        parsed[key] = t

    day_shift = 0
    previous_dt: Optional[datetime] = None
    previous_key: Optional[str] = None
    resolved: Dict[str, Optional[datetime]] = {k: None for k in time_inputs.keys()}
    notes: List[str] = []

    for key in event_order:
        t = parsed.get(key)
        if t is None:
            continue

        current_dt = datetime.combine(base_date, t) + timedelta(days=day_shift)
        if previous_dt is not None and current_dt < previous_dt:
            while current_dt < previous_dt:
                day_shift += 1
                current_dt = datetime.combine(base_date, t) + timedelta(days=day_shift)
            notes.append(
                f"Passage minuit détecté: {EVENT_LABELS.get(previous_key, previous_key)} -> {EVENT_LABELS.get(key, key)} (+1 jour)."
            )

        resolved[key] = current_dt
        previous_dt = current_dt
        previous_key = key

    return resolved, notes, errors


def minutes(a: Optional[datetime], b: Optional[datetime]) -> Optional[int]:
    if a is None or b is None:
        return None
    return int((b - a).total_seconds() // 60)


def fmt_dt(x: Optional[datetime]) -> str:
    return "" if x is None else x.strftime("%Y-%m-%d %H:%M")


def resolve_single_datetime(base_date: date, raw_time: str) -> Optional[datetime]:
    parsed = parse_time_hhmm(raw_time)
    if parsed is None:
        return None
    return datetime.combine(base_date, parsed)


def get_db_connection() -> sqlite3.Connection:
    db_file = Path(DB_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_records (
                id TEXT PRIMARY KEY,
                case_id TEXT,
                care_pathway TEXT,
                onset_source TEXT,
                ts_onset_known TEXT,
                ts_onset_unknown TEXT,
                ts_onset_reference TEXT,
                ts_samu_call TEXT,
                ts_other_hospital_call TEXT,
                ts_ed_arrival TEXT,
                ts_neuro_call TEXT,
                ts_irm_arrival TEXT,
                ts_imaging_start TEXT,
                ts_imaging_arrival TEXT,
                ts_imaging_end TEXT,
                ts_needle TEXT,
                dtn_min INTEGER,
                door_to_imaging_min INTEGER,
                imaging_to_needle_min INTEGER,
                samu_to_arrival_min INTEGER,
                samu_to_needle_min INTEGER,
                other_hospital_to_arrival_min INTEGER,
                other_hospital_to_needle_min INTEGER,
                internal_alert_to_arrival_min INTEGER,
                internal_alert_to_needle_min INTEGER,
                onset_to_needle_min INTEGER,
                commentaire TEXT,
                auto_fix_enabled INTEGER NOT NULL,
                auto_correction_applied INTEGER NOT NULL,
                notes TEXT,
                exported_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(patient_records)").fetchall()}
        required_cols = {
            "care_pathway": "TEXT",
            "onset_source": "TEXT",
            "ts_onset_known": "TEXT",
            "ts_onset_unknown": "TEXT",
            "ts_onset_reference": "TEXT",
            "ts_samu_call": "TEXT",
            "ts_other_hospital_call": "TEXT",
            "ts_ed_arrival": "TEXT",
            "ts_neuro_call": "TEXT",
            "ts_irm_arrival": "TEXT",
            "ts_imaging_start": "TEXT",
            "ts_imaging_arrival": "TEXT",
            "ts_imaging_end": "TEXT",
            "ts_needle": "TEXT",
            "dtn_min": "INTEGER",
            "door_to_imaging_min": "INTEGER",
            "imaging_to_needle_min": "INTEGER",
            "samu_to_arrival_min": "INTEGER",
            "samu_to_needle_min": "INTEGER",
            "other_hospital_to_arrival_min": "INTEGER",
            "other_hospital_to_needle_min": "INTEGER",
            "internal_alert_to_arrival_min": "INTEGER",
            "internal_alert_to_needle_min": "INTEGER",
            "onset_to_needle_min": "INTEGER",
            "commentaire": "TEXT",
            "auto_correction_applied": "INTEGER",
        }
        for col_name, col_type in required_cols.items():
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE patient_records ADD COLUMN {col_name} {col_type}")

        conn.commit()


def build_record(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": str(uuid4()),
        "case_id": "",
        "care_pathway": row.get("care_pathway", ""),
        "onset_source": row.get("onset_source", ""),
        "ts_onset_known": row.get("ts_onset_known", ""),
        "ts_onset_unknown": row.get("ts_onset_unknown", ""),
        "ts_onset_reference": row.get("ts_onset_reference", ""),
        "ts_samu_call": row.get("ts_samu_call", ""),
        "ts_other_hospital_call": row.get("ts_other_hospital_call", ""),
        "ts_ed_arrival": row.get("ts_ed_arrival", ""),
        "ts_neuro_call": row.get("ts_neuro_call", ""),
        "ts_irm_arrival": row.get("ts_irm_arrival", ""),
        "ts_imaging_start": row.get("ts_imaging_start", ""),
        "ts_imaging_arrival": row.get("ts_imaging_start", ""),
        "ts_imaging_end": row.get("ts_imaging_end", ""),
        "ts_needle": row.get("ts_needle", ""),
        "dtn_min": row.get("dtn_min"),
        "door_to_imaging_min": row.get("door_to_imaging_min"),
        "imaging_to_needle_min": row.get("imaging_to_needle_min"),
        "samu_to_arrival_min": row.get("samu_to_arrival_min"),
        "samu_to_needle_min": row.get("samu_to_needle_min"),
        "other_hospital_to_arrival_min": row.get("other_hospital_to_arrival_min"),
        "other_hospital_to_needle_min": row.get("other_hospital_to_needle_min"),
        "internal_alert_to_arrival_min": row.get("internal_alert_to_arrival_min"),
        "internal_alert_to_needle_min": row.get("internal_alert_to_needle_min"),
        "onset_to_needle_min": row.get("onset_to_needle_min"),
        "commentaire": row.get("commentaire", ""),
        "auto_fix_enabled": 1,
        "auto_correction_applied": 1 if row.get("auto_correction_applied", False) else 0,
        "notes": row.get("notes", ""),
        "exported_at": row.get("exported_at", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_patient_record_sqlite(record: Dict[str, object]) -> str:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO patient_records (
                id, case_id, care_pathway, onset_source,
                ts_onset_known, ts_onset_unknown, ts_onset_reference,
                ts_samu_call, ts_other_hospital_call, ts_ed_arrival, ts_neuro_call,
                ts_irm_arrival, ts_imaging_start, ts_imaging_arrival, ts_imaging_end, ts_needle,
                dtn_min, door_to_imaging_min, imaging_to_needle_min,
                samu_to_arrival_min, samu_to_needle_min, other_hospital_to_arrival_min, other_hospital_to_needle_min,
                internal_alert_to_arrival_min, internal_alert_to_needle_min,
                onset_to_needle_min, commentaire,
                auto_fix_enabled, auto_correction_applied, notes, exported_at, created_at
            ) VALUES (
                :id, :case_id, :care_pathway, :onset_source,
                :ts_onset_known, :ts_onset_unknown, :ts_onset_reference,
                :ts_samu_call, :ts_other_hospital_call, :ts_ed_arrival, :ts_neuro_call,
                :ts_irm_arrival, :ts_imaging_start, :ts_imaging_arrival, :ts_imaging_end, :ts_needle,
                :dtn_min, :door_to_imaging_min, :imaging_to_needle_min,
                :samu_to_arrival_min, :samu_to_needle_min, :other_hospital_to_arrival_min, :other_hospital_to_needle_min,
                :internal_alert_to_arrival_min, :internal_alert_to_needle_min,
                :onset_to_needle_min, :commentaire,
                :auto_fix_enabled, :auto_correction_applied, :notes, :exported_at, :created_at
            )
            """,
            record,
        )
        conn.commit()

    return str(record["id"])


def _get_gsheet_creds_info() -> Optional[Dict]:
    if "google_service_account" in st.secrets:
        return dict(st.secrets["google_service_account"])

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def worksheet_name_from_pathway(care_pathway: str) -> str:
    if care_pathway == PATHWAY_SAMU:
        return GSHEET_WORKSHEET_EXTRA
    if care_pathway == PATHWAY_OTHER:
        return GSHEET_WORKSHEET_OTHER_HOSPITAL
    return GSHEET_WORKSHEET_INTRA


def save_patient_record_google_sheet(record: Dict[str, object]) -> Tuple[bool, str]:
    if not GSHEET_ID:
        return False, "GOOGLE_SHEET_ID non configuré"
    if gspread is None or Credentials is None:
        return False, "Dépendances Google Sheets manquantes"

    creds_info = _get_gsheet_creds_info()
    if not creds_info:
        return False, "Credentials service account introuvables"

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GSHEET_ID)

        worksheet_name = worksheet_name_from_pathway(str(record.get("care_pathway", "")))
        gsheet_headers = [label for _, label in GSHEET_COLUMNS]
        gsheet_values = [record.get(key, "") for key, _ in GSHEET_COLUMNS]

        try:
            ws = sheet.worksheet(worksheet_name)
        except WorksheetNotFound:
            ws = sheet.add_worksheet(title=worksheet_name, rows=2000, cols=len(GSHEET_COLUMNS))

        header = ws.row_values(1)
        if not header:
            ws.append_row(gsheet_headers, value_input_option="RAW")
        elif header != gsheet_headers:
            ws.update("A1", [gsheet_headers], value_input_option="RAW")

        ws.append_row(gsheet_values, value_input_option="RAW")
        return True, f"OK ({worksheet_name})"
    except Exception as exc:
        return False, str(exc)


def load_recent_records(limit: int = 50) -> pd.DataFrame:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id, created_at, case_id, care_pathway, onset_source,
                ts_onset_known, ts_onset_unknown, ts_onset_reference,
                ts_samu_call, ts_other_hospital_call, ts_ed_arrival, ts_neuro_call,
                ts_irm_arrival, ts_imaging_start, ts_imaging_end, ts_needle,
                dtn_min, door_to_imaging_min, imaging_to_needle_min,
                samu_to_arrival_min, samu_to_needle_min, other_hospital_to_arrival_min, other_hospital_to_needle_min,
                internal_alert_to_arrival_min, internal_alert_to_needle_min,
                onset_to_needle_min, commentaire, auto_correction_applied
            FROM patient_records
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


init_db()

# -----------------------------
# UI
# -----------------------------
st.title("AVC hyperaigu - délais thrombolyse")
st.caption("Saisie simplifiée avec gestion automatique du passage minuit.")

with st.sidebar:
    st.header("Paramètres")
    st.write(f"Feuille extra-hospitalier: {GSHEET_WORKSHEET_EXTRA}")
    st.write(f"Feuille intra-hospitalier: {GSHEET_WORKSHEET_INTRA}")
    st.write(f"Feuille appel autre hôpital: {GSHEET_WORKSHEET_OTHER_HOSPITAL}")
    if GSHEET_ID:
        st.success("Google Sheets configuré")
    else:
        st.info("Google Sheets non configuré (fallback SQLite)")

st.subheader("Recueil")
care_pathway = st.radio(
    "Parcours AVC",
    options=[PATHWAY_SAMU, PATHWAY_INTRA, PATHWAY_OTHER],
    horizontal=True,
)
reference_date = st.date_input("Date de référence", value=date.today())
onset_mode = st.radio(
    "Heure de début disponible",
    options=["Heure début AVC connue", "Heure début inconnue (utiliser dernière fois vue normale)"],
    horizontal=True,
)

t_onset_known = ""
t_onset_unknown = ""
if onset_mode == "Heure début AVC connue":
    t_onset_known = st.text_input("Heure début AVC connue (HH:MM)", value="", placeholder="ex: 08:15")
else:
    t_onset_unknown = st.text_input(
        "Heure dernière fois vue normale (HH:MM)",
        value="",
        placeholder="ex: 07:40",
    )

onset_known_dt = resolve_single_datetime(reference_date, t_onset_known)
onset_unknown_dt = resolve_single_datetime(reference_date, t_onset_unknown)

manual_errors: List[str] = []
if onset_mode == "Heure début AVC connue":
    if onset_known_dt is None:
        manual_errors.append("Renseigner une heure valide pour 'Heure début AVC connue' (HH:MM).")
    onset_source = "Début AVC connu"
    onset_reference_raw = t_onset_known
else:
    if onset_unknown_dt is None:
        manual_errors.append("Renseigner une heure valide pour 'Heure dernière fois vue normale' (HH:MM).")
    onset_source = "Heure dernière fois vue normale"
    onset_reference_raw = t_onset_unknown

onset_notes: List[str] = []

if care_pathway == PATHWAY_SAMU:
    t_samu_call = st.text_input("Heure appel SAMU (HH:MM)", value="", placeholder="ex: 09:00")
    t_irm_arrival = st.text_input("Heure arrivée IRM (HH:MM)", value="", placeholder="ex: 09:35")
    t_imaging_start = st.text_input("Heure début IRM/imagerie (HH:MM)", value="", placeholder="ex: 09:40")
    t_imaging_end = st.text_input("Heure fin IRM/imagerie (HH:MM)", value="", placeholder="ex: 09:55")
    t_needle = st.text_input("Heure bolus rtPA (HH:MM)", value="", placeholder="ex: 10:05")

    time_inputs = {
        "onset_reference": onset_reference_raw,
        "samu_call": t_samu_call,
        "irm_arrival": t_irm_arrival,
        "imaging_start": t_imaging_start,
        "imaging_end": t_imaging_end,
        "needle": t_needle,
    }
    ordered_events = ["onset_reference", "samu_call", "irm_arrival", "imaging_start", "imaging_end", "needle"]
elif care_pathway == PATHWAY_INTRA:
    t_ed_arrival = st.text_input("Heure arrivée urgences (HH:MM)", value="", placeholder="ex: 09:05")
    t_neuro_call = st.text_input("Heure alerte interne (appel neurologue) (HH:MM)", value="", placeholder="ex: 09:10")
    t_imaging_start = st.text_input("Heure début IRM/imagerie (HH:MM)", value="", placeholder="ex: 09:35")
    t_imaging_end = st.text_input("Heure fin IRM/imagerie (HH:MM)", value="", placeholder="ex: 09:55")
    t_needle = st.text_input("Heure bolus rtPA (HH:MM)", value="", placeholder="ex: 10:05")

    time_inputs = {
        "onset_reference": onset_reference_raw,
        "ed_arrival": t_ed_arrival,
        "neuro_call": t_neuro_call,
        "imaging_start": t_imaging_start,
        "imaging_end": t_imaging_end,
        "needle": t_needle,
    }
    ordered_events = ["onset_reference", "ed_arrival", "neuro_call", "imaging_start", "imaging_end", "needle"]
else:
    t_other_hospital_call = st.text_input("Heure appel d'un autre hôpital (HH:MM)", value="", placeholder="ex: 08:50")
    t_irm_arrival = st.text_input("Heure arrivée IRM (HH:MM)", value="", placeholder="ex: 09:35")
    t_imaging_start = st.text_input("Heure début IRM/imagerie (HH:MM)", value="", placeholder="ex: 09:40")
    t_imaging_end = st.text_input("Heure fin IRM/imagerie (HH:MM)", value="", placeholder="ex: 09:55")
    t_needle = st.text_input("Heure bolus rtPA (HH:MM)", value="", placeholder="ex: 10:05")

    time_inputs = {
        "onset_reference": onset_reference_raw,
        "other_hospital_call": t_other_hospital_call,
        "irm_arrival": t_irm_arrival,
        "imaging_start": t_imaging_start,
        "imaging_end": t_imaging_end,
        "needle": t_needle,
    }
    ordered_events = ["onset_reference", "other_hospital_call", "irm_arrival", "imaging_start", "imaging_end", "needle"]

resolved_times, rollover_notes, time_errors = build_datetimes_with_rollover(reference_date, time_inputs, ordered_events)
all_errors = manual_errors + time_errors

if all_errors:
    for err in all_errors:
        st.error(err)

if onset_notes:
    st.info("\n".join(onset_notes))

if rollover_notes:
    st.info("Une correction automatique de date a été appliquée (passage minuit).")

arrival_dt = resolved_times.get("irm_arrival") if care_pathway in [PATHWAY_SAMU, PATHWAY_OTHER] else resolved_times.get("ed_arrival")
imaging_start_dt = resolved_times.get("imaging_start")
needle_dt = resolved_times.get("needle")
internal_alert_dt = resolved_times.get("neuro_call") if care_pathway == PATHWAY_INTRA else None
samu_call_dt = resolved_times.get("samu_call") if care_pathway == PATHWAY_SAMU else None
other_hospital_call_dt = resolved_times.get("other_hospital_call") if care_pathway == PATHWAY_OTHER else None

metrics = {
    "dtn_min": minutes(arrival_dt, needle_dt),
    "door_to_imaging_min": minutes(arrival_dt, imaging_start_dt),
    "imaging_to_needle_min": minutes(imaging_start_dt, needle_dt),
    "samu_to_arrival_min": minutes(samu_call_dt, arrival_dt),
    "samu_to_needle_min": minutes(samu_call_dt, needle_dt),
    "other_hospital_to_arrival_min": minutes(other_hospital_call_dt, arrival_dt),
    "other_hospital_to_needle_min": minutes(other_hospital_call_dt, needle_dt),
    "internal_alert_to_arrival_min": minutes(internal_alert_dt, arrival_dt),
    "internal_alert_to_needle_min": minutes(internal_alert_dt, needle_dt),
    "onset_to_needle_min": minutes(resolved_times.get("onset_reference"), needle_dt),
}

metric_labels = {
    "dtn_min": "Arrivée -> Bolus (DTN) (min)",
    "door_to_imaging_min": "Arrivée -> Début imagerie (min)",
    "imaging_to_needle_min": "Début imagerie -> Bolus (min)",
    "samu_to_arrival_min": "Appel SAMU -> Arrivée (min)",
    "samu_to_needle_min": "Appel SAMU -> Bolus (min)",
    "other_hospital_to_arrival_min": "Appel autre hôpital -> Arrivée (min)",
    "other_hospital_to_needle_min": "Appel autre hôpital -> Bolus (min)",
    "internal_alert_to_arrival_min": "Alerte interne -> Arrivée (min)",
    "internal_alert_to_needle_min": "Alerte interne -> Bolus (min)",
    "onset_to_needle_min": "Début symptômes/dernière fois vue normale -> Bolus (min)",
}

st.subheader("Performance")
dtn = metrics.get("dtn_min")
if dtn is not None:
    if dtn <= 30:
        st.success(f"1️⃣ DTN excellent: {dtn} min (niveau centre expert 20-30 min).")
    elif dtn <= 45:
        st.success(f"1️⃣ DTN performant: {dtn} min (médiane équipe performante 30-45 min).")
    elif dtn <= 60:
        st.info(f"1️⃣ DTN dans l'objectif recommandé: {dtn} min (<= 60 min).")
    else:
        st.warning(f"1️⃣ DTN au-dessus de l'objectif: {dtn} min (> 60 min).")

d2i = metrics.get("door_to_imaging_min")
if d2i is not None:
    if d2i < 20:
        st.success(f"2️⃣ Arrivée->Imagerie très fluide: {d2i} min (< 20 min).")
    elif d2i <= 30:
        st.info(f"2️⃣ Arrivée->Imagerie conforme: {d2i} min (<= 20-30 min).")
    else:
        st.warning(f"2️⃣ Arrivée->Imagerie lente: {d2i} min (> 30 min).")

i2n = metrics.get("imaging_to_needle_min")
if i2n is not None:
    if i2n <= 25:
        st.success(f"2️⃣ Imagerie->Bolus optimisé: {i2n} min (15-25 min).")
    elif i2n <= 30:
        st.info(f"2️⃣ Imagerie->Bolus conforme: {i2n} min (<= 30 min).")
    else:
        st.warning(f"2️⃣ Imagerie->Bolus à améliorer: {i2n} min (> 30 min).")

if care_pathway == PATHWAY_SAMU:
    s2n = metrics.get("samu_to_needle_min")
    if s2n is not None:
        if s2n < 120:
            st.success(f"3️⃣ Extra-hospitalier SAMU->Bolus: {s2n} min (< 120 min).")
        else:
            st.info(f"3️⃣ Extra-hospitalier SAMU->Bolus: {s2n} min (dépend du délai symptomatique initial).")
elif care_pathway == PATHWAY_OTHER:
    h2n = metrics.get("other_hospital_to_needle_min")
    if h2n is not None:
        if h2n < 120:
            st.success(f"3️⃣ Appel autre hôpital->Bolus: {h2n} min (< 120 min).")
        else:
            st.info(f"3️⃣ Appel autre hôpital->Bolus: {h2n} min (dépend du délai symptomatique initial).")

o2n = metrics.get("onset_to_needle_min")
if o2n is not None:
    if o2n <= 150:
        st.success(f"4️⃣ Délai global excellent: {o2n} min (< 120-150 min).")
    elif o2n <= 270:
        st.info(f"4️⃣ Délai global dans la fenêtre standard: {o2n} min (<= 4h30).")
    else:
        st.warning(f"4️⃣ Délai global hors fenêtre standard: {o2n} min (> 4h30).")

commentaire = st.text_area("Commentaire (optionnel)", value="", placeholder="Ajouter un commentaire libre...")

st.subheader("Export")
notes = list(rollover_notes)
notes.extend(onset_notes)
notes.append(f"Parcours: {care_pathway}")
notes.append(f"Mode début: {onset_mode}")

row = {
    "care_pathway": care_pathway,
    "onset_source": onset_source,
    "ts_onset_known": fmt_dt(onset_known_dt),
    "ts_onset_unknown": fmt_dt(onset_unknown_dt),
    "ts_onset_reference": fmt_dt(resolved_times.get("onset_reference")),
    "ts_samu_call": fmt_dt(resolved_times.get("samu_call")),
    "ts_other_hospital_call": fmt_dt(resolved_times.get("other_hospital_call")),
    "ts_ed_arrival": fmt_dt(resolved_times.get("ed_arrival")),
    "ts_neuro_call": fmt_dt(resolved_times.get("neuro_call")),
    "ts_irm_arrival": fmt_dt(resolved_times.get("irm_arrival")),
    "ts_imaging_start": fmt_dt(resolved_times.get("imaging_start")),
    "ts_imaging_end": fmt_dt(resolved_times.get("imaging_end")),
    "ts_needle": fmt_dt(resolved_times.get("needle")),
    "commentaire": commentaire.strip(),
    **metrics,
    "auto_fix_enabled": True,
    "auto_correction_applied": bool(rollover_notes),
    "notes": " | ".join(notes),
    "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

csv = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")
st.download_button("Télécharger CSV (1 ligne)", data=csv, file_name="avc_delais_thrombolyse.csv", mime="text/csv")

st.subheader("Sauvegarde serveur")
st.caption("Enregistrement SQLite local + Google Sheets (3 onglets: SAMU, intra, autre hôpital).")

can_save = not all_errors and bool(resolved_times.get("onset_reference"))
if not can_save:
    st.warning("Corriger les champs en erreur pour pouvoir enregistrer.")

if st.button("Enregistrer ce patient", type="primary", disabled=not can_save):
    record = build_record(row)
    record_id = save_patient_record_sqlite(record)
    st.success(f"Données enregistrées (SQLite). ID: {record_id}")

    ok_sheet, msg_sheet = save_patient_record_google_sheet(record)
    if ok_sheet:
        st.success(f"Données envoyées sur Google Sheets {msg_sheet}.")
    else:
        st.warning(f"Google Sheets: {msg_sheet}")

recent_records = load_recent_records(limit=20)
if recent_records.empty:
    st.info("Aucun enregistrement sauvegardé pour le moment.")
else:
    display_df = recent_records.rename(
        columns={
            "created_at": "Enregistré le",
            "care_pathway": "Parcours",
            "onset_source": "Source début",
            "ts_onset_known": "Début connu",
            "ts_onset_unknown": "Dernière fois vue normale",
            "ts_onset_reference": "Début retenu",
            "ts_samu_call": "Appel SAMU",
            "ts_other_hospital_call": "Appel autre hôpital",
            "ts_ed_arrival": "Arrivée urgences",
            "ts_neuro_call": "Appel neuro",
            "ts_irm_arrival": "Arrivée IRM",
            "ts_imaging_start": "Début IRM",
            "ts_imaging_end": "Fin IRM",
            "ts_needle": "Bolus",
            "dtn_min": "Arrivée->Bolus (min)",
            "door_to_imaging_min": "Arrivée->Début imagerie (min)",
            "imaging_to_needle_min": "Début imagerie->Bolus (min)",
            "samu_to_arrival_min": "SAMU->Arrivée (min)",
            "samu_to_needle_min": "SAMU->Bolus (min)",
            "other_hospital_to_arrival_min": "Autre hôpital->Arrivée (min)",
            "other_hospital_to_needle_min": "Autre hôpital->Bolus (min)",
            "internal_alert_to_arrival_min": "Alerte interne->Arrivée (min)",
            "internal_alert_to_needle_min": "Alerte interne->Bolus (min)",
            "onset_to_needle_min": "Début->Bolus (min)",
            "commentaire": "Commentaire",
            "auto_correction_applied": "Correction auto appliquée",
        }
    )
    st.write(f"{len(display_df)} derniers enregistrements")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
