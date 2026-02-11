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
GSHEET_WORKSHEET = os.environ.get("GOOGLE_SHEET_WORKSHEET", "patient_records")

SHEET_COLUMNS = [
    "id",
    "created_at",
    "case_id",
    "ts_symptom_onset",
    "ts_arrival",
    "ts_imaging",
    "ts_needle",
    "ts_end_infusion",
    "ts_other",
    "odt_min",
    "d2i_min",
    "d2n_min",
    "i2n_min",
    "onset_to_needle_min",
    "needle_to_end_min",
    "auto_fix_enabled",
    "notes",
    "exported_at",
]

EVENT_LABELS = {
    "symptom_onset": "Début AVC / Dernière fois vue normale",
    "arrival": "Arrivée",
    "imaging": "Imagerie",
    "needle": "Bolus rtPA",
    "end_infusion": "Fin de perfusion",
    "other": "Autre",
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
                ts_symptom_onset TEXT,
                ts_arrival TEXT,
                ts_imaging TEXT,
                ts_needle TEXT,
                ts_end_infusion TEXT,
                ts_other TEXT,
                odt_min INTEGER,
                d2i_min INTEGER,
                d2n_min INTEGER,
                i2n_min INTEGER,
                onset_to_needle_min INTEGER,
                needle_to_end_min INTEGER,
                auto_fix_enabled INTEGER NOT NULL,
                notes TEXT,
                exported_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def build_record(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": str(uuid4()),
        "case_id": row.get("case_id", ""),
        "ts_symptom_onset": row.get("ts_symptom_onset", ""),
        "ts_arrival": row.get("ts_arrival", ""),
        "ts_imaging": row.get("ts_imaging", ""),
        "ts_needle": row.get("ts_needle", ""),
        "ts_end_infusion": row.get("ts_end_infusion", ""),
        "ts_other": row.get("ts_other", ""),
        "odt_min": row.get("odt_min"),
        "d2i_min": row.get("d2i_min"),
        "d2n_min": row.get("d2n_min"),
        "i2n_min": row.get("i2n_min"),
        "onset_to_needle_min": row.get("onset_to_needle_min"),
        "needle_to_end_min": row.get("needle_to_end_min"),
        "auto_fix_enabled": 1,
        "notes": row.get("notes", ""),
        "exported_at": row.get("exported_at", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_patient_record_sqlite(record: Dict[str, object]) -> str:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO patient_records (
                id, case_id, ts_symptom_onset, ts_arrival, ts_imaging, ts_needle, ts_end_infusion, ts_other,
                odt_min, d2i_min, d2n_min, i2n_min, onset_to_needle_min, needle_to_end_min,
                auto_fix_enabled, notes, exported_at, created_at
            ) VALUES (
                :id, :case_id, :ts_symptom_onset, :ts_arrival, :ts_imaging, :ts_needle, :ts_end_infusion, :ts_other,
                :odt_min, :d2i_min, :d2n_min, :i2n_min, :onset_to_needle_min, :needle_to_end_min,
                :auto_fix_enabled, :notes, :exported_at, :created_at
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

        try:
            ws = sheet.worksheet(GSHEET_WORKSHEET)
        except WorksheetNotFound:
            ws = sheet.add_worksheet(title=GSHEET_WORKSHEET, rows=2000, cols=len(SHEET_COLUMNS))

        header = ws.row_values(1)
        if not header:
            ws.append_row(SHEET_COLUMNS, value_input_option="RAW")

        ws.append_row([record.get(col, "") for col in SHEET_COLUMNS], value_input_option="RAW")
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


def load_recent_records(limit: int = 50) -> pd.DataFrame:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id, created_at, case_id, ts_symptom_onset, ts_arrival, ts_imaging, ts_needle, ts_end_infusion,
                odt_min, d2i_min, d2n_min, i2n_min, onset_to_needle_min, needle_to_end_min
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
st.caption("Saisie simplifiée: une date initiale puis les heures uniquement.")

with st.sidebar:
    st.header("Paramètres")
    st.write("Utiliser un identifiant pseudonymisé (pas de nom/prénom).")
    if GSHEET_ID:
        st.success("Google Sheets configuré")
    else:
        st.info("Google Sheets non configuré (fallback SQLite)")

st.subheader("Identifiant (optionnel)")
case_id = st.text_input("Case ID (pseudonymisé)", value="")

st.subheader("Recueil des horaires")
reference_date = st.date_input("Date du début AVC ou de la dernière fois vue normale", value=date.today())
reference_mode = st.radio(
    "Point de départ",
    options=["Heure début AVC connue", "Heure vue pour la dernière fois normale (LKW)"],
    horizontal=True,
)

label_reference_time = (
    "Heure début AVC (HH:MM)"
    if reference_mode == "Heure début AVC connue"
    else "Heure dernière fois vue normale (HH:MM)"
)

t_symptom = st.text_input(label_reference_time, value="", placeholder="ex: 23:10")

col1, col2 = st.columns(2)
with col1:
    t_arrival = st.text_input("Heure arrivée (HH:MM)", value="", placeholder="ex: 23:55")
    t_imaging = st.text_input("Heure imagerie (HH:MM)", value="", placeholder="ex: 00:20")
with col2:
    t_needle = st.text_input("Heure bolus rtPA (HH:MM)", value="", placeholder="ex: 01:05")
    t_end = st.text_input("Heure fin perfusion (HH:MM)", value="", placeholder="ex: 02:05")

t_other = st.text_input("Heure autre (optionnel, HH:MM)", value="", placeholder="")

time_inputs = {
    "symptom_onset": t_symptom,
    "arrival": t_arrival,
    "imaging": t_imaging,
    "needle": t_needle,
    "end_infusion": t_end,
    "other": t_other,
}

ordered_events = ["symptom_onset", "arrival", "imaging", "needle", "end_infusion", "other"]
times_corr, rollover_notes, time_errors = build_datetimes_with_rollover(reference_date, time_inputs, ordered_events)

if time_errors:
    for err in time_errors:
        st.error(err)

st.subheader("Horodatages résolus")
rows_times = []
for key in ordered_events:
    if key == "other" and not times_corr.get("other"):
        continue
    rows_times.append({"Étape": EVENT_LABELS[key], "Date/heure": fmt_dt(times_corr.get(key))})
st.dataframe(pd.DataFrame(rows_times), use_container_width=True, hide_index=True)

if rollover_notes:
    st.info("\n".join(rollover_notes))

metrics = {
    "odt_min": minutes(times_corr["symptom_onset"], times_corr["arrival"]),
    "d2i_min": minutes(times_corr["arrival"], times_corr["imaging"]),
    "d2n_min": minutes(times_corr["arrival"], times_corr["needle"]),
    "i2n_min": minutes(times_corr["imaging"], times_corr["needle"]),
    "onset_to_needle_min": minutes(times_corr["symptom_onset"], times_corr["needle"]),
    "needle_to_end_min": minutes(times_corr["needle"], times_corr["end_infusion"]),
}

metric_labels = {
    "odt_min": "Debut AVC/LKW -> Arrivee (min)",
    "d2i_min": "Arrivee -> Imagerie (min)",
    "d2n_min": "Arrivee -> Bolus (min)",
    "i2n_min": "Imagerie -> Bolus (min)",
    "onset_to_needle_min": "Debut AVC/LKW -> Bolus (min)",
    "needle_to_end_min": "Bolus -> Fin perfusion (min)",
}

st.subheader("Délais calculés")
metric_rows = [{"Délai": metric_labels[k], "Valeur (min)": v} for k, v in metrics.items()]
st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

st.subheader("Contrôles de cohérence")
checks = []


def add_check(name: str, ok: bool, detail: str) -> None:
    checks.append({"Contrôle": name, "Statut": "OK" if ok else "À vérifier", "Détail": detail})

if metrics["d2n_min"] is not None:
    add_check("Arrivée -> Bolus <= 60 min", metrics["d2n_min"] <= 60, f"{metrics['d2n_min']} min")
else:
    add_check("Arrivée -> Bolus <= 60 min", False, "Horaires incomplets")

if metrics["d2i_min"] is not None:
    add_check("Arrivée -> Imagerie <= 25 min", metrics["d2i_min"] <= 25, f"{metrics['d2i_min']} min")
else:
    add_check("Arrivée -> Imagerie <= 25 min", False, "Horaires incomplets")

for a, b in [("arrival", "imaging"), ("imaging", "needle")]:
    if times_corr[a] and times_corr[b]:
        add_check(
            f"{EVENT_LABELS[a]} <= {EVENT_LABELS[b]}",
            times_corr[a] <= times_corr[b],
            f"{fmt_dt(times_corr[a])} -> {fmt_dt(times_corr[b])}",
        )
    else:
        add_check(f"{EVENT_LABELS[a]} <= {EVENT_LABELS[b]}", False, "Horaires incomplets")

st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

st.subheader("Export")
notes = list(rollover_notes)
notes.append(f"Point de départ: {reference_mode}")
row = {
    "case_id": case_id,
    "ts_symptom_onset": fmt_dt(times_corr.get("symptom_onset")),
    "ts_arrival": fmt_dt(times_corr.get("arrival")),
    "ts_imaging": fmt_dt(times_corr.get("imaging")),
    "ts_needle": fmt_dt(times_corr.get("needle")),
    "ts_end_infusion": fmt_dt(times_corr.get("end_infusion")),
    "ts_other": fmt_dt(times_corr.get("other")),
    **metrics,
    "auto_fix_enabled": True,
    "notes": " | ".join(notes),
    "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

csv = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")
st.download_button("Télécharger CSV (1 ligne)", data=csv, file_name="avc_delais_thrombolyse.csv", mime="text/csv")

st.subheader("Sauvegarde serveur")
st.caption("Enregistrement SQLite local + Google Sheets (si configuré).")

can_save = not time_errors and bool(times_corr.get("symptom_onset"))
if not can_save:
    st.warning("Renseigner au minimum la date + heure de début AVC/LKW avec un format HH:MM valide.")

if st.button("Enregistrer ce patient", type="primary", disabled=not can_save):
    record = build_record(row)
    record_id = save_patient_record_sqlite(record)
    st.success(f"Données enregistrées (SQLite). ID: {record_id}")

    ok_sheet, msg_sheet = save_patient_record_google_sheet(record)
    if ok_sheet:
        st.success("Données envoyées sur Google Sheets.")
    else:
        st.warning(f"Google Sheets: {msg_sheet}")

recent_records = load_recent_records(limit=20)
if recent_records.empty:
    st.info("Aucun enregistrement sauvegardé pour le moment.")
else:
    display_df = recent_records.rename(
        columns={
            "created_at": "Enregistré le",
            "case_id": "Case ID",
            "ts_symptom_onset": "Début AVC/LKW",
            "ts_arrival": "Arrivée",
            "ts_imaging": "Imagerie",
            "ts_needle": "Bolus",
            "ts_end_infusion": "Fin perfusion",
            "odt_min": "Début->Arrivée (min)",
            "d2i_min": "Arrivée->Imagerie (min)",
            "d2n_min": "Arrivée->Bolus (min)",
            "i2n_min": "Imagerie->Bolus (min)",
            "onset_to_needle_min": "Début->Bolus (min)",
            "needle_to_end_min": "Bolus->Fin (min)",
        }
    )
    st.write(f"{len(display_df)} derniers enregistrements")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
