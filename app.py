import streamlit as st
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
import os

pwd = os.environ.get("APP_PASSWORD", "")
if pwd:
    p = st.text_input("Mot de passe", type="password")
    if p != pwd:
        st.stop()

st.set_page_config(page_title="AVC Hyperaigu – Délais Thrombolyse", layout="centered")

# -----------------------------
# Helpers temps / corrections
# -----------------------------
def parse_dt(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Parse date (YYYY-MM-DD) + time (HH:MM) into datetime.
    Returns None if empty.
    """
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None

def ensure_monotonic(times: Dict[str, Optional[datetime]], order: List[str], max_backshift_hours: int = 24) -> Tuple[Dict[str, Optional[datetime]], List[str]]:
    """
    Tries to fix day-rollover errors:
    - If an event is earlier than the previous event, assume it might be the next day (+24h).
    - Applies iteratively along the expected order.
    """
    corrected = dict(times)
    notes = []
    prev_key = None
    for key in order:
        if corrected.get(key) is None:
            prev_key = key
            continue
        # Find previous non-null
        j = order.index(key) - 1
        prev_dt = None
        prev_name = None
        while j >= 0:
            pk = order[j]
            if corrected.get(pk) is not None:
                prev_dt = corrected[pk]
                prev_name = pk
                break
            j -= 1

        if prev_dt is None:
            prev_key = key
            continue

        cur_dt = corrected[key]
        if cur_dt < prev_dt:
            # Candidate fix: add 24h until it's >= prev_dt, up to max_backshift_hours
            tmp = cur_dt
            n = 0
            while tmp < prev_dt and n < (max_backshift_hours // 24 + 1):
                tmp = tmp + timedelta(days=1)
                n += 1
            if tmp >= prev_dt:
                corrected[key] = tmp
                notes.append(f"🛠️ {key} semblait avant {prev_name} → +{n} jour(s) appliqué(s).")
            else:
                notes.append(f"⚠️ {key} < {prev_name} et correction automatique impossible.")
        prev_key = key

    return corrected, notes

def minutes(a: Optional[datetime], b: Optional[datetime]) -> Optional[int]:
    if a is None or b is None:
        return None
    return int((b - a).total_seconds() // 60)

def fmt_dt(x: Optional[datetime]) -> str:
    return "" if x is None else x.strftime("%Y-%m-%d %H:%M")

# -----------------------------
# UI
# -----------------------------
st.title("AVC hyperaigu – délais thrombolyse (MVP)")
st.caption("Saisie d’horodatages → corrections simples → calcul automatique des délais → export CSV.")

with st.sidebar:
    st.header("Paramètres")
    auto_fix = st.toggle("Corrections automatiques (jour qui change)", value=True)
    st.write("Ordre logique attendu : symptômes → arrivée → imagerie → bolus → fin perfusion.")
    st.divider()
    st.write("Conseil : utiliser un identifiant pseudonymisé (pas de nom/prénom).")

st.subheader("Identifiant (optionnel)")
case_id = st.text_input("Case ID (pseudonymisé)", value="")

st.subheader("Horodatages")
c1, c2 = st.columns(2)

with c1:
    d_sym = st.date_input("Date début symptômes (ou LKW)", value=None)
    t_sym = st.text_input("Heure début symptômes (HH:MM)", value="", placeholder="ex: 08:15")

    d_arr = st.date_input("Date arrivée (Door)", value=None)
    t_arr = st.text_input("Heure arrivée (HH:MM)", value="", placeholder="ex: 09:02")

    d_img = st.date_input("Date imagerie (CT/MR)", value=None)
    t_img = st.text_input("Heure imagerie (HH:MM)", value="", placeholder="ex: 09:20")

with c2:
    d_needle = st.date_input("Date bolus rtPA (Needle)", value=None)
    t_needle = st.text_input("Heure bolus rtPA (HH:MM)", value="", placeholder="ex: 09:45")

    d_end = st.date_input("Date fin perfusion", value=None)
    t_end = st.text_input("Heure fin perfusion (HH:MM)", value="", placeholder="ex: 10:45")

    d_other = st.date_input("Date autre (optionnel)", value=None)
    t_other = st.text_input("Heure autre (HH:MM)", value="", placeholder="")

# Parse inputs
times_raw = {
    "symptom_onset": parse_dt(str(d_sym) if d_sym else "", t_sym.strip()),
    "arrival":       parse_dt(str(d_arr) if d_arr else "", t_arr.strip()),
    "imaging":       parse_dt(str(d_img) if d_img else "", t_img.strip()),
    "needle":        parse_dt(str(d_needle) if d_needle else "", t_needle.strip()),
    "end_infusion":  parse_dt(str(d_end) if d_end else "", t_end.strip()),
    "other":         parse_dt(str(d_other) if d_other else "", t_other.strip()),
}

expected_order = ["symptom_onset", "arrival", "imaging", "needle", "end_infusion"]

# Apply correction
if auto_fix:
    times_corr, notes = ensure_monotonic(times_raw, expected_order)
else:
    times_corr, notes = times_raw, []

st.subheader("Horodatages (après correction)")
df_times = pd.DataFrame(
    [{"event": k, "datetime": fmt_dt(v)} for k, v in times_corr.items() if k != "other"] +
    ([{"event": "other", "datetime": fmt_dt(times_corr.get("other"))}] if times_corr.get("other") else [])
)
st.dataframe(df_times, use_container_width=True, hide_index=True)

if notes:
    st.info("\n".join(notes))

# Compute metrics
metrics = {
    "ODT (Onset→Door) min": minutes(times_corr["symptom_onset"], times_corr["arrival"]),
    "D2I (Door→Imaging) min": minutes(times_corr["arrival"], times_corr["imaging"]),
    "D2N (Door→Needle) min": minutes(times_corr["arrival"], times_corr["needle"]),
    "I2N (Imaging→Needle) min": minutes(times_corr["imaging"], times_corr["needle"]),
    "Onset→Needle min": minutes(times_corr["symptom_onset"], times_corr["needle"]),
    "Needle→End min": minutes(times_corr["needle"], times_corr["end_infusion"]),
}

st.subheader("Délais calculés")
df_metrics = pd.DataFrame([{"metric": k, "value_min": v} for k, v in metrics.items()])
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# Simple plausibility checks
st.subheader("Contrôles de cohérence")
checks = []
def add_check(name, ok, detail):
    checks.append({"check": name, "status": "OK" if ok else "⚠️", "detail": detail})

# Example thresholds (editable)
D2N_TARGET = 60
D2I_TARGET = 25

d2n = metrics["D2N (Door→Needle) min"]
d2i = metrics["D2I (Door→Imaging) min"]

if d2n is not None:
    add_check("Door-to-Needle ≤ 60 min", d2n <= D2N_TARGET, f"{d2n} min")
else:
    add_check("Door-to-Needle ≤ 60 min", False, "Horodatages incomplets")

if d2i is not None:
    add_check("Door-to-Imaging ≤ 25 min", d2i <= D2I_TARGET, f"{d2i} min")
else:
    add_check("Door-to-Imaging ≤ 25 min", False, "Horodatages incomplets")

# Monotonicity checks
for a, b in [("arrival", "imaging"), ("imaging", "needle")]:
    if times_corr[a] and times_corr[b]:
        add_check(f"{a} ≤ {b}", times_corr[a] <= times_corr[b], f"{fmt_dt(times_corr[a])} → {fmt_dt(times_corr[b])}")
    else:
        add_check(f"{a} ≤ {b}", False, "Horodatages incomplets")

st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

# Export
st.subheader("Export")
row = {
    "case_id": case_id,
    **{f"ts_{k}": fmt_dt(v) for k, v in times_corr.items()},
    **{k: v for k, v in metrics.items()},
    "auto_fix_enabled": auto_fix,
    "notes": " | ".join(notes),
    "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
}
df_export = pd.DataFrame([row])

csv = df_export.to_csv(index=False).encode("utf-8")
st.download_button("Télécharger CSV (1 ligne)", data=csv, file_name="avc_delais_thrombolyse.csv", mime="text/csv")


