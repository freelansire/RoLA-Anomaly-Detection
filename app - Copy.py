import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.marine_sim import make_stream
from src.usgs_source import fetch_usgs_iv, PARAMS as USGS_PARAMS
from src.uk_hydrology_source import (
    search_stations as uk_search_stations,
    fetch_station_bundle as uk_fetch_station_bundle,
    OBSERVED_PROPERTIES as UK_OBS,
)
from src.mv_bayes_niw import OnlineNIW
from src.conformal import ConformalThreshold
from src.drift import PageHinkley
from src.corr_gate import CorrGate

st.set_page_config(page_title="RoLA V2 Live Stream Demo", layout="wide")
st.title("RoLA V2 — Live Streaming Demo")
st.caption("Synthetic + USGS + UK Hydrology (EA/Defra) • NIW→Student-t • Conformal • Page–Hinkley • Corr gate")


# -------------------------
# Helpers
# -------------------------
def stable_hash(obj) -> str:
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def coverage_report(df: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in candidate_cols:
        if c not in df.columns:
            rows.append({"feature": c, "present": False, "non_null": 0, "coverage_pct": 0.0})
        else:
            nn = int(df[c].notna().sum())
            rows.append({"feature": c, "present": True, "non_null": nn, "coverage_pct": round(100.0 * nn / max(n, 1), 2)})
    return pd.DataFrame(rows).sort_values(["present", "non_null"], ascending=[False, False])

def autoselect_features(df: pd.DataFrame, preferred: list[str], min_points: int = 80) -> list[str]:
    return [c for c in preferred if c in df.columns and df[c].notna().sum() >= min_points]

def align_to_grid(df: pd.DataFrame, cols: list[str], freq: str = "15min") -> pd.DataFrame:
    if "datetime" not in df.columns or df["datetime"].isna().all():
        return df
    out = df.sort_values("datetime").set_index("datetime")
    use_cols = [c for c in cols if c in out.columns]
    if not use_cols:
        return df
    out = out[use_cols].resample(freq).mean().ffill().reset_index()
    out["t"] = np.arange(len(out), dtype=int)
    if "label" not in out.columns:
        out["label"] = 0
    return out

def first_valid_row(df: pd.DataFrame, feats: list[str]) -> int | None:
    mask = df[feats].notna().all(axis=1)
    idx = np.where(mask.values)[0]
    return int(idx[0]) if len(idx) else None


# -------------------------
# Caches
# -------------------------
@st.cache_data(show_spinner=False)
def load_synth(_n: int, _seed: int) -> pd.DataFrame:
    return make_stream(n=int(_n), seed=int(_seed))

@st.cache_data(show_spinner=False)
def load_usgs(_site: str, _period: str, _codes: tuple) -> pd.DataFrame:
    return fetch_usgs_iv(site=_site, period=_period, parameter_cds=list(_codes))

@st.cache_data(show_spinner=False)
def load_uk_station_list(_search: str, _observed_property: str | None) -> pd.DataFrame:
    return uk_search_stations(search=_search, observed_property=_observed_property, limit=60)

@st.cache_data(show_spinner=False)
def load_uk_bundle(_station_guid: str, _props: tuple, _min_date: str, _max_date: str) -> pd.DataFrame:
    return uk_fetch_station_bundle(_station_guid, list(_props), _min_date, _max_date)


# -------------------------
# Prep functions
# -------------------------
def prepare_usgs_df(df_usgs: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df_usgs.copy())

    # Normalise names
    rename_map = {
        "oxygen_mgL": "oxygen",
        "oxygen_saturation_pct": "oxygen_saturation",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Flow: prefer metric if available
    if "flow_rate_m3s" in df.columns:
        df["flow_rate"] = df["flow_rate_m3s"]
    elif "flow_rate" not in df.columns:
        df["flow_rate"] = np.nan

    for c in ["temperature", "turbidity", "conductivity", "oxygen", "oxygen_saturation", "flow_rate"]:
        if c not in df.columns:
            df[c] = np.nan

    if "salinity" not in df.columns:
        df["salinity"] = np.nan

    df = df.sort_values("datetime").reset_index(drop=True)
    df["t"] = np.arange(len(df), dtype=int)
    df["label"] = 0

    cols = ["temperature", "turbidity", "conductivity", "oxygen", "oxygen_saturation", "flow_rate", "salinity"]
    df.loc[:, cols] = df.loc[:, cols].ffill()
    return df

def prepare_uk_df(df_uk: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df_uk.copy())
    if "datetime" not in df.columns:
        raise ValueError("UK Hydrology response missing datetime column.")

    # Add placeholders to match your “marine-ish” schema
    if "salinity" not in df.columns:
        df["salinity"] = np.nan

    df = df.sort_values("datetime").reset_index(drop=True)
    df["t"] = np.arange(len(df), dtype=int)
    df["label"] = 0

    # Forward-fill common columns if present
    for c in ["temperature", "turbidity", "conductivity", "flow_rate", "water_level", "oxygen_mgL", "oxygen_saturation_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill()
    return df


# -------------------------
# Model init
# -------------------------
def init_state(df: pd.DataFrame, feats: list[str], forget: float, alpha: float, gate_corr: float):
    first = first_valid_row(df, feats)
    if first is None:
        raise ValueError(
            "No row contains all selected features after alignment/ffill. "
            "Use compatibility check, choose fewer features, or increase lookback window."
        )

    x0 = df.loc[first, feats].values.astype(float)
    d = len(feats)

    st.session_state.df = df
    st.session_state.features = feats
    st.session_state.i = int(first)
    st.session_state.running = False
    st.session_state.history = []
    st.session_state.anoms = []
    st.session_state.run_id = None
    st.session_state.last_save_ts = None

    st.session_state.model = OnlineNIW(d=d, mu0=x0, forget=float(forget))
    st.session_state.calib = ConformalThreshold(alpha=float(alpha), window=320, warmup=80)
    st.session_state.drift = PageHinkley(delta=0.01, lamb=6.0)
    st.session_state.gate = CorrGate(d=d, gate_corr=float(gate_corr))


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Data Source")

    source = st.selectbox(
        "Source",
        ["Synthetic", "USGS (US, realtime)", "UK Hydrology (EA/Defra)"],
        index=0
    )

    # --- Shared controls ---
    st.divider()
    st.subheader("Time alignment + coverage")
    grid_freq = st.selectbox("Align to grid", ["None", "5min", "15min", "60min"], index=2)
    min_points = st.slider("Min points per feature", 10, 500, 80, 10)
    show_check = st.checkbox("Show compatibility report", value=True)

    st.divider()
    st.header("Model Controls")
    alpha = st.slider("Conformal alpha", 0.001, 0.05, 0.01, 0.001)
    forget = st.slider("Forgetting factor", 0.95, 0.999, 0.995, 0.001)
    gate_corr = st.slider("Correlation gate threshold", 0.0, 0.95, 0.6, 0.05)

    st.divider()
    st.header("Stream Controls")
    speed = st.slider("Stream speed (ticks/sec)", 1, 50, 15, 1)
    batch = st.slider("Samples per tick", 1, 50, 3, 1)

    st.divider()
    st.header("Run Controls")
    cA, cB, cC = st.columns(3)
    with cA:
        start = st.button("▶ Start / Resume", use_container_width=True)
    with cB:
        pause = st.button("⏸ Pause", use_container_width=True)
    with cC:
        reset = st.button("🔁 Reset", use_container_width=True)

    st.divider()
    autosave = st.checkbox("Enable autosave to disk", value=False)
    out_dir = st.text_input("Autosave folder", value="outputs") if autosave else None

    # --- Per-source controls (kept AFTER shared, so config-hash sees them) ---
    st.divider()
    if source == "Synthetic":
        n = st.slider("Stream length (synthetic)", 200, 10000, 1800, 50)
        seed = st.number_input("Random seed", 0, 9999, 7, 1)
        usgs_site = usgs_period = usgs_codes = None
        uk_search = uk_station = uk_days = uk_props = None
        chosen_features_ui = None

    elif source.startswith("USGS"):
        n = seed = None
        usgs_site = st.text_input("USGS Site ID", value="08158000").strip()
        usgs_period = st.selectbox("Lookback", ["P1D", "P7D", "P30D"], index=1)
        default_codes = ["00010", "63680", "00060", "00095", "00300", "00301"]
        usgs_codes = st.multiselect(
            "Parameter codes",
            options=list(USGS_PARAMS.keys()),
            default=default_codes,
            format_func=lambda c: f"{c} — {USGS_PARAMS.get(c, c)}"
        )
        uk_search = uk_station = uk_days = uk_props = None
        chosen_features_ui = None

    else:
        n = seed = None
        usgs_site = usgs_period = usgs_codes = None

        uk_search = st.text_input("Search stations (river/town/station)", value="thames")
        uk_days = st.selectbox("Lookback days", [1, 3, 7, 14, 30], index=2)

        # Let user pick which observedProperties to attempt (UK Hydrology API can provide many)
        uk_props = st.multiselect(
            "Signals to request (UK Hydrology)",
            options=list(UK_OBS.keys()),
            default=["waterFlow", "temperature", "conductivity", "turbidity", "dissolved-oxygen"],
            format_func=lambda k: f"{k} → {UK_OBS[k]}"
        )

        # Station list: filter by one chosen signal (first) to reduce junk results
        filter_prop = uk_props[0] if uk_props else None
        stations_df = load_uk_station_list(uk_search, filter_prop)

        if stations_df is None or len(stations_df) == 0:
            uk_station = ""
            st.warning("No stations found. Try a different search term (e.g., 'thames', 'severn', 'trent').")
        else:
            label_options = [
                f"{r['label']}  •  {r.get('notation','')}  •  {r.get('stationGuid','')}"
                for _, r in stations_df.iterrows()
            ]
            pick = st.selectbox("Pick a station", label_options, index=0)
            uk_station = stations_df.iloc[label_options.index(pick)]["stationGuid"]

        chosen_features_ui = None


# -------------------------
# Load / (Re)Init state on config change
# -------------------------
config = {
    "source": source,
    "n": n,
    "seed": seed,
    "grid": grid_freq,
    "min_points": min_points,
    "usgs_site": usgs_site,
    "usgs_period": usgs_period,
    "usgs_codes": tuple(usgs_codes) if usgs_codes is not None else None,
    "uk_search": uk_search,
    "uk_station": uk_station,
    "uk_days": uk_days,
    "uk_props": tuple(uk_props) if uk_props is not None else None,
    "alpha": alpha,
    "forget": forget,
    "gate_corr": gate_corr,
}
config_hash = stable_hash(config)

if reset or ("config_hash" not in st.session_state) or (st.session_state.config_hash != config_hash):
    st.cache_data.clear()
    st.session_state.config_hash = config_hash

    # ---- Load df ----
    if source == "Synthetic":
        df_loaded = load_synth(int(n), int(seed))

        # Synthetic untouched: keep your canonical features
        feats = ["temperature", "turbidity", "oxygen", "salinity"]
        init_state(df_loaded, feats, forget=forget, alpha=alpha, gate_corr=gate_corr)

    elif source.startswith("USGS"):
        if not usgs_site:
            st.error("Enter a USGS site ID.")
            st.stop()
        if not usgs_codes:
            st.error("Pick at least one USGS parameter code.")
            st.stop()

        df_raw = load_usgs(usgs_site, usgs_period, tuple(usgs_codes))
        df_loaded = prepare_usgs_df(df_raw)

        preferred = ["temperature", "turbidity", "oxygen", "conductivity", "flow_rate", "oxygen_saturation"]
        if grid_freq != "None":
            df_loaded = align_to_grid(df_loaded, preferred + ["salinity"], freq=grid_freq)

        rep = coverage_report(df_loaded, preferred)
        auto_feats = autoselect_features(df_loaded, preferred, min_points=min_points)
        if len(auto_feats) < 2:
            auto_feats = [c for c in preferred if c in df_loaded.columns and df_loaded[c].notna().sum() > 0][:2]

        feats = st.sidebar.multiselect(
            "Features to use (USGS)",
            options=[c for c in preferred if c in df_loaded.columns],
            default=auto_feats,
        )

        if show_check:
            st.sidebar.dataframe(rep, use_container_width=True)

        if not feats:
            st.error("No usable features selected (USGS). Try a different site/period or fewer features.")
            st.stop()

        init_state(df_loaded, feats, forget=forget, alpha=alpha, gate_corr=gate_corr)

    else:
        if not uk_station:
            st.error("Pick a UK station (or change search term).")
            st.stop()
        if not uk_props:
            st.error("Pick at least one UK signal to request.")
            st.stop()

        end = datetime.utcnow().date()
        startd = end - timedelta(days=int(uk_days))
        min_date = startd.strftime("%Y-%m-%d")
        max_date = end.strftime("%Y-%m-%d")

        df_raw = load_uk_bundle(uk_station, tuple(uk_props), min_date, max_date)
        df_loaded = prepare_uk_df(df_raw)

        # Map UK columns to a “nice” uniform feature set
        # (some stations won’t have both oxygen_mgL and oxygen_saturation_pct)
        preferred = [
            "temperature",
            "turbidity",
            "conductivity",
            "flow_rate",
            "water_level",
            "oxygen_mgL",
            "oxygen_saturation_pct",
        ]
        if grid_freq != "None":
            df_loaded = align_to_grid(df_loaded, preferred + ["salinity"], freq=grid_freq)

        rep = coverage_report(df_loaded, preferred)
        auto_feats = autoselect_features(df_loaded, preferred, min_points=min_points)
        if len(auto_feats) < 2:
            auto_feats = [c for c in preferred if c in df_loaded.columns and df_loaded[c].notna().sum() > 0][:2]

        feats = st.sidebar.multiselect(
            "Features to use (UK Hydrology)",
            options=[c for c in preferred if c in df_loaded.columns],
            default=auto_feats,
        )

        if show_check:
            st.sidebar.dataframe(rep, use_container_width=True)

        if not feats:
            st.error("No usable features selected (UK). Increase lookback days or pick fewer signals/features.")
            st.stop()

        init_state(df_loaded, feats, forget=forget, alpha=alpha, gate_corr=gate_corr)


# live update hyperparams mid-run
st.session_state.model.forget = float(forget)
st.session_state.calib.alpha = float(alpha)
st.session_state.gate.gate_corr = float(gate_corr)

if start:
    st.session_state.running = True
if pause:
    st.session_state.running = False


# -------------------------
# Streaming step
# -------------------------
def step_once():
    df = st.session_state.df
    feats = st.session_state.features
    model = st.session_state.model
    calib = st.session_state.calib
    drift = st.session_state.drift
    gate = st.session_state.gate

    i = int(st.session_state.i)
    if i >= len(df):
        st.session_state.running = False
        return

    row = df.iloc[i]
    x = row[feats].values.astype(float)

    if not np.isfinite(x).all():
        st.session_state.i += 1
        return

    gate.update(x)

    nll, _ = model.score(x)
    th = calib.threshold()
    is_raw_anom = (nll > th) if calib.ready() else False

    _, mu, Sigma = model.predictive_params()
    resid = np.abs(x - mu)
    scale = np.sqrt(np.maximum(np.diag(Sigma), 1e-9))
    sensor_flags = resid > (3.2 * scale)

    gated_any, gate_reason = gate.gate(sensor_flags)
    is_anom = bool(is_raw_anom and (gated_any or sensor_flags.sum() > 1))

    drifted = bool(drift.update(float(nll)))

    model.update(x)
    if not is_anom:
        calib.update(float(nll))

    rec = {
        "t": int(row.get("t", i)),
        "datetime": row.get("datetime", pd.NaT),
        "nll": float(nll),
        "thr": float(calib.threshold()),
        "anom": int(is_anom),
        "drift": int(drifted),
        "gate_reason": gate_reason,
        "label": int(row.get("label", 0)) if pd.notna(row.get("label", 0)) else 0,
    }
    for f in feats:
        rec[f] = float(row[f])

    st.session_state.history.append(rec)

    if is_anom:
        st.session_state.anoms.append({
            "t": rec["t"],
            "datetime": rec["datetime"],
            "nll": rec["nll"],
            "threshold": rec["thr"],
            "gate": gate_reason,
            **{f: rec[f] for f in feats},
        })

    st.session_state.i += 1


if st.session_state.running:
    for _ in range(int(batch)):
        step_once()


# -------------------------
# UI
# -------------------------
col1, col2 = st.columns([2.1, 1.0], gap="large")
with col1:
    st.subheader("Streaming anomaly score (NLL) vs conformal threshold")
    chart_placeholder = st.empty()
    markers_placeholder = st.empty()

with col2:
    st.subheader("Run status")
    status_placeholder = st.empty()
    st.subheader("Recent anomalies")
    anom_table_placeholder = st.empty()

st.divider()
st.subheader("Sensor snapshot (current)")
snap_placeholder = st.empty()

hist_df = pd.DataFrame(st.session_state.history)
anom_df = pd.DataFrame(st.session_state.anoms)

df = st.session_state.df
feats = st.session_state.features
i = int(st.session_state.i)

with status_placeholder.container():
    st.metric("Source", source)
    st.metric("Features", ", ".join(feats))
    st.metric("Progress", f"{min(i, len(df))}/{len(df)}")
    if len(hist_df) > 0:
        st.metric("Current NLL", f"{hist_df['nll'].iloc[-1]:.3f}")
        st.metric("Current Threshold", f"{hist_df['thr'].iloc[-1]:.3f}")
        st.metric("Anomalies flagged", int(hist_df["anom"].sum()))
        st.metric("Drift triggers", int(hist_df["drift"].sum()))

snap_i = min(max(i - 1, 0), len(df) - 1)
row = df.iloc[snap_i]
snap_df = pd.DataFrame({"sensor": feats, "value": [row[f] for f in feats]})
if "datetime" in df.columns and pd.notna(row.get("datetime", pd.NaT)):
    snap_df.insert(0, "datetime", [row["datetime"]] * len(feats))
snap_placeholder.dataframe(snap_df, use_container_width=True)

if len(hist_df) > 10:
    plot_df = hist_df.copy()
    if "t" not in plot_df.columns:
        plot_df["t"] = np.arange(len(plot_df))
    plot_df = plot_df[["t", "nll", "thr", "anom", "drift"]].set_index("t")
    chart_placeholder.line_chart(plot_df[["nll", "thr"]], height=320)

    drift_times = plot_df.index[plot_df["drift"] == 1].to_list()[-10:]
    anom_times = plot_df.index[plot_df["anom"] == 1].to_list()[-10:]
    markers_placeholder.info(
        f"Recent drift triggers: {drift_times if drift_times else 'None'}\n\n"
        f"Recent anomalies: {anom_times if anom_times else 'None'}"
    )
else:
    chart_placeholder.info("Click **Start / Resume** to begin streaming.")

if len(anom_df) > 0:
    anom_table_placeholder.dataframe(anom_df.tail(10), use_container_width=True)
else:
    anom_table_placeholder.caption("No anomalies flagged yet.")

st.divider()
st.subheader("Export results")

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download history (CSV)",
        data=hist_df.to_csv(index=False).encode("utf-8"),
        file_name="rola_v2_history.csv",
        mime="text/csv",
        disabled=(len(hist_df) == 0),
        use_container_width=True,
    )
with c2:
    st.download_button(
        "Download anomalies (CSV)",
        data=anom_df.to_csv(index=False).encode("utf-8"),
        file_name="rola_v2_anomalies.csv",
        mime="text/csv",
        disabled=(len(anom_df) == 0),
        use_container_width=True,
    )

if autosave and out_dir:
    now = time.time()
    last = st.session_state.get("last_save_ts", None)
    if (last is None) or (now - last > 3.0):
        out_root = Path(out_dir)
        run_id = st.session_state.get("run_id")
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.run_id = run_id
        run_path = out_root / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        hist_df.to_csv(run_path / "history.csv", index=False)
        anom_df.to_csv(run_path / "anomalies.csv", index=False)
        st.session_state.last_save_ts = now
        st.caption(f"Autosaving to: {run_path}")

if st.session_state.running:
    time.sleep(1.0 / float(max(speed, 1)))
    st.rerun()
