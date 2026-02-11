# app.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

# ---------------- Std / 3rd party imports ----------------
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestRegressor

# ---------------- Streamlit UI config ----------------
st.set_page_config(page_title="PoD Forecast Playground", page_icon="📈", layout="wide")
st.caption("App initialized — dataset loads on demand.")

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
MEASURES = ["OffPeakConsumption", "StandardConsumption", "PeakConsumption"]
DATA_PATHS = [
    Path("industrial_10yrs.csv"),
    Path("data/industrial_10yrs.csv")
]

XGB_BASE = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

RF_BASE = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

ARIMA_BASE = (1, 0, 1)
SARIMA_BASE = (1, 1, 0, 12)

DEFAULT_TEST_HORIZON = 12     # 12-month horizon
MIN_LAGS = 12                 # build 12 lags

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim/normalize column names and map expected canonical names."""
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    canon = {}
    for col in ["PodID", "ReportingMonth"] + MEASURES:
        low = col.lower()
        if low in colmap:
            canon[colmap[low]] = col
    return df.rename(columns=canon)


def make_sample_df(months: int = 36, seed: int = 42) -> pd.DataFrame:
    """Small synthetic sample (stable across runs with a fixed seed)."""
    rng = pd.date_range("2020-01-31", periods=months, freq="M")
    rs = np.random.RandomState(seed)
    rows = []
    pod = "SAMPLEPOD01.MEGAFLEX"
    for i, d in enumerate(rng):
        base = 2000 + 300*np.sin(2*np.pi*i/12) + rs.randn()*100
        rows.append([pod, d, base, base*0.7, base*0.4])
    return pd.DataFrame(rows, columns=["PodID", "ReportingMonth"] + MEASURES)


def to_month_end(series: pd.Series) -> pd.Series:
    """Aggregate to month-end index (sums if duplicates occur)."""
    idx = series.index.to_period("M").to_timestamp(how="end")
    s = pd.Series(series.values, index=idx)
    return s.groupby(s.index).sum(min_count=1)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.nanmean((y_true - y_pred)**2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def build_supervised(series: pd.Series, n_lags: int = MIN_LAGS) -> pd.DataFrame:
    """Create a supervised learning frame with N lags."""
    s = series.copy()
    df = pd.DataFrame(index=s.index)
    df["y"] = s
    for L in range(1, n_lags + 1):
        df[f"lag{L}"] = s.shift(L)
    return df.dropna()


def prepare_xy(series: pd.Series, horizon: int, n_lags: int = MIN_LAGS):
    """Split series into (X_train, y_train, X_test, y_test, test_index)."""
    df = build_supervised(series, n_lags=n_lags)
    if len(df) <= horizon:
        raise ValueError(
            f"Not enough data after creating {n_lags} lags to hold a test horizon of {horizon}."
        )
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    return (
        train.drop(columns=["y"]).values,
        train["y"].values,
        test.drop(columns=["y"]).values,
        test["y"].values,
        test.index
    )


@st.cache_data(show_spinner=True)
def load_backend_df():
    """Load and preprocess backend CSV, with light timing logs. Returns (df, note)."""
    t0 = time.time()
    for p in DATA_PATHS:
        if p.exists():
            # Read CSV
            df = pd.read_csv(p, dtype=str, low_memory=False)
            t_read = time.time()

            # Normalize / Types
            df = normalize_columns(df)

            # Validate required columns
            missing = [c for c in ["PodID", "ReportingMonth"] + MEASURES if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in bundled file {p}: {missing}")

            # Types
            df["ReportingMonth"] = pd.to_datetime(df["ReportingMonth"], errors="coerce")
            for m in MEASURES:
                df[m] = pd.to_numeric(
                    df[m].astype(str).str.replace(",", "", regex=False),
                    errors="coerce"
                )

            df = df.dropna(subset=["ReportingMonth", "PodID"]).copy()
            t_prep = time.time()

            # Simple timing messages (visible in the app)
            log = (
                f"📄 Loaded backend dataset: {p}  \n"
                f"• Read CSV: {t_read - t0:.2f}s  \n"
                f"• Preprocess: {t_prep - t_read:.2f}s  \n"
                f"• Total: {t_prep - t0:.2f}s"
            )
            return df, log

    # Fallback to synthetic data (cached)
    return make_sample_df(), "⚠️ Using synthetic sample — backend dataset not found."


@st.cache_data(show_spinner=False)
def pod_monthly(df: pd.DataFrame, pod: str) -> pd.DataFrame:
    """Monthly aggregation for a given PoD."""
    d = df[df["PodID"] == pod]
    if d.empty:
        return pd.DataFrame()
    return d.groupby("ReportingMonth")[MEASURES].sum(min_count=1).sort_index()


def plot_series(ax, dates, y_true, y_pred, title, subtitle=None):
    """Simple forecast plot helper."""
    ax.plot(dates, y_true, label="Actual", color="#1f77b4")
    ax.plot(dates, y_pred, label="Forecast", color="#ff7f0e")
    ax.set_title(title if subtitle is None else f"{title} — {subtitle}")
    ax.grid(alpha=0.3)
    ax.legend()


def _optional_import(module_name: str):
    """Import a module if available, else return (None, False)."""
    try:
        mod = __import__(module_name)
        return mod, True
    except Exception:
        return None, False


# ---------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "df" not in st.session_state:
    st.session_state.df = None
if "note" not in st.session_state:
    st.session_state.note = ""
if "pod_idx" not in st.session_state:
    st.session_state.pod_idx = 0

# ---------------------------------------------------------------------
# MAIN — Defer backend load until user clicks
# ---------------------------------------------------------------------
st.title("📈 PoD Forecast Playground")

load_block = st.container()
if not st.session_state.loaded:
    with load_block:
        st.info("Dataset is not loaded yet.")
        if st.button("Load backend dataset now", type="primary"):
            with st.spinner("Loading backend dataset..."):
                df, note = load_backend_df()
                st.session_state.df = df
                st.session_state.note = note
                st.session_state.loaded = True

# If still not loaded after interaction, stop here
if not st.session_state.loaded:
    st.stop()

# Dataset is available
df = st.session_state.df
st.caption(st.session_state.note)
st.caption(f"Rows: {len(df):,} | PoDs: {df['PodID'].nunique():,}")

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔎 Explore PoDs", "🧠 Model & Compare"])

# ============ EXPLORE TAB ============
with tab1:
    st.subheader("Browse PoDs")
    pods = sorted(df["PodID"].unique().tolist())

    c1, c2, c3 = st.columns([3, 1, 1])
    selected = c1.selectbox("Select PoD", pods, index=st.session_state.pod_idx)
    if c2.button("Prev"):
        st.session_state.pod_idx = (st.session_state.pod_idx - 1) % len(pods)
        st.rerun()
    if c3.button("Next"):
        st.session_state.pod_idx = (st.session_state.pod_idx + 1) % len(pods)
        st.rerun()

    monthly = pod_monthly(df, selected)
    if monthly.empty:
        st.warning("No data for this PoD.")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        for m in MEASURES:
            ax.plot(monthly.index, monthly[m], label=m)
        ax.set_title(f"PoD {selected}")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

# ============ MODEL TAB ============
with tab2:
    st.subheader("Model Playground")

    pods = sorted(df["PodID"].unique().tolist())
    c1, c2 = st.columns([3, 2])
    pod_for_model = c1.selectbox("PoD", pods, index=min(st.session_state.pod_idx, len(pods)-1))
    measure = c2.radio("Measure", MEASURES, horizontal=True)

    series_raw = pod_monthly(df, pod_for_model).get(measure, pd.Series(dtype=float))

    st.write(f"History: {len(series_raw)} months")

    # Clean series, aggregate to month-end, fill small gaps
    series_clean = to_month_end(series_raw).ffill().bfill()

    with st.expander("Preview series", expanded=False):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(series_raw.index, series_raw.values, label="Raw")
        ax.plot(series_clean.index, series_clean.values, label="Cleaned")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Guard: ensure enough observations for 12 lags + 12 horizon
    min_required = MIN_LAGS + DEFAULT_TEST_HORIZON + 1
    if len(series_clean) < min_required:
        st.warning(
            f"Not enough history for modeling. Need at least {min_required} observations "
            f"(12 lags + {DEFAULT_TEST_HORIZON}-month test horizon)."
        )

    # Try optional imports so tabs degrade gracefully
    xgb_mod, has_xgb = _optional_import("xgboost")
    sm_mod, has_sm = _optional_import("statsmodels")

    t_xgb, t_rf, t_arima, t_sarima = st.tabs(["🌲 XGB", "🌳 RF", "📐 ARIMA", "📏 SARIMA"])

    # ---------------- XGB ----------------
    with t_xgb:
        if not has_xgb:
            st.info("XGBoost not installed in this environment. Add `xgboost` to requirements to enable this tab.")
        elif len(series_clean) < min_required:
            st.info("Provide more history to enable XGB.")
        else:
            c1, c2, c3 = st.columns(3)
            n_est = c1.slider("n_estimators", 50, 1000, XGB_BASE["n_estimators"], 50, key="xgb_n")
            depth = c1.slider("max_depth", 2, 15, XGB_BASE["max_depth"], 1, key="xgb_d")
            lr    = c2.slider("learning_rate", 0.01, 0.5, XGB_BASE["learning_rate"], 0.01, key="xgb_lr")
            subs  = c2.slider("subsample", 0.5, 1.0, XGB_BASE["subsample"], 0.05, key="xgb_s")
            colsm = c3.slider("colsample_bytree", 0.5, 1.0, XGB_BASE["colsample_bytree"], 0.05, key="xgb_cs")

            if st.button("Run XGB"):
                try:
                    from xgboost import XGBRegressor
                    t0 = time.time()
                    X_train, y_train, X_test, y_test, dates = prepare_xy(series_clean, DEFAULT_TEST_HORIZON)
                    model = XGBRegressor(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        subsample=subs,
                        colsample_bytree=colsm,
                        random_state=42,
                        tree_method="hist"
                    )
                    with st.spinner("Training XGB..."):
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)

                    fig, ax = plt.subplots(figsize=(12, 4))
                    plot_series(ax, dates, y_test, pred, "XGB Forecast", f"depth={depth}")
                    st.pyplot(fig)
                    st.metric("RMSE", f"{rmse(y_test, pred):.2f}")
                    st.metric("MAPE", f"{mape(y_test, pred):.2f}%")
                    st.caption(f"⏱️ End-to-end: {time.time() - t0:.2f}s")
                except Exception as e:
                    st.error(f"XGB error: {e}")

    # ---------------- RF ----------------
    with t_rf:
        if len(series_clean) < min_required:
            st.info("Provide more history to enable RF.")
        else:
            c1, c2, c3 = st.columns(3)
            n_est = c1.slider("n_estimators", 50, 1000, RF_BASE["n_estimators"], 50, key="rf_n")
            depth = c1.slider("max_depth", 2, 20, RF_BASE["max_depth"], 1, key="rf_d")
            minsp = c2.slider("min_samples_split", 2, 20, RF_BASE["min_samples_split"], 1, key="rf_ms")
            minlf = c2.slider("min_samples_leaf", 1, 20, RF_BASE["min_samples_leaf"], 1, key="rf_ml")
            maxft = c3.slider("max_features", 0.1, 1.0, RF_BASE["max_features"], 0.05, key="rf_mf")
            boot  = c3.checkbox("bootstrap", RF_BASE["bootstrap"], key="rf_boot")

            if st.button("Run RF"):
                try:
                    t0 = time.time()
                    X_train, y_train, X_test, y_test, dates = prepare_xy(series_clean, DEFAULT_TEST_HORIZON)
                    model = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_split=minsp,
                        min_samples_leaf=minlf,
                        max_features=maxft,
                        bootstrap=boot,
                        random_state=42,
                        n_jobs=-1
                    )
                    with st.spinner("Training RF..."):
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)

                    fig, ax = plt.subplots(figsize=(12, 4))
                    plot_series(ax, dates, y_test, pred, "RF Forecast", f"depth={depth}")
                    st.pyplot(fig)
                    st.metric("RMSE", f"{rmse(y_test, pred):.2f}")
                    st.metric("MAPE", f"{mape(y_test, pred):.2f}%")
                    st.caption(f"⏱️ End-to-end: {time.time() - t0:.2f}s")
                except Exception as e:
                    st.error(f"RF error: {e}")

    # ---------------- ARIMA ----------------
    with t_arima:
        if not has_sm:
            st.info("Statsmodels not installed. Add `statsmodels` to requirements to enable ARIMA.")
        elif len(series_clean) < DEFAULT_TEST_HORIZON + 5:
            st.info("Provide more history to enable ARIMA.")
        else:
            c1, c2, c3 = st.columns(3)
            p = c1.slider("p", 0, 10, ARIMA_BASE[0], 1, key="arima_p")
            d = c2.slider("d", 0, 2, ARIMA_BASE[1], 1, key="arima_d")
            q = c3.slider("q", 0, 10, ARIMA_BASE[2], 1, key="arima_q")
            trend = st.selectbox("Trend", ["none", "c", "t"], index=1, key="arima_trend")
            trend = None if trend == "none" else trend
            use_log = st.checkbox("log1p transform", False, key="arima_log")

            if st.button("Run ARIMA"):
                try:
                    import statsmodels.api as sm
                    y = series_clean.copy()
                    y_mod = np.log1p(np.clip(y, 0, None)) if use_log else y
                    y_train = y_mod.iloc[:-DEFAULT_TEST_HORIZON]
                    y_test  = y.iloc[-DEFAULT_TEST_HORIZON:]

                    with st.spinner("Fitting ARIMA (via SARIMAX with fixed seasonal part)..."):
                        model = sm.tsa.SARIMAX(
                            y_train,
                            order=(p, d, q),
                            seasonal_order=SARIMA_BASE,  # retains 12M seasonality
                            trend=trend,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            initialization="approximate_diffuse"
                        )
                        res = model.fit(disp=False, maxiter=300)

                    fc = res.get_forecast(DEFAULT_TEST_HORIZON).predicted_mean
                    pred = np.expm1(fc) if use_log else fc

                    fig, ax = plt.subplots(figsize=(12, 4))
                    plot_series(ax, y_test.index, y_test.values, pred.values,
                                "ARIMA Forecast", f"{(p, d, q)}")
                    st.pyplot(fig)
                    st.metric("RMSE", f"{rmse(y_test, pred):.2f}")
                    st.metric("MAPE", f"{mape(y_test, pred):.2f}%")
                except Exception as e:
                    st.error(f"ARIMA error: {e}")

    # ---------------- SARIMA ----------------
    with t_sarima:
        if not has_sm:
            st.info("Statsmodels not installed. Add `statsmodels` to requirements to enable SARIMA.")
        elif len(series_clean) < DEFAULT_TEST_HORIZON + 5:
            st.info("Provide more history to enable SARIMA.")
        else:
            c1, c2, c3 = st.columns(3)
            p_s = c1.slider("p", 0, 10, 1, 1, key="sarima_p")
            d_s = c2.slider("d", 0, 2, 1, 1, key="sarima_d")
            q_s = c3.slider("q", 0, 10, 0, 1, key="sarima_q")

            c4, c5, c6, c7 = st.columns(4)
            P_s = c4.slider("P", 0, 10, 1, 1, key="sarima_P")
            D_s = c5.slider("D", 0, 2, 1, 1, key="sarima_D")
            Q_s = c6.slider("Q", 0, 10, 0, 1, key="sarima_Q")
            s_s = c7.slider("s", 6, 24, 12, 1, key="sarima_s")

            trend_s = st.selectbox("Trend", ["none", "c", "t"], index=1, key="sarima_trend")
            trend_s = None if trend_s == "none" else trend_s
            use_log_s = st.checkbox("log1p transform", False, key="sarima_log")

            if st.button("Run SARIMA"):
                try:
                    import statsmodels.api as sm
                    y = series_clean.copy()
                    y_mod = np.log1p(np.clip(y, 0, None)) if use_log_s else y
                    y_train = y_mod.iloc[:-DEFAULT_TEST_HORIZON]
                    y_test  = y.iloc[-DEFAULT_TEST_HORIZON:]

                    with st.spinner("Fitting SARIMA..."):
                        model = sm.tsa.SARIMAX(
                            y_train,
                            order=(p_s, d_s, q_s),
                            seasonal_order=(P_s, D_s, Q_s, s_s),
                            trend=trend_s,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            initialization="approximate_diffuse"
                        )
                        res = model.fit(disp=False, maxiter=300)

                    fc = res.get_forecast(DEFAULT_TEST_HORIZON).predicted_mean
                    pred = np.expm1(fc) if use_log_s else fc

                    fig, ax = plt.subplots(figsize=(12, 4))
                    plot_series(ax, y_test.index, y_test.values, pred.values,
                                "SARIMA Forecast",
                                f"{(p_s, d_s, q_s)} × {(P_s, D_s, Q_s, s_s)}")
                    st.pyplot(fig)
                    st.metric("RMSE", f"{rmse(y_test, pred):.2f}")
                    st.metric("MAPE", f"{mape(y_test, pred):.2f}%")
                except Exception as e:
                    st.error(f"SARIMA error: {e}")

# ------------------ Footer / About ------------------
with st.expander("About this app", expanded=False):
    st.markdown(
        """
**PoD Forecast Playground**  
- Loads backend CSV on-demand and caches results.  
- Explore PoDs and visualize monthly consumption by measure.  
- Compare quick baseline models: XGB (optional), Random Forest, ARIMA/SARIMA (optional).  
- Uses 12 lags and a 12-month holdout by default.  
"""
    )