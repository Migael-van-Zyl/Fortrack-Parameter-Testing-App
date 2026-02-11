# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app: PoD Explorer + Interactive Forecast Playground
Author: Migael Van Zyl
Date: 2026-02-10
"""

import warnings
warnings.filterwarnings("ignore")

import io
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PoD Forecast Playground", page_icon="📈", layout="wide")
st.caption("✅ App initialized — select a data source, PoD and model to run.")

# -----------------------------------------------------------------------------
# CONSTANTS & BASELINES
# -----------------------------------------------------------------------------
MEASURES = ["OffPeakConsumption", "StandardConsumption", "PeakConsumption"]

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
    "n_jobs": -1
}

ARIMA_BASE = (1, 0, 1)            # (p,d,q)
SARIMA_BASE = (1, 1, 0, 12)       # (P,D,Q,s)
DEFAULT_TEST_HORIZON = 12

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces and unify exact expected column names if they differ only by case/spacing."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # Try gentle normalization (case-insensitive exact matches)
    colmap = {c.lower(): c for c in df.columns}
    required = ["podid", "reportingmonth"] + [m.lower() for m in MEASURES]
    missing = [r for r in required if r not in colmap]
    # If case-insensitive names exist, reassign to our canonical names
    canon_map = {}
    for name in ["PodID", "ReportingMonth"] + MEASURES:
        low = name.lower()
        if low in colmap:
            canon_map[colmap[low]] = name
    if canon_map:
        df = df.rename(columns=canon_map)
    return df

@st.cache_data(show_spinner=False)
def load_anyfile_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Robust loader: handles CSV/TXT (any delimiter), Excel (xlsx/xls), and zipped CSV.
    Returns raw DataFrame with strings in measure cols; downstream will convert.
    """
    name = (filename or "").lower()
    bio = io.BytesIO(file_bytes)

    # Try Excel
    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(bio, dtype=str)  # requires openpyxl/xlrd depending on format
            return _normalize_columns(df)
        except Exception as ex:
            raise RuntimeError(f"Failed to read Excel file. Try saving as CSV. Details: {ex}")

    # Try zipped CSV
    if name.endswith(".zip"):
        try:
            df = pd.read_csv(bio, dtype=str, compression="zip", engine="python")
            return _normalize_columns(df)
        except Exception as ex:
            raise RuntimeError(f"Failed to read zipped CSV. Details: {ex}")

    # Default: CSV/TXT with delimiter sniffing
    try:
        # Python engine + sep=None to sniff delimiters (comma/semicolon/tab)
        df = pd.read_csv(bio, dtype=str, sep=None, engine="python", on_bad_lines="skip")
        return _normalize_columns(df)
    except Exception as ex:
        # Fallback encodings
        for enc in ("utf-8", "latin1"):
            try:
                bio.seek(0)
                df = pd.read_csv(bio, dtype=str, sep=None, engine="python",
                                 on_bad_lines="skip", encoding=enc)
                return _normalize_columns(df)
            except Exception:
                continue
        raise RuntimeError(f"Failed to read as CSV/TXT. Tip: Save as UTF-8 CSV. Details: {ex}")

def to_month_end(series: pd.Series) -> pd.Series:
    """Normalize timestamps to month-end without reindexing (prevents mass NaNs)."""
    idx = series.index.to_period("M").to_timestamp(how="end")
    s = pd.Series(series.values, index=idx)
    return s.groupby(s.index).sum(min_count=1).sort_index()

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

def mape_percent(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def plot_series(ax, dates, actual, pred, title, label_pred):
    ax.plot(dates, actual, label="Actual", color="black", linewidth=2)
    ax.plot(dates, pred, label=label_pred, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Consumption (kWh)")
    ax.grid(alpha=0.3)
    ax.legend()

def clean_preview_series(y: pd.Series, mask_negatives=True, ffill=3) -> pd.Series:
    """Light preview cleaning. ARIMA/SARIMA use their own gentle cleaning flow."""
    z = y.copy()
    if mask_negatives:
        z = z.mask(z < 0, np.nan)
    z = to_month_end(z)
    return z.ffill(limit=ffill).bfill(limit=ffill)

def build_supervised(series: pd.Series, mask_negatives=True) -> pd.DataFrame:
    s = series.mask(series < 0, np.nan) if mask_negatives else series.copy()
    df = pd.DataFrame(index=s.index); df["y"] = s
    for L in range(1, 13):
        df[f"lag{L}"] = s.shift(L)
    return df.dropna()

def prepare_xy(series: pd.Series, horizon: int, mask_negatives=True):
    df_sup = build_supervised(series, mask_negatives)
    if len(df_sup) <= horizon:
        raise ValueError(
            f"Not enough rows after lagging ({len(df_sup)}) for test horizon={horizon}. "
            f"Need at least 12 + horizon months."
        )
    train = df_sup.iloc[:-horizon]
    test  = df_sup.iloc[-horizon:]
    return (
        train.drop(columns=["y"]).values,
        train["y"].values,
        test.drop(columns=["y"]).values,
        test["y"].values,
        test.index,
    )

@st.cache_data(show_spinner=False)
def postload_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist, convert types.
    Returns a cleaned DataFrame suitable for the app.
    """
    if df.empty:
        raise ValueError("Uploaded file is empty.")

    # Normalize names and check required columns
    df = _normalize_columns(df)
    missing_cols = []
    for name in ["PodID", "ReportingMonth"] + MEASURES:
        if name not in df.columns:
            missing_cols.append(name)
    if missing_cols:
        cols_str = ", ".join(missing_cols)
        have_str = ", ".join(df.columns)
        raise ValueError(
            f"Missing required column(s): {cols_str}.\n"
            f"Found columns: {have_str}\n"
            f"Tip: Make sure your headers match exactly (case‑insensitive is OK; we'll map)."
        )

    # Types
    df["ReportingMonth"] = pd.to_datetime(df["ReportingMonth"], errors="coerce")
    for c in MEASURES:
        # Remove commas if present, then parse numeric
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df = df.dropna(subset=["ReportingMonth", "PodID"]).copy()
    return df

@st.cache_data(show_spinner=False)
def pod_monthly(df: pd.DataFrame, pod_id: str) -> pd.DataFrame:
    d = df[df["PodID"] == pod_id]
    if d.empty:
        return pd.DataFrame()
    return d.groupby("ReportingMonth")[MEASURES].sum(min_count=1).sort_index()

def fmt_mb(nbytes: int) -> str:
    if nbytes is None or math.isnan(nbytes): return "-"
    return f"{nbytes/1024/1024:.2f} MB"

def make_sample_df(n_pods: int = 3, months: int = 60) -> pd.DataFrame:
    """Small synthetic sample to demo the app when upload is tricky."""
    rng = pd.date_range("2019-01-31", periods=months, freq="M")
    rows = []
    for i in range(n_pods):
        pod = f"SAMPLEPOD{i+1:03d}.MEGAFLEX"
        base = np.linspace(1000, 2000, months) + 200*np.sin(2*np.pi*np.arange(months)/12)
        noise = 100*np.random.randn(months)
        off = np.maximum(base + noise, 100)
        std = np.maximum(base*0.7 + noise*0.8, 50)
        peak = np.maximum(base*0.4 + noise*0.6, 20)
        for d, a, b, c in zip(rng, off, std, peak):
            rows.append([pod, d, a, b, c])
    sdf = pd.DataFrame(rows, columns=["PodID", "ReportingMonth"] + MEASURES)
    return sdf

# -----------------------------------------------------------------------------
# SIDEBAR – Data source & settings
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

data_source = st.sidebar.radio(
    "Data source",
    ["Upload file", "Use sample data"],
    index=0,
    help="If upload fails (type/encoding/size), switch to sample data to demo."
)

uploaded_file = None
if data_source == "Upload file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV / TXT / Excel / zipped CSV",
        type=["csv", "txt", "xlsx", "xls", "zip"],
        accept_multiple_files=False,
        key="uploader_main"
    )
else:
    st.sidebar.success("Using built-in sample dataset.")

test_horizon = st.sidebar.slider("Test horizon (months)", 6, 24, DEFAULT_TEST_HORIZON)
mask_negatives = st.sidebar.checkbox("Mask negative values", True, help="Treat negatives as missing.")
ffill_limit = st.sidebar.slider("Short forward/backfill limit", 0, 12, 3)

# -----------------------------------------------------------------------------
# MAIN – Load data
# -----------------------------------------------------------------------------
st.title("📈 PoD Forecast Playground")

# Load the DataFrame depending on data source
df = None
if data_source == "Upload file":
    if not uploaded_file:
        st.info("⬆️ Upload a file to proceed, or switch to **Use sample data** in the sidebar.")
        st.stop()

    # Info about the uploaded file
    size_info = getattr(uploaded_file, "size", None)
    name_info = getattr(uploaded_file, "name", "uploaded")
    st.caption(f"📄 Uploaded: **{name_info}** ({fmt_mb(size_info)})")

    try:
        content = uploaded_file.getvalue()
        # Cache by bytes + filename for stable keys
        raw_df = load_anyfile_cached(content, name_info)
        df = postload_clean(raw_df)
    except Exception as ex:
        st.error(f"❌ Could not load the file: {ex}")
        st.stop()

else:
    # Sample data path
    df = make_sample_df()
    st.caption("📄 Using a small synthetic sample data set.")

# Basic validation
if df is None or df.empty:
    st.error("No rows to display after loading/validation.")
    st.stop()

pods = sorted(df["PodID"].unique().tolist())
if "pod_idx" not in st.session_state:
    st.session_state.pod_idx = 0

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_explore, tab_models = st.tabs(["🔎 Explore PoDs", "🧠 Model & Compare"])

# ==========================
# EXPLORE TAB
# ==========================
with tab_explore:
    st.subheader("Browse PoDs")

    col1, col2, col3 = st.columns([3,1,1])
    selected_pod = col1.selectbox("Select PoD", pods, index=st.session_state.pod_idx)

    if col2.button("Prev"):
        st.session_state.pod_idx = (st.session_state.pod_idx - 1) % len(pods)
        st.rerun()
    if col3.button("Next"):
        st.session_state.pod_idx = (st.session_state.pod_idx + 1) % len(pods)
        st.rerun()

    monthly = pod_monthly(df, selected_pod)
    if monthly.empty:
        st.warning("No data for this PoD.")
    else:
        fig, ax = plt.subplots(figsize=(12,5))
        for m in MEASURES:
            ax.plot(monthly.index, monthly[m], label=m)
        ax.set_title(f"PoD {selected_pod} – Monthly Consumption")
        ax.set_xlabel("Month"); ax.set_ylabel("kWh")
        ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig, clear_figure=True)

        # Small preview
        with st.expander("Preview first 5 rows"):
            st.dataframe(monthly.head())

# ==========================
# MODEL TAB
# ==========================
with tab_models:
    st.subheader("Model Playground")

    col1, col2 = st.columns([3,2])
    pod_for_model = col1.selectbox("PoD (for modeling)", pods)
    measure = col2.radio("Measure", MEASURES, horizontal=True)

    series_raw = pod_monthly(df, pod_for_model)[measure]
    st.write(f"History length: **{len(series_raw)} months**")

    series_clean = clean_preview_series(series_raw, mask_negatives, ffill=ffill_limit)
    with st.expander("Show Cleaned Preview"):
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(series_raw.index, series_raw.values, label="Raw", alpha=0.6)
        ax.plot(series_clean.index, series_clean.values, label="Normalized Preview", color="black")
        ax.set_title(f"{pod_for_model} / {measure}")
        ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig, clear_figure=True)

    # Model Tabs
    tab_xgb, tab_rf, tab_arima, tab_sarima = st.tabs(["🌲 XGBoost", "🌳 Random Forest", "📐 ARIMA", "📏 Full SARIMA"])

    # ===========================================
    # XGBOOST (lazy import inside button)
    # ===========================================
    with tab_xgb:
        col1, col2, col3 = st.columns(3)
        n_est = col1.slider("n_estimators", 50, 1000, XGB_BASE["n_estimators"], 50, key="xgb_n")
        max_depth = col1.slider("max_depth", 2, 15, XGB_BASE["max_depth"], 1, key="xgb_d")
        lr = col2.slider("learning_rate", 0.01, 0.5, float(XGB_BASE["learning_rate"]), 0.01, key="xgb_lr")
        subs = col2.slider("subsample", 0.5, 1.0, float(XGB_BASE["subsample"]), 0.05, key="xgb_s")
        colsample = col3.slider("colsample_bytree", 0.5, 1.0, float(XGB_BASE["colsample_bytree"]), 0.05, key="xgb_cs")

        if st.button("Run XGBoost"):
            try:
                try:
                    from xgboost import XGBRegressor  # lazy import
                except Exception as ex:
                    st.error(f"`xgboost` not available: {ex}")
                    st.stop()

                X_train, y_train, X_test, y_test, dates = prepare_xy(series_raw, test_horizon, mask_negatives)
                model = XGBRegressor(
                    n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
                    subsample=subs, colsample_bytree=colsample, random_state=42, tree_method="hist"
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(ax, dates, y_test, pred, "XGBoost Forecast", f"depth={max_depth}, est={n_est}")
                st.pyplot(fig)

                colA, colB = st.columns(2)
                colA.metric("RMSE", f"{rmse(y_test,pred):,.2f}")
                colB.metric("MAPE", f"{mape_percent(y_test,pred):.2f}%")

            except Exception as ex:
                st.error(f"XGBoost run failed: {ex}")

    # ===========================================
    # RANDOM FOREST
    # ===========================================
    with tab_rf:
        col1, col2, col3 = st.columns(3)
        n_est = col1.slider("n_estimators", 50, 1000, RF_BASE["n_estimators"], 50, key="rf_n")
        max_depth = col1.slider("max_depth", 2, 20, RF_BASE["max_depth"], 1, key="rf_d")
        min_split = col2.slider("min_samples_split", 2, 20, RF_BASE["min_samples_split"], 1, key="rf_ms")
        min_leaf = col2.slider("min_samples_leaf", 1, 20, RF_BASE["min_samples_leaf"], 1, key="rf_ml")
        max_feat = col3.slider("max_features", 0.1, 1.0, float(RF_BASE["max_features"]), 0.05, key="rf_mf")
        boot = col3.checkbox("bootstrap", RF_BASE["bootstrap"], key="rf_boot")

        if st.button("Run Random Forest"):
            try:
                X_train, y_train, X_test, y_test, dates = prepare_xy(series_raw, test_horizon, mask_negatives)
                model = RandomForestRegressor(
                    n_estimators=n_est, max_depth=max_depth,
                    min_samples_split=min_split, min_samples_leaf=min_leaf,
                    max_features=max_feat, bootstrap=boot, random_state=42, n_jobs=-1
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(ax, dates, y_test, pred, "Random Forest Forecast", f"depth={max_depth}, est={n_est}")
                st.pyplot(fig)

                colA, colB = st.columns(2)
                colA.metric("RMSE", f"{rmse(y_test,pred):,.2f}")
                colB.metric("MAPE", f"{mape_percent(y_test,pred):,.2f}%")

            except Exception as ex:
                st.error(f"Random Forest run failed: {ex}")

    # ===========================================
    # ARIMA (lazy import inside button)
    # ===========================================
    with tab_arima:
        col1, col2, col3 = st.columns(3)
        p = col1.slider("p", 0, 10, ARIMA_BASE[0], 1, key="arima_p")
        d = col2.slider("d", 0, 2, ARIMA_BASE[1], 1, key="arima_d")
        q = col3.slider("q", 0, 10, ARIMA_BASE[2], 1, key="arima_q")

        trend = st.selectbox("Trend", ["none", "c", "t"], index=1, key="arima_trend")
        trend = None if trend == "none" else trend
        use_log = st.checkbox("Use log1p transform", False, key="arima_log")

        if st.button("Run ARIMA"):
            try:
                import statsmodels.api as sm  # lazy import

                y = to_month_end(series_raw)
                if mask_negatives:
                    y = y.mask(y < 0, np.nan)
                y = y.ffill(limit=ffill_limit).bfill(limit=ffill_limit)

                y_mod = np.log1p(np.clip(y, 0, None)) if use_log else y

                if len(y_mod) <= test_horizon + 24:
                    st.warning("Series may be short for stable ARIMA; results could be noisy.")

                y_train = y_mod.iloc[:-test_horizon]
                y_test  = y.iloc[-test_horizon:]

                model = sm.tsa.SARIMAX(
                    y_train,
                    order=(p, d, q),
                    seasonal_order=SARIMA_BASE,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization="approximate_diffuse",
                )
                res = model.fit(disp=False, maxiter=500)
                fc = res.get_forecast(test_horizon).predicted_mean

                pred = np.expm1(fc) if use_log else fc
                pred = np.asarray(pred.values, dtype=float)

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(ax, y_test.index, y_test.values, pred,
                            "ARIMA Forecast", f"({p},{d},{q}), trend={trend or 'None'}")
                st.pyplot(fig)

                colA, colB = st.columns(2)
                colA.metric("RMSE", f"{rmse(y_test,pred):,.2f}")
                colB.metric("MAPE", f"{mape_percent(y_test,pred):,.2f}%")

            except Exception as ex:
                st.error(f"ARIMA run failed: {ex}")

    # ===========================================
    # FULL SARIMA: (p,d,q,P,D,Q,s) (lazy import inside button)
    # ===========================================
    with tab_sarima:
        st.markdown("### Full SARIMA Model")

        c1, c2, c3 = st.columns(3)
        p_s = c1.slider("p", 0, 10, ARIMA_BASE[0], 1, key="sarima_p")
        d_s = c2.slider("d", 0, 2, ARIMA_BASE[1], 1, key="sarima_d")
        q_s = c3.slider("q", 0, 10, ARIMA_BASE[2], 1, key="sarima_q")

        c4, c5, c6, c7 = st.columns(4)
        P_s = c4.slider("P", 0, 10, SARIMA_BASE[0], 1, key="sarima_P")
        D_s = c5.slider("D", 0, 2, SARIMA_BASE[1], 1, key="sarima_D")
        Q_s = c6.slider("Q", 0, 10, SARIMA_BASE[2], 1, key="sarima_Q")
        s_s = c7.slider("s", 6, 24, SARIMA_BASE[3], 1, key="sarima_s")

        trend_s = st.selectbox("Trend", ["none", "c", "t"], index=1, key="sarima_trend")
        trend_s = None if trend_s == "none" else trend_s
        use_log_s = st.checkbox("Use log1p transform", False, key="sarima_log")

        if st.button("Run SARIMA"):
            try:
                import statsmodels.api as sm  # lazy import

                y = to_month_end(series_raw)
                if mask_negatives:
                    y = y.mask(y < 0, np.nan)
                y = y.ffill(limit=ffill_limit).bfill(limit=ffill_limit)

                y_mod = np.log1p(np.clip(y, 0, None)) if use_log_s else y

                if len(y_mod) <= test_horizon + 24:
                    st.warning("Series may be short for stable SARIMA; results could be noisy.")

                y_train = y_mod.iloc[:-test_horizon]
                y_test  = y.iloc[-test_horizon:]

                model = sm.tsa.SARIMAX(
                    y_train,
                    order=(p_s, d_s, q_s),
                    seasonal_order=(P_s, D_s, Q_s, s_s),
                    trend=trend_s,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization="approximate_diffuse",
                )
                res = model.fit(disp=False, maxiter=500)
                fc = res.get_forecast(test_horizon).predicted_mean

                pred = np.expm1(fc) if use_log_s else fc
                pred = np.asarray(pred.values, dtype=float)

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(
                    ax, y_test.index, y_test.values, pred,
                    "SARIMA Forecast",
                    f"({p_s},{d_s},{q_s}) x ({P_s},{D_s},{Q_s},{s_s}), trend={trend_s or 'None'}"
                )
                st.pyplot(fig)

                colA, colB = st.columns(2)
                colA.metric("RMSE", f"{rmse(y_test,pred):,.2f}")
                colB.metric("MAPE", f"{mape_percent(y_test,pred):,.2f}%")

            except Exception as ex:
                st.error(f"SARIMA run failed: {ex}")