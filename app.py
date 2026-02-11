# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app: PoD Explorer + Interactive Forecast Playground
Author: Migael Van Zyl
Date: 2026-02-10
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# Optional: XGBoost
# -----------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    XGBRegressor = None
    HAS_XGB = False

from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PoD Forecast Playground", page_icon="📈", layout="wide")

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
@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file, dtype=str)
    df["ReportingMonth"] = pd.to_datetime(df["ReportingMonth"], errors="coerce")
    for c in MEASURES:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return df.dropna(subset=["ReportingMonth", "PodID"])

def to_month_end(series: pd.Series):
    idx = series.index.to_period("M").to_timestamp(how="end")
    s = pd.Series(series.values, index=idx)
    return s.groupby(s.index).sum(min_count=1).sort_index()

def rmse(y_true, y_pred):
    return float(np.sqrt(np.nanmean((np.asarray(y_true)-np.asarray(y_pred))**2)))

def mape_percent(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)

def plot_series(ax, dates, actual, pred, title, label_pred):
    ax.plot(dates, actual, label="Actual", color="black", linewidth=2)
    ax.plot(dates, pred, label=label_pred, color="#1f77b4")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

def clean_preview_series(y, mask_negatives=True, ffill=3):
    z = y.copy()
    if mask_negatives: z = z.mask(z < 0, np.nan)
    z = to_month_end(z)
    return z.ffill(limit=ffill).bfill(limit=ffill)

def build_supervised(series, mask_negatives=True):
    s = series.mask(series < 0, np.nan) if mask_negatives else series.copy()
    df = pd.DataFrame(index=s.index); df["y"] = s
    for L in range(1, 13):
        df[f"lag{L}"] = s.shift(L)
    return df.dropna()

def prepare_xy(series, horizon, mask_negatives=True):
    df_sup = build_supervised(series, mask_negatives)
    train = df_sup.iloc[:-horizon]
    test  = df_sup.iloc[-horizon:]
    return (
        train.drop(columns=["y"]).values,
        train["y"].values,
        test.drop(columns=["y"]).values,
        test["y"].values,
        test.index,
    )

def pod_monthly(df, pod):
    return df[df["PodID"] == pod].groupby("ReportingMonth")[MEASURES].sum(min_count=1).sort_index()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
test_horizon = st.sidebar.slider("Test horizon", 6, 24, DEFAULT_TEST_HORIZON)
mask_negatives = st.sidebar.checkbox("Mask negative values", True)
ffill_limit = st.sidebar.slider("Short forward/backfill limit", 0, 12, 3)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
st.title("📈 PoD Forecast Playground")

if not file:
    st.info("Upload `industrial_10yrs.csv` to proceed.")
    st.stop()

# Load data
df = load_csv(file)
pods = sorted(df["PodID"].unique())

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
        ax.set_title(f"PoD {selected_pod}")
        ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig, clear_figure=True)

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
        ax.plot(series_raw.index, series_raw, label="Raw", alpha=0.6)
        ax.plot(series_clean.index, series_clean, label="Normalized", color="black")
        ax.grid(alpha=0.3); ax.legend()
        st.pyplot(fig, clear_figure=True)

    # Model Tabs
    tab_xgb, tab_rf, tab_arima, tab_sarima = st.tabs(["🌲 XGBoost", "🌳 Random Forest", "📐 ARIMA", "📏 Full SARIMA"])

    # ===========================================
    # XGBOOST
    # ===========================================
    with tab_xgb:
        if not HAS_XGB:
            st.info("Install xgboost to use this model.")
        else:
            col1, col2, col3 = st.columns(3)
            n_est = col1.slider("n_estimators", 50, 1000, XGB_BASE["n_estimators"], 50, key="xgb_n")
            max_depth = col1.slider("max_depth", 2, 15, XGB_BASE["max_depth"], 1, key="xgb_d")
            lr = col2.slider("learning_rate", 0.01, 0.5, float(XGB_BASE["learning_rate"]), 0.01, key="xgb_lr")
            subs = col2.slider("subsample", 0.5, 1.0, float(XGB_BASE["subsample"]), 0.05, key="xgb_s")
            colsample = col3.slider("colsample_bytree", 0.5, 1.0, float(XGB_BASE["colsample_bytree"]), 0.05, key="xgb_cs")

            if st.button("Run XGBoost"):
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
            colB.metric("MAPE", f"{mape_percent(y_test,pred):.2f}%")

    # ===========================================
    # ARIMA
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
            y = to_month_end(series_raw)
            if mask_negatives: y = y.mask(y<0, np.nan)
            y = y.ffill(limit=ffill_limit).bfill(limit=ffill_limit)

            y_mod = np.log1p(np.clip(y,0,None)) if use_log else y

            y_train = y_mod.iloc[:-test_horizon]
            y_test  = y.iloc[-test_horizon:]

            model = sm.tsa.SARIMAX(
                y_train, order=(p,d,q), seasonal_order=SARIMA_BASE,
                trend=trend, enforce_stationarity=False, enforce_invertibility=False,
                initialization="approximate_diffuse"
            )
            res = model.fit(disp=False,maxiter=500)
            fc = res.get_forecast(test_horizon).predicted_mean

            pred = np.expm1(fc) if use_log else fc
            pred = np.asarray(pred.values,float)

            fig, ax = plt.subplots(figsize=(12,4))
            plot_series(ax, y_test.index, y_test.values, pred,
                        "ARIMA Forecast", f"({p},{d},{q}), trend={trend}")
            st.pyplot(fig)

            colA, colB = st.columns(2)
            colA.metric("RMSE", f"{rmse(y_test,pred):,.2f}")
            colB.metric("MAPE", f"{mape_percent(y_test,pred):.2f}%")

    # ===========================================
    # FULL SARIMA: (p,d,q,P,D,Q,s)
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

        trend_s = st.selectbox("Trend", ["none","c","t"], index=1, key="sarima_trend")
        trend_s = None if trend_s == "none" else trend_s

        use_log_s = st.checkbox("Use log1p transform", False, key="sarima_log")

        if st.button("Run SARIMA"):
            y = to_month_end(series_raw)
            if mask_negatives: y = y.mask(y<0, np.nan)
            y = y.ffill(limit=ffill_limit).bfill(limit=ffill_limit)

            y_mod = np.log1p(np.clip(y,0,None)) if use_log_s else y

            y_train = y_mod.iloc[:-test_horizon]
            y_test  = y.iloc[-test_horizon:]

            model = sm.tsa.SARIMAX(
                y_train,
                order=(p_s,d_s,q_s),
                seasonal_order=(P_s,D_s,Q_s,s_s),
                trend=trend_s,
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization="approximate_diffuse",
            )
            res = model.fit(disp=False,maxiter=500)
            fc = res.get_forecast(test_horizon).predicted_mean

            pred = np.expm1(fc) if use_log_s else fc
            pred = np.asarray(pred.values,float)

            fig, ax = plt.subplots(figsize=(12,4))
            plot_series(ax, y_test.index, y_test.values, pred,
                        "SARIMA Forecast",
                        f"({p_s},{d_s},{q_s}) x ({P_s},{D_s},{Q_s},{s_s}) trend={trend_s}")
            st.pyplot(fig)

            colA, colB = st.columns(2)
            colA.metric("RMSE", f"{rmse(y_test,pred):,.2f}")
            colB.metric("MAPE", f"{mape_percent(y_test,pred):.2f}%")