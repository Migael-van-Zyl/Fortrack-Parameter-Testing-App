# app.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

# Streamlit UI config
st.set_page_config(page_title="PoD Forecast Playground", page_icon="📈", layout="wide")
st.caption("App initialized — backend dataset will be loaded when you click the button.")

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
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
DEFAULT_TEST_HORIZON = 12

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def normalize_columns(df):
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    canon = {}
    for col in ["PodID", "ReportingMonth"] + MEASURES:
        low = col.lower()
        if low in colmap:
            canon[colmap[low]] = col
    return df.rename(columns=canon)

@st.cache_data(show_spinner=True)
def load_backend_df():
    for p in DATA_PATHS:
        if p.exists():
            df = pd.read_csv(
                p,
                dtype=str,
                low_memory=False
            )
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
            return df, f"📄 Loaded backend dataset: {p}"

    # Fallback: small synthetic sample
    return make_sample_df(), "Using synthetic sample — backend dataset not found."

def make_sample_df(months=36):
    rng = pd.date_range("2020-01-31", periods=months, freq="M")
    rows = []
    pod = "SAMPLEPOD01.MEGAFLEX"
    for i, d in enumerate(rng):
        base = 2000 + 300*np.sin(2*np.pi*i/12) + np.random.randn()*100
        rows.append([pod, d, base, base*0.7, base*0.4])
    return pd.DataFrame(rows, columns=["PodID", "ReportingMonth"] + MEASURES)

def to_month_end(series: pd.Series):
    idx = series.index.to_period("M").to_timestamp(how="end")
    s = pd.Series(series.values, index=idx)
    return s.groupby(s.index).sum(min_count=1)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.nanmean((y_true - y_pred)**2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.where(y_true==0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred)/denom))*100)

def build_supervised(series):
    s = series.copy()
    df = pd.DataFrame(index=s.index)
    df["y"] = s
    for L in range(1,13):
        df[f"lag{L}"] = s.shift(L)
    return df.dropna()

def prepare_xy(series, horizon):
    df = build_supervised(series)
    train = df.iloc[:-horizon]
    test  = df.iloc[-horizon:]
    return (
        train.drop(columns=["y"]).values,
        train["y"].values,
        test.drop(columns=["y"]).values,
        test["y"].values,
        test.index
    )

@st.cache_data(show_spinner=False)
def pod_monthly(df, pod):
    d = df[df["PodID"]==pod]
    if d.empty:
        return pd.DataFrame()
    return d.groupby("ReportingMonth")[MEASURES].sum(min_count=1).sort_index()

# -----------------------------------------------------------------------------
# MAIN START (DEFER BACKEND LOAD)
# -----------------------------------------------------------------------------
st.title("📈 PoD Forecast Playground")

# session state
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.note = ""
    st.session_state.loaded = False

if not st.session_state.loaded:
    st.info("Dataset is not loaded yet.")
    if st.button("Load backend dataset now", type="primary"):
        with st.spinner("Loading backend dataset..."):
            df, note = load_backend_df()
            st.session_state.df = df
            st.session_state.note = note
            st.session_state.loaded = True
        st.experimental_rerun()
    st.stop()

# Now dataset exists
df = st.session_state.df
st.caption(st.session_state.note)
st.caption(f"Rows: {len(df):,} | PoDs: {df['PodID'].nunique():,}")

# -----------------------------------------------------------------------------
# EXPLORE TAB
# -----------------------------------------------------------------------------
tabs = st.tabs(["🔎 Explore PoDs", "🧠 Model & Compare"])
tab1, tab2 = tabs

pods = sorted(df["PodID"].unique().tolist())
if "pod_idx" not in st.session_state:
    st.session_state.pod_idx = 0

with tab1:
    st.subheader("Browse PoDs")
    c1,c2,c3 = st.columns([3,1,1])
    selected = c1.selectbox("Select PoD", pods, index=st.session_state.pod_idx)
    if c2.button("Prev"):
        st.session_state.pod_idx = (st.session_state.pod_idx-1) % len(pods)
        st.experimental_rerun()
    if c3.button("Next"):
        st.session_state.pod_idx = (st.session_state.pod_idx+1) % len(pods)
        st.experimental_rerun()

    monthly = pod_monthly(df, selected)
    if monthly.empty:
        st.warning("No data for this PoD.")
    else:
        fig, ax = plt.subplots(figsize=(12,5))
        for m in MEASURES:
            ax.plot(monthly.index, monthly[m], label=m)
        ax.set_title(f"PoD {selected}")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# MODEL TAB
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Model Playground")

    c1,c2 = st.columns([3,2])
    pod_for_model = c1.selectbox("PoD", pods)
    measure = c2.radio("Measure", MEASURES, horizontal=True)

    series_raw = pod_monthly(df, pod_for_model)[measure]

    st.write(f"History: {len(series_raw)} months")

    # Preview only
    series_clean = to_month_end(series_raw).ffill().bfill()
    with st.expander("Preview series"):
        fig, ax = plt.subplots(figsize=(12,3))
        ax.plot(series_raw.index, series_raw.values, label="Raw")
        ax.plot(series_clean.index, series_clean.values, label="Cleaned")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Tabs
    t_xgb, t_rf, t_arima, t_sarima = st.tabs(
        ["🌲 XGB", "🌳 RF", "📐 ARIMA", "📏 SARIMA"]
    )

    # ---------------- XGB ----------------
    with t_xgb:
        c1,c2,c3 = st.columns(3)
        n_est = c1.slider("n_estimators", 50,1000,XGB_BASE["n_estimators"],50,key="xgb_n")
        depth = c1.slider("max_depth", 2,15,XGB_BASE["max_depth"],1,key="xgb_d")
        lr    = c2.slider("learning_rate",0.01,0.5,0.20,0.01,key="xgb_lr")
        subs  = c2.slider("subsample",0.5,1.0,0.8,0.05,key="xgb_s")
        colsm = c3.slider("colsample_bytree",0.5,1.0,0.8,0.05,key="xgb_cs")

        if st.button("Run XGB"):
            try:
                from xgboost import XGBRegressor
                X_train,y_train,X_test,y_test,dates = prepare_xy(series_clean,DEFAULT_TEST_HORIZON)
                model = XGBRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=lr,
                    subsample=subs,
                    colsample_bytree=colsm,
                    random_state=42,
                    tree_method="hist"
                )
                model.fit(X_train,y_train)
                pred = model.predict(X_test)

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(ax,dates,y_test,pred,"XGB Forecast",f"depth={depth}")
                st.pyplot(fig)
                st.metric("RMSE", f"{rmse(y_test,pred):.2f}")
                st.metric("MAPE", f"{mape(y_test,pred):.2f}%")
            except Exception as e:
                st.error(str(e))

    # ---------------- RF ----------------
    with t_rf:
        c1,c2,c3 = st.columns(3)
        n_est = c1.slider("n_estimators",50,1000,500,50,key="rf_n")
        depth = c1.slider("max_depth",2,20,6,1,key="rf_d")
        minsp = c2.slider("min_samples_split",2,20,2,1,key="rf_ms")
        minlf = c2.slider("min_samples_leaf",1,20,2,1,key="rf_ml")
        maxft = c3.slider("max_features",0.1,1.0,0.8,0.05,key="rf_mf")
        boot  = c3.checkbox("bootstrap",True,key="rf_boot")

        if st.button("Run RF"):
            try:
                X_train,y_train,X_test,y_test,dates = prepare_xy(series_clean,DEFAULT_TEST_HORIZON)
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
                model.fit(X_train,y_train)
                pred = model.predict(X_test)

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(ax,dates,y_test,pred,"RF Forecast",f"depth={depth}")
                st.pyplot(fig)
                st.metric("RMSE", f"{rmse(y_test,pred):.2f}")
                st.metric("MAPE", f"{mape(y_test,pred):.2f}%")
            except Exception as e:
                st.error(str(e))

    # ---------------- ARIMA ----------------
    with t_arima:
        import sys

        c1,c2,c3 = st.columns(3)
        p = c1.slider("p",0,10,1,1,key="arima_p")
        d = c2.slider("d",0,2,0,1,key="arima_d")
        q = c3.slider("q",0,10,1,1,key="arima_q")
        trend = st.selectbox("Trend",["none","c","t"],index=1,key="arima_trend")
        trend = None if trend=="none" else trend
        use_log = st.checkbox("log1p transform",False,key="arima_log")

        if st.button("Run ARIMA"):
            try:
                import statsmodels.api as sm
                y = series_clean.copy()
                if use_log:
                    y_mod = np.log1p(np.clip(y,0,None))
                else:
                    y_mod = y
                y_train = y_mod.iloc[:-DEFAULT_TEST_HORIZON]
                y_test  = y.iloc[-DEFAULT_TEST_HORIZON:]

                model = sm.tsa.SARIMAX(
                    y_train,
                    order=(p,d,q),
                    seasonal_order=SARIMA_BASE,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization="approximate_diffuse"
                )
                res = model.fit(disp=False,maxiter=500)
                fc = res.get_forecast(DEFAULT_TEST_HORIZON).predicted_mean
                pred = np.expm1(fc) if use_log else fc

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(ax,y_test.index,y_test.values,pred.values,
                            "ARIMA Forecast",f"{p,d,q}")
                st.pyplot(fig)
                st.metric("RMSE",f"{rmse(y_test,pred):.2f}")
                st.metric("MAPE",f"{mape(y_test,pred):.2f}%")
            except Exception as e:
                st.error(str(e))

    # ---------------- SARIMA ----------------
    with t_sarima:
        c1,c2,c3 = st.columns(3)
        p_s = c1.slider("p",0,10,1,1,key="sarima_p")
        d_s = c2.slider("d",0,2,0,1,key="sarima_d")
        q_s = c3.slider("q",0,10,1,1,key="sarima_q")

        c4,c5,c6,c7 = st.columns(4)
        P_s = c4.slider("P",0,10,1,1,key="sarima_P")
        D_s = c5.slider("D",0,2,1,1,key="sarima_D")
        Q_s = c6.slider("Q",0,10,0,1,key="sarima_Q")
        s_s = c7.slider("s",6,24,12,1,key="sarima_s")

        trend_s = st.selectbox("Trend",["none","c","t"],index=1,key="sarima_trend")
        trend_s = None if trend_s=="none" else trend_s
        use_log_s = st.checkbox("log1p transform",False,key="sarima_log")

        if st.button("Run SARIMA"):
            try:
                import statsmodels.api as sm
                y = series_clean.copy()
                if use_log_s:
                    y_mod = np.log1p(np.clip(y,0,None))
                else:
                    y_mod = y

                y_train = y_mod.iloc[:-DEFAULT_TEST_HORIZON]
                y_test  = y.iloc[-DEFAULT_TEST_HORIZON:]

                model = sm.tsa.SARIMAX(
                    y_train,
                    order=(p_s,d_s,q_s),
                    seasonal_order=(P_s,D_s,Q_s,s_s),
                    trend=trend_s,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    initialization="approximate_diffuse"
                )
                res = model.fit(disp=False,maxiter=500)
                fc = res.get_forecast(DEFAULT_TEST_HORIZON).predicted_mean
                pred = np.expm1(fc) if use_log_s else fc

                fig, ax = plt.subplots(figsize=(12,4))
                plot_series(ax,y_test.index,y_test.values,pred.values,
                            "SARIMA Forecast",
                            f"{p_s,d_s,q_s} x {P_s,D_s,Q_s,s_s}")
                st.pyplot(fig)
                st.metric("RMSE",f"{rmse(y_test,pred):.2f}")
                st.metric("MAPE",f"{mape(y_test,pred):.2f}%")
            except Exception as e:
                st.error(str(e))