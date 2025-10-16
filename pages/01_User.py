import os
from datetime import date

import pandas as pd
import streamlit as st

from lib.db import (
    get_engine_and_session,
    list_investors_df,
    list_stakes_df,
    list_payouts_df,
    list_allocations_df,
    model_return_stats,
    pending_gains_by_investor,
)
from lib.numerai import get_napi, get_models

st.set_page_config(page_title="Numerai — Utilisateur", layout="wide")
st.title("Vue Utilisateur")

# Vérifs
conn_ok = True
try:
    engine, SessionLocal = get_engine_and_session()
    st.sidebar.success("Base de données prête")
except Exception as e:
    conn_ok = False
    st.sidebar.error(f"DB KO: {e}")

napi_ok = True
try:
    napi = get_napi()
    st.sidebar.success("Numerai API prête")
except Exception as e:
    napi_ok = False
    st.sidebar.error(f"Numerai API KO: {e}")

# Modèles: tous
models_map = get_models(napi) if napi_ok else {}
name_by_id = {v: k for k, v in models_map.items()} if models_map else {}

# Data
if conn_ok:
    with SessionLocal() as s:
        inv_df = list_investors_df(s)
        stks_df = list_stakes_df(s)
        pay_df = list_payouts_df(s, None)
        allc_df = list_allocations_df(s, None)
        stats_df = model_return_stats(s)
        pend_df = pending_gains_by_investor(s)
else:
    inv_df = pd.DataFrame()
    stks_df = pd.DataFrame()
    pay_df = pd.DataFrame()
    allc_df = pd.DataFrame()
    stats_df = pd.DataFrame()
    pend_df = pd.DataFrame()

# UI
T_overview, T_payouts, T_alloc, T_stats = st.tabs(
    ["📊 Vue d'ensemble", "💸 Payouts (season)", "🧮 Allocations", "📈 Stats"]
)

with T_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Investisseurs", 0 if inv_df.empty else inv_df.shape[0])
    total_stake = (
        0.0
        if stks_df.empty
        else float(
            pd.to_numeric(
                stks_df.get("amount", pd.Series([], dtype=float)), errors="coerce"
            )
            .fillna(0)
            .sum()
        )
    )
    total_payouts = (
        0.0
        if pay_df.empty
        else float(
            pd.to_numeric(
                pay_df.get("payout_value", pd.Series([], dtype=float)), errors="coerce"
            )
            .fillna(0)
            .sum()
        )
    )
    total_alloc = (
        0.0
        if allc_df.empty
        else float(
            pd.to_numeric(
                allc_df.get("share_value", pd.Series([], dtype=float)), errors="coerce"
            )
            .fillna(0)
            .sum()
        )
    )
    col2.metric("Stake cumulé", f"{total_stake:,.2f}")
    col3.metric("Payouts cumulés", f"{total_payouts:,.4f}")
    col4.metric("Allocations cumulées", f"{total_alloc:,.4f}")

with T_payouts:
    st.subheader("Payouts (calculés) — Season score × Payout factor")
    if pay_df.empty:
        st.info("Aucun payout. Synchronisez côté Admin.")
    else:
        show = pay_df.copy()
        if name_by_id:
            show["model"] = show["model_id"].map(name_by_id)
        st.dataframe(
            show.sort_values(["model", "roundNumber"])
            if "model" in show.columns
            else show
        )

with T_alloc:
    st.subheader("Allocations")
    if allc_df.empty:
        st.info("Aucune allocation (recalcul via Admin)")
    else:
        show = allc_df.copy()
        if name_by_id:
            show["model"] = show["model_id"].map(name_by_id)
        st.dataframe(
            show.sort_values(["model", "roundNumber", "investor_id"])
            if "model" in show.columns
            else show
        )
        # Gains en attente par investisseur
        st.markdown("### Gains en attente (rounds non résolus) — peuvent changer")
        if not pend_df.empty and not inv_df.empty:
            pend_show = pend_df.merge(
                inv_df[["investor_id", "name"]], on="investor_id", how="left"
            )
            st.dataframe(pend_show.rename(columns={"pending_gain": "en_attente"}))
        else:
            st.info("Aucun gain en attente calculable")

with T_stats:
    st.subheader("Stats géométriques par modèle")
    if stats_df.empty:
        st.info("Aucune stat disponible")
    else:
        show = stats_df.copy()
        if name_by_id:
            show["model"] = show["model_id"].map(name_by_id)
            show = show[
                ["model", "geo_daily", "geo_monthly", "geo_annual"]
            ].sort_values("model")
        st.dataframe(show)
