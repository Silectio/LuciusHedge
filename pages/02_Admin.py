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
    upsert_investor,
    add_or_update_stake,
    compute_allocations,
    replace_allocations,
    model_return_stats,
    pending_gains_by_investor,
    upsert_payouts,
)
from lib.numerai import get_napi, get_models, fetch_v2_rounds, extract_season_payouts

st.set_page_config(page_title="Numerai — Admin", layout="wide")
st.title("Panneau Admin")

ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")

# Auth simple
pw = st.sidebar.text_input("Mot de passe admin", type="password")
if not ADMIN_PASSWORD or pw != ADMIN_PASSWORD:
    st.error("Accès admin refusé")
    st.stop()

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

# Modèles (toujours tous)
models_map = get_models(napi) if napi_ok else {}
name_by_id = {v: k for k, v in models_map.items()} if models_map else {}

T_inv, T_stk, T_payout_admin, T_stats = st.tabs(
    ["👤 Investisseurs", "📌 Stakes", "💸 Payouts (admin)", "📈 Stats modèles"]
)

with T_inv:
    st.subheader("Investisseurs")
    if conn_ok:
        with SessionLocal() as s:
            inv_df = list_investors_df(s)
    else:
        inv_df = pd.DataFrame()
    st.dataframe(inv_df if not inv_df.empty else pd.DataFrame())

    st.markdown("---")
    st.subheader("Ajouter / Mettre à jour un investisseur")
    with st.form("add_inv"):
        inv_id = st.text_input("ID (laisser vide pour créer)")
        name = st.text_input("Nom")
        email = st.text_input("Email")
        active = st.checkbox("Actif", value=True)
        notes = st.text_area("Notes", "")
        submit = st.form_submit_button("Enregistrer")
        if submit and conn_ok and name:
            with SessionLocal() as s:
                upsert_investor(
                    s,
                    investor_id=inv_id or name.lower().replace(" ", "_"),
                    name=name,
                    email=email,
                    active=active,
                    notes=notes,
                )
                s.commit()
            st.success("Investisseur enregistré")

with T_stk:
    st.subheader("Stakes")
    if conn_ok:
        with SessionLocal() as s:
            stks_df = list_stakes_df(s)
    else:
        stks_df = pd.DataFrame()
    show = stks_df.copy()
    if name_by_id and not show.empty:
        show["model"] = show["model_id"].map(name_by_id).fillna("Global")
    st.dataframe(
        show.sort_values(["model", "start_date"])
        if not show.empty and "model" in show.columns
        else show
    )

    st.markdown("---")
    st.subheader("Ajouter / Mettre à jour un stake")
    with st.form("add_stk"):
        stake_id = st.text_input("Stake ID (laisser vide pour créer)")
        investor_id = st.text_input("Investor ID")
        mdl = (
            st.selectbox("Modèle", list(models_map.keys()) if models_map else ["—"])
            if models_map
            else ""
        )
        model_id = models_map.get(mdl) if models_map else None
        amount = st.number_input("Montant", min_value=0.0, value=0.0, step=0.1)
        start = st.date_input("Début", value=date.today())
        end_on = st.checkbox("Avec fin", value=False)
        end_date = st.date_input("Fin", value=date.today()) if end_on else None
        notes = st.text_input("Notes", "")
        submit = st.form_submit_button("Enregistrer")
        if submit and conn_ok and investor_id and model_id:
            with SessionLocal() as s:
                add_or_update_stake(
                    s,
                    stake_id=stake_id or f"stk_{investor_id}_{model_id}_{start}",
                    investor_id=investor_id,
                    model_id=model_id,
                    amount=amount,
                    start_date=start,
                    end_date=end_date,
                    notes=notes,
                )
                s.commit()
            st.success("Stake enregistré")

with T_payout_admin:
    st.subheader("Payouts (admin) — saison basés sur Season Score × PayoutFactor")
    col1, col2 = st.columns(2)
    with col1:
        sync = st.button("Synchroniser tous les modèles")
    with col2:
        recalc = st.button("Recalculer allocations (tous modèles)")

    if sync and conn_ok and napi_ok and models_map:
        total_ins, total_skip = 0, 0
        with SessionLocal() as s:
            for name, mid in models_map.items():
                rounds = fetch_v2_rounds(napi, mid)
                df = extract_season_payouts(rounds, model_id=mid)
                ins, sk = upsert_payouts(s, df)
                total_ins += ins
                total_skip += sk
            s.commit()
        st.success(f"Payouts insérés: {total_ins}, ignorés: {total_skip}")

    if recalc and conn_ok and models_map:
        with SessionLocal() as s:
            for name, mid in models_map.items():
                alloc_df = compute_allocations(s, model_id=mid, payout_metric="season")
                replace_allocations(s, alloc_df)
            s.commit()
        st.success("Allocations recalculées pour tous les modèles")

    # Vue
    if conn_ok:
        with SessionLocal() as s:
            pay_df = list_payouts_df(s, None)
    else:
        pay_df = pd.DataFrame()
    show = pay_df.copy()
    if name_by_id and not show.empty:
        show["model"] = show["model_id"].map(name_by_id)
    st.dataframe(
        show.sort_values(["model", "roundNumber"])
        if not show.empty and "model" in show.columns
        else show
    )

with T_stats:
    st.subheader("Stats géométriques par modèle (season)")
    if conn_ok:
        with SessionLocal() as s:
            stats_df = model_return_stats(s)
    else:
        stats_df = pd.DataFrame()
    if not stats_df.empty and name_by_id:
        stats_df["model"] = stats_df["model_id"].map(name_by_id)
        stats_df = stats_df[
            ["model", "geo_daily", "geo_monthly", "geo_annual"]
        ].sort_values("model")
    st.dataframe(stats_df)

    st.markdown("---")
    st.subheader("Gains en attente par investisseur (non résolus)")
    if conn_ok:
        with SessionLocal() as s:
            pend_df = pending_gains_by_investor(s)
            inv_df = list_investors_df(s)
    else:
        pend_df = pd.DataFrame()
        inv_df = pd.DataFrame()
    if not pend_df.empty and not inv_df.empty:
        pend_df = pend_df.merge(
            inv_df[["investor_id", "name"]], on="investor_id", how="left"
        )
    st.dataframe(pend_df.rename(columns={"pending_gain": "en_attente"}))
