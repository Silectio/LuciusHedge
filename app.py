import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Optional

from db import (
    ensure_investment_tables,
    get_investors,
    add_investor,
    update_investor,
    delete_investor,
    get_stakes,
    add_stake,
    update_stake,
    delete_stake,
)
from numerapi.numerapi import NumerAPI
from numerai import build_numerai_wide_df, DEFAULT_METRICS, gain_timeseries
from numerai import build_numerai_wide_df, DEFAULT_METRICS, gain_timeseries


@st.cache_data(show_spinner=False)
def _get_models() -> list[str]:
    try:
        # Essaye secrets Streamlit
        pub = st.secrets.get("NUMERAI_PUBLIC_ID", None)
        sec = st.secrets.get("NUMERAI_SECRET_KEY", None)
    except Exception:
        pub = None
        sec = None
    import os

    pub = pub or os.environ.get("NUMERAI_PUBLIC_ID")
    sec = sec or os.environ.get("NUMERAI_SECRET_KEY")
    if not pub or not sec:
        return []
    try:
        api = NumerAPI(public_id=pub, secret_key=sec)
        models_map = api.get_models()  # {name: id}
        return sorted(list(models_map.keys()))
    except Exception:
        return []


def ui_investors():
    st.header("Investisseurs")
    ensure_investment_tables()

    # Liste des investisseurs
    investors = get_investors(order_by=["id"])
    inv_map = {
        f"#{i['id']} - {i.get('name') or i.get('email')}": i["id"] for i in investors
    }

    st.subheader("Ajouter un investisseur")
    with st.form("add_investor_form", clear_on_submit=True):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Ajouter")
        if submitted:
            if not name and not email:
                st.warning("Renseignez au moins un nom ou un email.")
            else:
                row = add_investor(name=name, email=email or None, notes=notes or None)
                st.success(f"Investisseur créé: {row}")
                st.rerun()

    st.subheader("Modifier / Supprimer un investisseur")
    if investors:
        label = st.selectbox("Sélectionner", list(inv_map.keys()))
        inv_id = inv_map[label]
        inv = next((i for i in investors if i["id"] == inv_id), None)
        if inv:
            with st.form("edit_investor_form"):
                name = st.text_input("Nom", value=inv.get("name") or "")
                email = st.text_input("Email", value=inv.get("email") or "")
                notes = st.text_area("Notes", value=inv.get("notes") or "")
                c1, c2 = st.columns(2)
                with c1:
                    if st.form_submit_button("Enregistrer"):
                        update_investor(
                            inv_id, {"name": name, "email": email, "notes": notes}
                        )
                        st.success("Investisseur mis à jour.")
                        st.rerun()
                with c2:
                    if st.form_submit_button("Supprimer", type="primary"):
                        delete_investor(inv_id)
                        st.success("Investisseur supprimé (et stakes associés).")
                        st.rerun()
    else:
        st.info("Aucun investisseur pour le moment.")


def ui_stakes():
    st.header("Stakes")
    ensure_investment_tables()
    investors = get_investors(order_by=["id"])
    if not investors:
        st.info("Créez d'abord un investisseur.")
        return
    inv_map = {
        f"#{i['id']} - {i.get('name') or i.get('email')}": i["id"] for i in investors
    }

    st.subheader("Ajouter un stake")
    with st.form("add_stake_form", clear_on_submit=True):
        label = st.selectbox("Investisseur", list(inv_map.keys()))
        inv_id = inv_map[label]
        amount = st.number_input("Montant", min_value=0.0, value=0.0, step=0.0001)
        start_date = st.date_input("Date de début", value=datetime.utcnow().date())
        models = _get_models()
        model_name = st.selectbox("Modèle", options=(models if models else ["(aucun)"]))
        has_end = st.checkbox("Ajouter une date de fin ?", value=False)
        end_date: Optional[date] = None
        if has_end:
            end_date = st.date_input("Date de fin", value=datetime.utcnow().date())
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Ajouter")
        if submitted:
            row = add_stake(
                investor_id=inv_id,
                amount=amount,
                start_date=start_date,
                end_date=end_date or None,
                model_name=(
                    None if not models or model_name == "(aucun)" else model_name
                ),
                notes=notes or None,
            )
            st.success(f"Stake créé: {row}")
            st.rerun()

    st.subheader("Liste / Modifier / Supprimer")
    tab_all, tab_by_inv, tab_by_model = st.tabs(
        ["Tous", "Par investisseur", "Par modèle"]
    )
    with tab_all:
        stakes = get_stakes()
        st.dataframe(stakes, width="stretch")
    with tab_by_inv:
        label = st.selectbox(
            "Investisseur filtre", list(inv_map.keys()), key="stake_filter"
        )
        inv_id = inv_map[label]
        stakes = get_stakes(investor_id=inv_id)
        st.dataframe(stakes, width="stretch")
    with tab_by_model:
        models = _get_models()
        if models:
            m = st.selectbox("Modèle", options=models, key="stake_model_filter")
            st.dataframe(get_stakes(model_name=m), width="stretch")
        else:
            st.info("Aucun modèle disponible (vérifiez vos identifiants Numerai).")

    st.subheader("Modifier / Supprimer un stake par ID")
    stake_id = st.number_input("Stake ID", min_value=0, step=1)
    c1, c2 = st.columns(2)
    with c1:
        with st.form("edit_stake_form"):
            amount = st.number_input("Montant (nouveau)", min_value=0.0, step=0.0001)
            # Dates toujours visibles pour éviter l'effet "rien n'apparaît" dans un form
            set_start = st.checkbox("Modifier date début")
            start_date = st.date_input(
                "Date début (nouvelle)", value=datetime.utcnow().date()
            )
            set_end = st.checkbox("Modifier date fin")
            end_date = st.date_input(
                "Date fin (nouvelle)", value=datetime.utcnow().date()
            )
            clear_end = st.checkbox("Supprimer la date de fin (mettre à vide)")
            # Changement de modèle rendu AVANT soumission
            models = _get_models()
            new_model = None
            if models:
                new_model = st.selectbox(
                    "Changer de modèle", options=["(inchangé)"] + models, index=0
                )
            notes = st.text_area("Notes (nouvelle)")
            if st.form_submit_button("Enregistrer"):
                vals = {}
                if amount and amount > 0:
                    vals["amount"] = amount
                if set_start:
                    vals["start_date"] = start_date
                if clear_end:
                    vals["end_date"] = None
                elif set_end:
                    vals["end_date"] = end_date
                if new_model and new_model != "(inchangé)":
                    vals["model_name"] = new_model
                if notes:
                    vals["notes"] = notes
                if not vals:
                    st.warning("Rien à mettre à jour.")
                else:
                    update_stake(int(stake_id), vals)
                    st.success("Stake mis à jour.")
                    st.rerun()
    with c2:
        if st.button("Supprimer stake", type="primary"):
            if stake_id > 0:
                delete_stake(int(stake_id))
                st.success("Stake supprimé.")
                st.rerun()

                # Ajout d'une vérification de mot de passe dans la sidebar


@st.cache_data(show_spinner=True)
def _fetch_wide(metrics: tuple[str, ...], add_payout: bool) -> "pd.DataFrame":
    # build_numerai_wide_df utilise les credentials via secrets/env si non fournis
    df = build_numerai_wide_df(metrics=list(metrics), add_payout=add_payout)
    # Retourner une copie pour éviter des effets de bord éventuels
    try:
        return df.copy(deep=True)
    except Exception:
        return df


def ui_models_performance():
    st.header("Performance des modèles")
    # Streamlit rerun automatiquement à chaque changement de widget (hors forms)
    resolved_only = st.checkbox(
        "Uniquement rounds résolus",
        value=True,
        key="resolved_only_checkbox",
    )
    with st.spinner("Chargement des données Numerai..."):
        df_wide = _fetch_wide(tuple(DEFAULT_METRICS), True)
    if df_wide is None or df_wide.empty:
        st.info("Aucune donnée Numerai disponible. Vérifiez les identifiants.")
        return
    df = df_wide.copy()
    # Filtre selon le toggle: résolu si roundResolveTime < aujourd'hui (comparaison à la date du jour)
    if "roundResolveTime" in df.columns:
        rt = pd.to_datetime(df["roundResolveTime"], errors="coerce")
        mask_resolved = rt.notna() & (rt.dt.date < date.today())
        if resolved_only:
            df = df[mask_resolved]

    # Recalculer payout après filtrage pour garantir un cumul exact selon le choix
    if {"roundPayoutFactor", "season_score"}.issubset(df.columns):
        df["payout"] = df["roundPayoutFactor"] * df["season_score"]
    else:
        st.warning(
            "Colonnes nécessaires absentes (roundPayoutFactor et/ou season_score); impossible de calculer le payout."
        )
        st.dataframe(df.head(50), width="stretch")
        return
    # Agrégations par modèle: total payout, nombre de rounds, moyenne et médiane de payout
    if "roundNumber" in df.columns:
        perf = (
            df.groupby("modelName", as_index=False)
            .agg(
                payout=("payout", "sum"),
                rounds=("roundNumber", "nunique"),
                payout_avg=("payout", "mean"),
                payout_median=("payout", "median"),
            )
            .sort_values(by="payout", ascending=False)
        )
    else:
        perf = (
            df.groupby("modelName", as_index=False)
            .agg(
                payout=("payout", "sum"),
                rounds=("payout", "count"),
                payout_avg=("payout", "mean"),
                payout_median=("payout", "median"),
            )
            .sort_values(by="payout", ascending=False)
        )
    st.subheader("Payout cumulé par modèle")
    st.dataframe(perf, width="stretch")
    # Légende globale: nombre total de rounds agrégés
    try:
        total_rows = len(df)
        total_unique_rounds = (
            int(df["roundNumber"].nunique())
            if "roundNumber" in df.columns
            else total_rows
        )
        st.caption(
            f"Rounds agrégés: {total_rows} lignes • {total_unique_rounds} rounds uniques"
        )
    except Exception:
        pass
    try:
        st.bar_chart(perf.set_index("modelName")["payout"])
    except Exception:
        pass


# ====================== Vue: Gain (évolution des stakes) ======================


def _build_stake_timeseries(stakes_df: list[dict]) -> pd.DataFrame:
    """Construit une série temporelle du montant cumulé des stakes.
    Approche: +amount à start_date, -amount au lendemain de end_date (si présent).
    Puis cumul dans le temps pour obtenir le montant courant quotidien.
    """
    if not stakes_df:
        return pd.DataFrame(columns=["date", "stake"])
    df = pd.DataFrame(stakes_df)
    # Normaliser dates
    for c in ("start_date", "end_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    # Événements
    events = []
    for _, row in df.iterrows():
        amt = float(row.get("amount") or 0.0)
        sd = row.get("start_date")
        ed = row.get("end_date")
        if pd.notna(sd):
            events.append({"date": sd, "delta": amt})
        if pd.notna(ed):
            # le lendemain -> retrait
            events.append(
                {"date": pd.Timestamp(ed) + pd.Timedelta(days=1), "delta": -amt}
            )
    if not events:
        return pd.DataFrame(columns=["date", "stake"])
    ev = pd.DataFrame(events)
    ev["date"] = pd.to_datetime(ev["date"]).dt.date
    # Agréger par jour
    ev = (
        ev.groupby("date", as_index=False)
        .agg(delta=("delta", "sum"))
        .sort_values(by="date")
    )
    # Étendre calendrier de la première date à aujourd'hui
    cal = pd.DataFrame(
        {
            "date": pd.date_range(
                start=ev["date"].min(), end=pd.Timestamp.today().date(), freq="D"
            ).date
        }
    )
    ts = cal.merge(ev, on="date", how="left").fillna({"delta": 0.0})
    ts["stake"] = ts["delta"].cumsum()
    return ts[["date", "stake"]]


def ui_gain():
    st.header("Gain / Évolution des stakes")
    ensure_investment_tables()
    investors = get_investors(order_by=["id"]) or []
    if not investors:
        st.info("Aucun investisseur. Ajoutez-en un dans l'onglet Investisseurs.")
        return
    inv_map = {
        f"#{i['id']} - {i.get('name') or i.get('email')}": i["id"] for i in investors
    }
    label = st.sidebar.selectbox(
        "Investisseur", list(inv_map.keys()), key="gain_investor_select"
    )
    inv_id = inv_map[label]
    # Filtrage optionnel par modèle
    models = _get_models()
    model_filter = None
    if models:
        model_filter = st.sidebar.selectbox(
            "Modèle (optionnel)", ["(tous)"] + models, index=0, key="gain_model_select"
        )
        if model_filter == "(tous)":
            model_filter = None
    # Choix de prise en compte des rounds résolus (pour le graphe de gains)
    resolved_only = st.sidebar.checkbox(
        "Uniquement rounds résolus", value=True, key="gain_resolved_only_checkbox"
    )

    # Charger le wide Numerai et préparer les vues all/settled, puis appliquer filtre modèle
    with st.spinner("Chargement des données Numerai..."):
        df_wide = _fetch_wide(tuple(DEFAULT_METRICS), True)
    if df_wide is None or df_wide.empty or "payout" not in df_wide.columns:
        st.info(
            "Données Numerai indisponibles (payout manquant). Vérifiez les identifiants."
        )
        return
    # Vue all
    df_w_all = df_wide.copy()
    if model_filter and "modelName" in df_w_all.columns:
        df_w_all = df_w_all[df_w_all["modelName"] == model_filter]
    # Vue settled (resolved)
    if "roundResolveTime" in df_w_all.columns:
        rt_all = pd.to_datetime(df_w_all["roundResolveTime"], errors="coerce")
        mask_resolved = rt_all.notna() & (rt_all.dt.date < date.today())
        df_w_settled = df_w_all[mask_resolved]
    else:
        df_w_settled = df_w_all.iloc[0:0]
    # Vue utilisée pour tracer les gains
    df_w = df_w_settled if resolved_only else df_w_all

    # Récupérer stakes pour cet investisseur (et modèle éventuellement)
    stakes = get_stakes(investor_id=inv_id, model_name=model_filter)
    if not stakes:
        st.info("Aucune stake pour cet investisseur selon le filtre.")
        return

    # Afficher systématiquement le graphe des stakes (montant)
    ts_all = _build_stake_timeseries(stakes)
    if ts_all is not None and not ts_all.empty:
        st.subheader("Montant stakes dans le temps")
        st.line_chart(ts_all.set_index("date")["stake"])
        try:
            st.caption(
                f"Période: {ts_all['date'].min()} → {ts_all['date'].max()} • {len(ts_all)} jours (1 point/jour)"
            )
        except Exception:
            pass

    # Cas 1: modèle filtré -> gains directs
    if model_filter:
        ts = _build_stake_timeseries(stakes)
        if ts is None or ts.empty:
            st.info("Aucune stake pour cet investisseur.")
            return
        # Subset minimal pour la fonction gain_timeseries
        cols = [c for c in ["roundDate", "payout"] if c in df_w.columns]
        if len(cols) < 2:
            st.info("Colonnes nécessaires absentes pour le calcul des gains.")
            return
        gains = gain_timeseries(df_w[cols], ts)
        if gains is None or gains.empty:
            st.info("Aucun gain calculable pour cette période.")
            return
        st.subheader("Gain quotidien (payout × stake)")
        st.line_chart(gains.set_index("date")["gain"])
        try:
            st.caption(
                f"Période: {gains['date'].min()} → {gains['date'].max()} • {len(gains)} jours (1 point/jour)"
            )
        except Exception:
            pass
        # KPIs: stake actuel, gains totaux settled et tous
        cols_set = [c for c in ["roundDate", "payout"] if c in df_w_settled.columns]
        gains_set = (
            gain_timeseries(df_w_settled[cols_set], ts)
            if len(cols_set) == 2
            else pd.DataFrame(columns=["date", "gain"])
        )
        cols_all = [c for c in ["roundDate", "payout"] if c in df_w_all.columns]
        gains_all = (
            gain_timeseries(df_w_all[cols_all], ts)
            if len(cols_all) == 2
            else pd.DataFrame(columns=["date", "gain"])
        )
        total_settled = float(gains_set["gain"].sum()) if not gains_set.empty else 0.0
        total_all = float(gains_all["gain"].sum()) if not gains_all.empty else 0.0
        stake_now = (
            float(ts_all["stake"].iloc[-1])
            if ts_all is not None and not ts_all.empty
            else 0.0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Stake actuel", f"{stake_now:,.4f}")
        c2.metric("Gain total (settled)", f"{total_settled:,.4f}")
        c3.metric("Gain total (tous)", f"{total_all:,.4f}")
    else:
        # Cas 2: tous modèles -> calcul par modèle puis somme jour par jour
        stakes_df = pd.DataFrame(stakes)
        if "model_name" not in stakes_df.columns:
            st.info("Aucun modèle associé aux stakes de cet investisseur.")
            return
        stakes_df = stakes_df[stakes_df["model_name"].notna()].copy()
        if stakes_df.empty:
            st.info("Aucun stake avec modèle défini pour cet investisseur.")
            return
        gains_list = []
        for m in sorted(stakes_df["model_name"].dropna().unique()):
            stakes_m = stakes_df[stakes_df["model_name"] == m].to_dict("records")
            ts_m = _build_stake_timeseries(stakes_m)
            if ts_m is None or ts_m.empty:
                continue
            df_m = (
                df_w[df_w["modelName"] == m]
                if "modelName" in df_w.columns
                else df_w.iloc[0:0]
            )
            cols = [c for c in ["roundDate", "payout"] if c in df_m.columns]
            if df_m.empty or len(cols) < 2:
                continue
            g_m = gain_timeseries(df_m[cols], ts_m)
            if g_m is not None and not g_m.empty:
                gains_list.append(g_m)
        if not gains_list:
            st.info(
                "Aucun gain calculable (pensez à vérifier les dates et les payouts)."
            )
            return
        gains_all = pd.concat(gains_list, ignore_index=True)
        gains_sum = (
            gains_all.groupby("date", as_index=False)
            .agg(gain=("gain", "sum"))
            .sort_values(by="date")
        )
        # Étendre calendrier complet jusqu'à aujourd'hui
        full_cal = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=gains_sum["date"].min(),
                    end=pd.Timestamp.today().date(),
                    freq="D",
                ).date
            }
        )
        gains_sum = full_cal.merge(gains_sum, on="date", how="left").fillna(
            {"gain": 0.0}
        )
        st.subheader("Gain quotidien agrégé (tous modèles)")
        st.line_chart(gains_sum.set_index("date")["gain"])
        try:
            st.caption(
                f"Période: {gains_sum['date'].min()} → {gains_sum['date'].max()} • {len(gains_sum)} jours (1 point/jour)"
            )
        except Exception:
            pass
        # KPIs: calculer les totaux settled et tous (somme des modèles)
        total_settled = 0.0
        total_all = 0.0
        for m in sorted(stakes_df["model_name"].dropna().unique()):
            stakes_m = stakes_df[stakes_df["model_name"] == m].to_dict("records")
            ts_m = _build_stake_timeseries(stakes_m)
            if ts_m is None or ts_m.empty:
                continue
            df_m_set = (
                df_w_settled[df_w_settled["modelName"] == m]
                if "modelName" in df_w_settled.columns
                else df_w_settled.iloc[0:0]
            )
            cols_set = [c for c in ["roundDate", "payout"] if c in df_m_set.columns]
            if not df_m_set.empty and len(cols_set) == 2:
                g_set = gain_timeseries(df_m_set[cols_set], ts_m)
                if g_set is not None and not g_set.empty:
                    total_settled += float(g_set["gain"].sum())
            df_m_all = (
                df_w_all[df_w_all["modelName"] == m]
                if "modelName" in df_w_all.columns
                else df_w_all.iloc[0:0]
            )
            cols_all = [c for c in ["roundDate", "payout"] if c in df_m_all.columns]
            if not df_m_all.empty and len(cols_all) == 2:
                g_all = gain_timeseries(df_m_all[cols_all], ts_m)
                if g_all is not None and not g_all.empty:
                    total_all += float(g_all["gain"].sum())
        stake_now = (
            float(ts_all["stake"].iloc[-1])
            if ts_all is not None and not ts_all.empty
            else 0.0
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Stake actuel", f"{stake_now:,.4f}")
        c2.metric("Gain total (settled)", f"{total_settled:,.4f}")
        c3.metric("Gain total (tous)", f"{total_all:,.4f}")


def main():
    st.set_page_config(page_title="Gestion Investors & Stakes", layout="wide")
    st.title("Gestion de la base: Investisseurs et Stakes")

    # Auth facultative: protège uniquement les sections Admin (Investisseurs/Stakes)
    password = st.sidebar.text_input("Mot de passe (Admin)", type="password")
    try:
        correct_password = st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        correct_password = None
    auth_ok = bool(password) and (
        correct_password is None or password == correct_password
    )
    if not auth_ok:
        pages_list = ["Performance modèles", "Gain"]
    else:
        pages_list = ["Performance modèles", "Gain", "Investisseurs", "Stakes"]

    page = st.sidebar.radio(
        "Navigation", options=pages_list, index=0, key="main_nav_radio"
    )
    if page == "Performance modèles":
        ui_models_performance()
        return
    if page == "Gain":
        ui_gain()
        return
    if not auth_ok:
        st.warning(
            "Accès réservé: entrez le mot de passe Admin pour gérer investisseurs et stakes."
        )
        return
    if page == "Investisseurs":
        ui_investors()
    elif page == "Stakes":
        ui_stakes()


# ====================== Vue: Performance modèles ======================


if __name__ == "__main__":
    main()
