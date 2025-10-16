from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from numerapi.numerapi import NumerAPI


QUERY_V2 = """
query($modelId: String!) {
  v2RoundModelPerformances(modelId: $modelId) {
    atRisk
    churnThreshold
    corrMultiplier
    mmcMultiplier
    prevWeekChurnMax
    prevWeekTurnoverMax
    roundCloseStakingTime
    roundDataDatestamp
    roundNumber
    roundPayoutFactor
    roundResolveTime
    roundResolved
    roundScoreTime
    roundTarget
    tcMultiplier
    turnoverThreshold
    submissionScores {
      date
      day
      displayName
      payoutPending
      payoutSettled
      percentile
      resolveDate
      resolved
      value
    }
  }
}
"""


DEFAULT_METRICS: List[str] = [
    "corj60",
    "corr60",
    "corr_w_meta_model",
    "cort20",
    "fnc_v3",
    "mcwnm",
    "mmc",
    "mmc60",
    "season_score",
    "v2_corr20",
    "apcwnm",
    "bmc",
    "canon_bmc",
    "canon_corj60",
    "canon_corr",
    "canon_corr60",
    "canon_cort20",
    "canon_fnc_v3",
    "canon_mmc",
    "canon_mmc60",
]


def _get_credentials(
    public_id: Optional[str] = None, secret_key: Optional[str] = None
) -> tuple[str, str]:
    if public_id and secret_key:
        return public_id, secret_key
    try:
        import streamlit as st  # type: ignore

        pid = public_id or st.secrets.get("NUMERAI_PUBLIC_ID")
        sk = secret_key or st.secrets.get("NUMERAI_SECRET_KEY")
        if pid and sk:
            return str(pid), str(sk)
    except Exception:
        pass
    pid = public_id or os.environ.get("NUMERAI_PUBLIC_ID")
    sk = secret_key or os.environ.get("NUMERAI_SECRET_KEY")
    if not pid or not sk:
        raise RuntimeError(
            "Identifiants Numerai manquants. Fournir public_id/secret_key ou configurer secrets/env."
        )
    return pid, sk


def _dt_naive(x: Any) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    t = pd.to_datetime(x, errors="coerce")
    if pd.isna(t):
        return None
    try:
        if getattr(t, "tzinfo", None) is not None:
            return t.tz_localize(None)
    except Exception:
        pass
    return t


def build_numerai_wide_df(
    public_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    *,
    metrics: Optional[Sequence[str]] = None,
    add_payout: bool = True,
) -> pd.DataFrame:
    """Construit et retourne le DataFrame wide Numerai pour tous les modèles.
    - Résout les credentials via args > secrets Streamlit > env
    - Récupère la liste des modèles
    - Télécharge les rounds pour chaque modèle
    - Construit un DF long puis un DF wide (1 ligne par modelName & roundNumber)
      en n'utilisant que les dates du round
    - Ajoute la colonne payout si possible (roundPayoutFactor * season_score)
    """
    pid, sk = _get_credentials(public_id, secret_key)
    api = NumerAPI(public_id=pid, secret_key=sk)
    models_map: Dict[str, str] = api.get_models()

    # Fetch rounds et construire long DF
    all_rows: list[dict] = []
    for model_name, model_id in models_map.items():
        res = api.raw_query(QUERY_V2, {"modelId": model_id})
        rounds = res.get("data", {}).get("v2RoundModelPerformances", [])
        for r in rounds:
            rd = (
                pd.to_datetime(str(r.get("roundDataDatestamp")))
                if r.get("roundDataDatestamp")
                else None
            )
            rst = (
                _dt_naive(r.get("roundScoreTime")) if r.get("roundScoreTime") else None
            )
            rrt = (
                _dt_naive(r.get("roundResolveTime"))
                if r.get("roundResolveTime")
                else None
            )
            payout = (
                float(r["roundPayoutFactor"])
                if r.get("roundPayoutFactor") is not None
                else None
            )
            at_risk = float(r["atRisk"]) if r.get("atRisk") is not None else None
            for s in r.get("submissionScores", []) or []:
                all_rows.append(
                    {
                        "modelName": model_name,
                        "roundNumber": r.get("roundNumber"),
                        "roundDate": rd,
                        "roundScoreTime": rst,
                        "roundResolveTime": rrt,
                        "roundPayoutFactor": payout,
                        "atRisk": at_risk,
                        "displayName": s.get("displayName"),
                        "value": s.get("value"),
                        "percentile": s.get("percentile"),
                        "day": s.get("day"),
                        "date": _dt_naive(s.get("date")) if s.get("date") else None,
                        "resolveDate": (
                            _dt_naive(s.get("resolveDate"))
                            if s.get("resolveDate")
                            else None
                        ),
                        "resolved": s.get("resolved"),
                    }
                )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Types
    for col in ["roundScoreTime", "roundResolveTime", "date", "resolveDate"]:
        if col in df.columns:
            try:
                tz = getattr(df[col].dt, "tz", None)
                if tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
            except Exception:
                pass
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float32")
    if "percentile" in df.columns:
        df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce").astype(
            "float32"
        )
    if "roundPayoutFactor" in df.columns:
        df["roundPayoutFactor"] = pd.to_numeric(
            df["roundPayoutFactor"], errors="coerce"
        ).astype("float32")
    if "atRisk" in df.columns:
        df["atRisk"] = pd.to_numeric(df["atRisk"], errors="coerce").astype("float32")
    if "roundNumber" in df.columns:
        df["roundNumber"] = pd.to_numeric(
            df["roundNumber"], errors="coerce", downcast="integer"
        )
    if "displayName" in df.columns:
        try:
            df["displayName"] = df["displayName"].astype("category")
        except Exception:
            pass
    if "day" in df.columns:
        try:
            df["day"] = df["day"].astype("category")
        except Exception:
            pass

    # Filtre metrics
    metrics = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    df_filt = df[df["displayName"].isin(metrics)].copy()
    try:
        if isinstance(df_filt["displayName"].dtype, pd.CategoricalDtype):
            df_filt["displayName"] = df_filt[
                "displayName"
            ].cat.remove_unused_categories()
    except Exception:
        pass

    # Dédup — 1 par (modelName, roundNumber, displayName), priorité résolu puis plus récent
    if "resolved" in df_filt.columns:
        df_filt["resolved"] = df_filt["resolved"].fillna(False)
    sort_cols = [
        c
        for c in ["modelName", "roundNumber", "displayName", "resolved", "date"]
        if c in df_filt.columns
    ]
    if sort_cols:
        df_filt = df_filt.sort_values(sort_cols)
    df_filt = df_filt.drop_duplicates(
        subset=["modelName", "roundNumber", "displayName"], keep="last"
    )

    # Pivot wide
    index_cols = [
        "modelName",
        "roundNumber",
        "roundDate",
        "roundScoreTime",
        "roundResolveTime",
        "roundPayoutFactor",
        "atRisk",
    ]
    index_cols = [c for c in index_cols if c in df_filt.columns]
    if not index_cols:
        index_cols = ["modelName", "roundNumber"]

    df_wide = df_filt.pivot(
        index=index_cols, columns="displayName", values="value"
    ).reset_index()

    if (
        add_payout
        and "roundPayoutFactor" in df_wide.columns
        and "season_score" in df_wide.columns
    ):
        df_wide["payout"] = df_wide["roundPayoutFactor"] * df_wide["season_score"]
    return df_wide


def gain_timeseries(wide_df: pd.DataFrame, stakes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le gain/perte quotidien en fonction des stakes et du payout.
    - wide_df: DataFrame avec au moins ['roundDate', 'payout']
    - stakes_df: DataFrame avec au moins ['date', 'stake']
    Retourne un DataFrame avec ['date', 'gain'] (gain/perte par jour).
    """
    # S'assurer que les dates sont au format datetime.date
    if "roundDate" in wide_df.columns:
        wide_df = wide_df.copy()
        wide_df["roundDate"] = pd.to_datetime(wide_df["roundDate"]).dt.date
    if "date" in stakes_df.columns:
        stakes_df = stakes_df.copy()
        stakes_df["date"] = pd.to_datetime(stakes_df["date"]).dt.date
    # Fusionner les deux tables sur la date
    merged = pd.merge(
        stakes_df,
        wide_df[["roundDate", "payout"]],
        left_on="date",
        right_on="roundDate",
        how="left",
    )
    # Calculer le gain: payout * stake (si payout non nul)
    merged["gain"] = merged["stake"] * merged["payout"].fillna(0)
    # Retourner la série temporelle
    return merged[["date", "gain"]]


__all__ = ["build_numerai_wide_df", "DEFAULT_METRICS", "gain_timeseries"]


if __name__ == "__main__":
    # Test rapide
    df = build_numerai_wide_df()
    print(df.head(3))
    print(df.dtypes)
    print(df.shape)
    print(df.columns.tolist())
    print(df.describe(include="all").T)
    print(df.info())
