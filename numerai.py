import json, io, gc, numpy as np, pandas as pd, streamlit as st


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


def fetch_v2_rounds(model_id, pub, sec):
    api = NumerAPI(public_id=pub, secret_key=sec)
    res = api.raw_query(QUERY_V2, {"modelId": model_id})
    return res["data"]["v2RoundModelPerformances"]


def _dt_naive(x):
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


def build_long_df(rounds):
    rows = []
    for r in rounds:
        rd = (
            pd.to_datetime(str(r.get("roundDataDatestamp")))
            if r.get("roundDataDatestamp")
            else None
        )
        rst = _dt_naive(r.get("roundScoreTime")) if r.get("roundScoreTime") else None
        rrt = (
            _dt_naive(r.get("roundResolveTime")) if r.get("roundResolveTime") else None
        )
        payout = (
            float(r["roundPayoutFactor"])
            if r.get("roundPayoutFactor") is not None
            else None
        )
        at_risk = float(r["atRisk"]) if r.get("atRisk") is not None else None
        for s in r.get("submissionScores", []):
            rows.append(
                {
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
    df = pd.DataFrame(rows)
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
    return df


def get_rounds_df(model_id, pub, sec):
    rounds = fetch_v2_rounds(model_id, pub, sec)
    df = build_long_df(rounds)
    return df


public_id = st.secrets["NUMERAI_PUBLIC_ID"]
secret_key = st.secrets["NUMERAI_SECRET_KEY"]
# model_id = "VOTRE_MODEL_ID"
# df = get_rounds_df(model_id, public_id, secret_key)
# print(df.head())
try:
    from numerapi.numerapi import NumerAPI
except Exception:  # fallback pour environnements où l'API est exposée différemment
    from numerapi import NumerAPI
