import os
import datetime as dt
from typing import Dict, List

import pandas as pd

# Import NumerAPI de manière robuste
try:
    from numerapi.numerapi import NumerAPI  # type: ignore
except Exception:  # pragma: no cover
    from numerapi import NumerAPI  # type: ignore[attr-defined]

NUMERAI_PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID")
NUMERAI_SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY")


QUERY_V2 = """
query($modelId: String!) {
  v2RoundModelPerformances(modelId: $modelId) {
    roundNumber
    roundDataDatestamp
    roundScoreTime
    roundResolveTime
    roundPayoutFactor
    roundResolved
    submissionScores {
      displayName
      value
    }
  }
}
"""


def get_napi() -> NumerAPI:
    if NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY:
        return NumerAPI(public_id=NUMERAI_PUBLIC_ID, secret_key=NUMERAI_SECRET_KEY)
    # non authentifié
    return NumerAPI()


def get_models(napi: NumerAPI) -> Dict[str, str]:
    # Essaie get_models(), si dict; sinon fallback GraphQL me{models}
    try:
        raw = napi.get_models()
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items() if k and v}
    except Exception:
        pass
    # Fallback GraphQL
    try:
        q = """
        query { me { models { id name } } }
        """
        res = napi.raw_query(q)
        items = res.get("data", {}).get("me", {}).get("models", [])
        return {
            str(i.get("name") or i.get("id")): str(i.get("id"))
            for i in items
            if i.get("id")
        }
    except Exception:
        return {}


def fetch_rounds_for_models(napi: NumerAPI, model_ids: List[str]) -> List[dict]:
    if not model_ids:
        return []
    q = """
    query($modelIds: [String!]!) {
      roundsV2: rounds {
        number
        tournament
        openTime
        resolveTime
        modelPerformances(modelIds: $modelIds) {
          modelId
          corr { payout pendingPayout }
          mmc { payout pendingPayout }
        }
      }
    }
    """
    res = napi.raw_query(q, variables={"modelIds": model_ids})
    return res.get("data", {}).get("roundsV2", [])


def _to_date(val) -> pd.Timestamp | None:
    if not val:
        return None
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def fetch_v2_rounds(napi: NumerAPI, model_id: str) -> List[dict]:
    try:
        res = napi.raw_query(QUERY_V2, variables={"modelId": model_id})
        return res.get("data", {}).get("v2RoundModelPerformances", []) or []
    except Exception:
        return []


def extract_payouts_from_rounds(
    rounds: List[dict], settled_only: bool = True
) -> pd.DataFrame:
    rows: list[dict] = []
    for r in rounds:
        try:
            rn_val = r.get("number")
            if rn_val is None:
                continue
            rnd = int(rn_val)
            rdate_ts = _to_date(r.get("openTime"))
            resolve_ts = _to_date(r.get("resolveTime"))
            for mp in r.get("modelPerformances", []) or []:
                mid = mp.get("modelId")
                for metric in ("corr", "mmc"):
                    node = mp.get(metric) or {}
                    settled = node.get("payout")
                    pending = node.get("pendingPayout")
                    if settled_only:
                        val = settled
                        resolved = True
                    else:
                        val = settled if settled is not None else pending
                        resolved = settled is not None
                    if val is None:
                        continue
                    rows.append(
                        {
                            "model_id": mid,
                            "roundNumber": rnd,
                            "roundDate": (
                                rdate_ts.date() if rdate_ts is not None else None
                            ),
                            "resolveDate": (
                                resolve_ts.date() if resolve_ts is not None else None
                            ),
                            "payout_metric": metric,
                            "payout_value": float(val),
                            "resolved": bool(resolved),
                        }
                    )
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["model_id", "roundNumber", "payout_metric"], inplace=True)
    return df


def extract_season_payouts(rounds: List[dict], model_id: str) -> pd.DataFrame:
    """Construit un DataFrame de payouts 'season' pour un modèle.
    payout_value = r = season_score * roundPayoutFactor (retour par round)
    """
    out: list[dict] = []
    now = pd.Timestamp.utcnow()
    for r in rounds:
        try:
            rn = r.get("roundNumber")
            if rn is None:
                continue
            rn = int(rn)
            rdate = r.get("roundDataDatestamp") or r.get("roundScoreTime")
            rdate_ts = _to_date(rdate)
            rres_ts = _to_date(r.get("roundResolveTime"))
            rpf = r.get("roundPayoutFactor")
            rpf = float(rpf) if rpf is not None else None
            if rpf is None:
                continue
            # Cherche le season score
            svalue = None
            for s in r.get("submissionScores", []) or []:
                name = str(s.get("displayName", "")).strip().lower()
                if "season" in name:  # inclut Season score
                    try:
                        svalue = (
                            float(s.get("value"))
                            if s.get("value") is not None
                            else None
                        )
                    except Exception:
                        svalue = None
                    if svalue is not None:
                        break
            if svalue is None:
                continue
            ret = float(svalue) * float(rpf)
            resolved_flag = bool(r.get("roundResolved"))
            if not resolved_flag and rres_ts is not None:
                # fallback: considérer résolu si la date est passée
                try:
                    resolved_flag = rres_ts.tz_localize(None) <= now.tz_localize(None)  # type: ignore
                except Exception:
                    resolved_flag = False
            out.append(
                {
                    "model_id": str(model_id),
                    "roundNumber": rn,
                    "roundDate": rdate_ts.date() if rdate_ts is not None else None,
                    "resolveDate": rres_ts.date() if rres_ts is not None else None,
                    "payout_metric": "season",
                    "payout_value": ret,
                    "resolved": bool(resolved_flag),
                }
            )
        except Exception:
            continue
    df = pd.DataFrame(out)
    if not df.empty:
        df.sort_values(["model_id", "roundNumber"], inplace=True)
    return df
