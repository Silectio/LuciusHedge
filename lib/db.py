import os
import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Investor, Stake, Payout, Allocation


def get_engine_and_session() -> Tuple[Engine, sessionmaker]:
    # Lecture DATABASE_URL depuis env puis secrets Streamlit
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        try:
            import streamlit as st  # type: ignore

            db_url = st.secrets.get("DATABASE_URL")  # type: ignore
        except Exception:
            db_url = None
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL manquant. Définissez-le dans .streamlit/secrets.toml ou comme variable d'environnement."
        )
    engine = create_engine(db_url, future=True)
    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    # Crée/ajuste le schéma
    ensure_schema(engine)
    return engine, SessionLocal


def ensure_schema(engine: Engine) -> None:
    # Crée les tables si absentes
    Base.metadata.create_all(engine)

    # Migration légère: contraintes uniques
    with engine.begin() as conn:
        # Supprimer l'ancien index/contrainte si elle existe
        for stmt in (
            "DROP INDEX IF EXISTS uq_round_metric",
            "ALTER TABLE payouts DROP CONSTRAINT IF EXISTS uq_round_metric",
            "DROP INDEX IF EXISTS uq_round_metric_idx",
        ):
            try:
                conn.execute(text(stmt))
            except Exception:
                pass
    # Base.metadata.create_all se charge du nouvel UniqueConstraint


# CRUD Investors


def list_investors_df(session: Session) -> pd.DataFrame:
    rows = session.query(Investor).all()
    data = [
        {
            "investor_id": r.investor_id,
            "name": r.name,
            "email": r.email,
            "active": r.active,
            "created_at": r.created_at,
            "notes": r.notes,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def upsert_investor(
    session: Session,
    investor_id: str,
    name: str,
    email: Optional[str] = None,
    active: bool = True,
    notes: Optional[str] = None,
) -> None:
    inv = session.get(Investor, investor_id)
    now = dt.datetime.utcnow()
    if inv:
        # utiliser setattr pour éviter les soucis de typage statique
        setattr(inv, "name", name)
        setattr(inv, "email", email)
        setattr(inv, "active", active)
        setattr(inv, "notes", notes)
    else:
        inv = Investor(
            investor_id=investor_id,
            name=name,
            email=email,
            active=active,
            created_at=now,
            notes=notes,
        )
        session.add(inv)


# CRUD Stakes


def list_stakes_df(session: Session) -> pd.DataFrame:
    rows = session.query(Stake).all()
    data = [
        {
            "stake_id": r.stake_id,
            "investor_id": r.investor_id,
            "model_id": r.model_id,
            "amount": r.amount,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "created_at": r.created_at,
            "notes": r.notes,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def add_or_update_stake(
    session: Session,
    stake_id: str,
    investor_id: str,
    model_id: str,
    amount: float,
    start_date: dt.date,
    end_date: Optional[dt.date] = None,
    notes: Optional[str] = None,
) -> None:
    stobj = session.get(Stake, stake_id)
    now = dt.datetime.utcnow()
    if stobj:
        setattr(stobj, "investor_id", investor_id)
        setattr(stobj, "model_id", model_id)
        setattr(stobj, "amount", float(amount))
        setattr(stobj, "start_date", start_date)
        setattr(stobj, "end_date", end_date)
        setattr(stobj, "notes", notes)
    else:
        stobj = Stake(
            stake_id=stake_id,
            investor_id=investor_id,
            model_id=model_id,
            amount=float(amount),
            start_date=start_date,
            end_date=end_date,
            created_at=now,
            notes=notes,
        )
        session.add(stobj)


# Payouts


def list_payouts_df(
    session: Session, model_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    q = session.query(Payout)
    if model_ids:
        q = q.filter(Payout.model_id.in_(model_ids))
    rows = q.all()
    data = [
        {
            "model_id": r.model_id,
            "roundNumber": r.roundNumber,
            "roundDate": r.roundDate,
            "resolveDate": r.resolveDate,
            "payout_metric": r.payout_metric,
            "payout_value": r.payout_value,
            "resolved": r.resolved,
            "inserted_at": r.inserted_at,
        }
        for r in rows
    ]
    df = pd.DataFrame(data)
    if not df.empty:
        df.sort_values(["model_id", "roundNumber", "payout_metric"], inplace=True)
    return df


def upsert_payouts(session: Session, payouts: pd.DataFrame) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    now = dt.datetime.utcnow().date()

    if payouts.empty:
        return inserted, skipped

    for _, r in payouts.iterrows():
        obj = Payout(
            model_id=str(r["model_id"]),
            roundNumber=int(r["roundNumber"]),
            roundDate=(
                pd.to_datetime(r["roundDate"]).date()
                if pd.notna(r["roundDate"])
                else None
            ),
            resolveDate=(
                pd.to_datetime(r["resolveDate"]).date()
                if pd.notna(r["resolveDate"])
                else None
            ),
            payout_metric=str(r["payout_metric"]),
            payout_value=float(r["payout_value"]),
            resolved=bool(r.get("resolved", True)),
            inserted_at=now,
        )
        try:
            session.add(obj)
            session.flush()
            inserted += 1
        except IntegrityError:
            session.rollback()
            skipped += 1
            # Duplicate based on unique constraint; ignore
    return inserted, skipped


# Allocations


def list_allocations_df(
    session: Session, model_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    q = session.query(Allocation)
    if model_ids:
        q = q.filter(Allocation.model_id.in_(model_ids))
    rows = q.all()
    data = [
        {
            "model_id": r.model_id,
            "roundNumber": r.roundNumber,
            "roundDate": r.roundDate,
            "investor_id": r.investor_id,
            "stake_used": r.stake_used,
            "pool_active": r.pool_active,
            "share_value": r.share_value,
            "payout_metric": r.payout_metric,
            "inserted_at": r.inserted_at,
        }
        for r in rows
    ]
    df = pd.DataFrame(data)
    if not df.empty:
        df.sort_values(
            ["model_id", "roundNumber", "investor_id", "payout_metric"], inplace=True
        )
    return df


def replace_allocations(session: Session, allocs: pd.DataFrame) -> None:
    if allocs.empty:
        return
    now = dt.datetime.utcnow().date()
    # Supprimer les anciennes allocs pour les rounds concernés
    rounds = (
        allocs[["model_id", "roundNumber"]]
        .drop_duplicates()
        .to_records(index=False)
        .tolist()
    )
    for mid, rn in rounds:
        session.query(Allocation).filter(
            Allocation.model_id == str(mid), Allocation.roundNumber == int(rn)
        ).delete(synchronize_session=False)
    # Inserer les nouvelles
    for _, r in allocs.iterrows():
        obj = Allocation(
            model_id=str(r["model_id"]),
            roundNumber=int(r["roundNumber"]),
            roundDate=(
                pd.to_datetime(r["roundDate"]).date()
                if pd.notna(r["roundDate"])
                else None
            ),
            investor_id=str(r["investor_id"]),
            stake_used=float(r["stake_used"]),
            pool_active=float(r["pool_active"]),
            share_value=float(r["share_value"]),
            payout_metric=str(r["payout_metric"]),
            inserted_at=now,
        )
        session.add(obj)


def compute_allocations(
    session: Session, model_id: str, payout_metric: str = "season"
) -> pd.DataFrame:
    payouts_df = list_payouts_df(session, [model_id])
    payouts_df = payouts_df[payouts_df["payout_metric"] == payout_metric]
    if payouts_df.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "roundNumber",
                "roundDate",
                "investor_id",
                "stake_used",
                "pool_active",
                "share_value",
                "payout_metric",
            ]
        )  # empty

    stakes_df = list_stakes_df(session)
    stakes_df = stakes_df[stakes_df["model_id"] == model_id].copy()

    allocations: List[dict] = []
    for _, prow in payouts_df.iterrows():
        resolve_date = prow["resolveDate"] or prow["roundDate"]
        active = stakes_df[
            (stakes_df["start_date"] <= resolve_date)
            & ((stakes_df["end_date"].isna()) | (stakes_df["end_date"] >= resolve_date))
        ].copy()
        pool = float(active["amount"].sum()) if not active.empty else 0.0
        if pool <= 0:
            continue
        for _, s in active.iterrows():
            weight = float(s["amount"]) / pool
            share = float(prow["payout_value"]) * weight
            allocations.append(
                {
                    "model_id": model_id,
                    "roundNumber": int(prow["roundNumber"]),
                    "roundDate": prow["roundDate"],
                    "investor_id": s["investor_id"],
                    "stake_used": float(s["amount"]),
                    "pool_active": pool,
                    "share_value": share,
                    "payout_metric": payout_metric,
                }
            )
    return pd.DataFrame(allocations)


def _geom_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    # Si x est un facteur de rendement (1+r), géométrique = prod(x)**(1/n)
    # Ici payout_value ~ r, on convertit en facteur 1+r
    factors = 1.0 + x
    factors = factors[factors > 0]  # éviter valeurs non-positives
    if factors.empty:
        return float("nan")
    return float(np.prod(factors) ** (1.0 / len(factors)) - 1.0)


def model_return_stats(session: Session) -> pd.DataFrame:
    """Retourne pour chaque modèle les moyennes géométriques journalière, mensuelle (~21j) et annuelle (~251j) sur payout_metric='season'."""
    df = list_payouts_df(session, None)
    if df.empty:
        return pd.DataFrame(
            columns=["model_id", "geo_daily", "geo_monthly", "geo_annual"]
        )
    df = df[df["payout_metric"] == "season"].copy()
    # On suppose une observation par round (quotidienne). On calcule sur l’ensemble dispo.
    stats = []
    for mid, grp in df.groupby("model_id"):
        daily = _geom_mean(grp["payout_value"])  # géométrique quotidien
        # extrapolations: (1+daily)^21-1 et (1+daily)^251-1
        if not np.isnan(daily):
            monthly = float((1.0 + daily) ** 21 - 1.0)
            annual = float((1.0 + daily) ** 251 - 1.0)
        else:
            monthly = float("nan")
            annual = float("nan")
        stats.append(
            {
                "model_id": mid,
                "geo_daily": daily,
                "geo_monthly": monthly,
                "geo_annual": annual,
            }
        )
    return pd.DataFrame(stats)


def pending_gains_by_investor(session: Session) -> pd.DataFrame:
    """Calcule, par investisseur, la somme des allocations sur rounds non résolus (resolved=False) pour payout_metric='season'."""
    allocs = list_allocations_df(session, None)
    if allocs.empty:
        return pd.DataFrame(columns=["investor_id", "pending_gain"])
    df = allocs.copy()
    # Joindre info resolved depuis payouts
    pays = list_payouts_df(session, None)
    if pays.empty:
        return pd.DataFrame(columns=["investor_id", "pending_gain"])
    pays = (
        pays[pays["payout_metric"] == "season"]["model_id", "roundNumber", "resolved"]
        if "resolved" in pays.columns
        else pays
    )
    try:
        pays = pays[["model_id", "roundNumber", "resolved"]]
    except Exception:
        return pd.DataFrame(columns=["investor_id", "pending_gain"])
    merged = df.merge(pays, on=["model_id", "roundNumber"], how="left")
    pending = merged[merged["resolved"] == False]
    out = pending.groupby("investor_id")["share_value"].sum().reset_index()
    out = out.rename(columns={"share_value": "pending_gain"})
    return out
