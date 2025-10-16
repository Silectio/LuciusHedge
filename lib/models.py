from sqlalchemy import (
    Column,
    String,
    Date,
    Float,
    Boolean,
    Integer,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Investor(Base):
    __tablename__ = "investors"
    investor_id = Column(String, primary_key=True)
    name = Column(String)
    email = Column(String)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime)
    notes = Column(String)


class Stake(Base):
    __tablename__ = "stakes"
    stake_id = Column(String, primary_key=True)
    investor_id = Column(String)
    model_id = Column(String, index=True)
    amount = Column(Float)
    start_date = Column(Date)
    end_date = Column(Date, nullable=True)
    created_at = Column(DateTime)
    notes = Column(String)


class Payout(Base):
    __tablename__ = "payouts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, index=True)
    roundNumber = Column(Integer, index=True)
    roundDate = Column(Date)
    resolveDate = Column(Date)
    payout_metric = Column(String)
    payout_value = Column(Float)
    resolved = Column(Boolean)
    inserted_at = Column(Date)
    __table_args__ = (
        UniqueConstraint(
            "model_id", "roundNumber", "payout_metric", name="uq_round_metric_model"
        ),
    )


class Allocation(Base):
    __tablename__ = "allocations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, index=True)
    roundNumber = Column(Integer, index=True)
    roundDate = Column(Date)
    investor_id = Column(String)
    stake_used = Column(Float)
    pool_active = Column(Float)
    share_value = Column(Float)
    payout_metric = Column(String)
    inserted_at = Column(Date)
