"""
Utilitaires base de données pour PostgreSQL (SQLAlchemy Core)
- Connexion via DATABASE_URL (env ou Streamlit secrets)
- Fonctions CRUD: fetch_rows, insert_row, bulk_insert, upsert_row, update_rows, delete_rows
- SQL brut: execute_sql
- DDL de base: create_table, add_column, drop_column, rename_column, alter_column_type

Exemple rapide:
    from db import insert_row, fetch_rows
    insert_row("users", {"email": "a@b.com", "name": "Alice"})
    users = fetch_rows("users", where={"email": "a@b.com"})
"""

from __future__ import annotations

import os
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from sqlalchemy import (
    Column,
    MetaData,
    Table,
    UniqueConstraint,
    Index,
    and_,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import ClauseElement
from sqlalchemy.types import (
    Integer,
    String,
    Text,
    Boolean,
    Float,
    Numeric,
    DateTime,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Types utilitaires
WhereType = Optional[Union[Mapping[str, Any], Any]]
ColumnsSpec = Mapping[str, Union[str, Any]]  # Any peut être un type SQLAlchemy

_engine: Optional[Engine] = None
_metadata: Optional[MetaData] = None


# ------------------------- Connexion & Meta -------------------------


def _get_database_url() -> str:
    # 1) env var
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    # 2) streamlit secrets
    try:
        import streamlit as st  # type: ignore

        if "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])  # pragma: no cover
    except Exception:
        pass
    raise RuntimeError(
        "DATABASE_URL introuvable. Définissez une variable d'environnement ou .streamlit/secrets.toml."
    )


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = _get_database_url()
        _engine = create_engine(
            url,
            future=True,
            pool_pre_ping=True,
        )
    return _engine


def get_metadata() -> MetaData:
    global _metadata
    if _metadata is None:
        _metadata = MetaData()
    return _metadata


def _validate_identifier(name: str) -> str:
    """Validation minimiste pour éviter l'injection dans les identifiants.
    Autorise lettres, chiffres, underscore et le point (pour schema.table).
    """
    if not re.fullmatch(r"[A-Za-z0-9_\.]+", name or ""):
        raise ValueError(f"Identifiant invalide: {name!r}")
    return name


def _table_exists(table_name: str, schema: Optional[str] = None) -> bool:
    eng = get_engine()
    insp = inspect(eng)
    return insp.has_table(table_name, schema=schema)


def _column_exists(
    table_name: str, column_name: str, schema: Optional[str] = None
) -> bool:
    eng = get_engine()
    insp = inspect(eng)
    for col in insp.get_columns(table_name, schema=schema):
        if col.get("name") == column_name:
            return True
    return False


def _get_table(table_name: str, schema: Optional[str] = None) -> Table:
    _validate_identifier(table_name)
    if schema:
        _validate_identifier(schema)
    md = get_metadata()
    eng = get_engine()
    # Refléter la table existante en forçant un refresh du cache MetaData
    table_key = f"{schema}.{table_name}" if schema else table_name
    if table_key in md.tables:
        md.remove(md.tables[table_key])
    return Table(table_name, md, schema=schema, autoload_with=eng)  # type: ignore


def _build_where_clause(table: Table, where: WhereType) -> Optional[ClauseElement]:
    if where is None:
        return None
    if isinstance(where, ClauseElement):
        return where
    if isinstance(where, Mapping):
        clauses = []
        for k, v in where.items():
            if k not in table.c:
                raise KeyError(f"Colonne inconnue dans WHERE: {k}")
            clauses.append(table.c[k] == v)
        return and_(*clauses) if clauses else None
    raise TypeError(
        "Paramètre 'where' doit être un mapping ou une expression SQLAlchemy"
    )


# ------------------------- Helpers Types -------------------------


def _parse_type(col_type: Union[str, Any]) -> Any:
    """Convertit une chaîne simple en type SQLAlchemy si nécessaire.
    Si col_type est déjà un type SQLAlchemy (ex: String(100)), il est renvoyé tel quel.
    """
    if not isinstance(col_type, str):
        return col_type
    t = col_type.strip().lower()
    # patterns simples
    if t in {"int", "integer"}:
        return Integer()
    if t.startswith("varchar(") or t.startswith("string("):
        m = re.search(r"(varchar|string)\((\d+)\)", t)
        size = int(m.group(2)) if m else 255
        return String(length=size)
    if t in {"varchar", "string", "str"}:
        return String(length=255)
    if t in {"text"}:
        return Text()
    if t in {"bool", "boolean"}:
        return Boolean()
    if t in {"float", "double"}:
        return Float()
    if t.startswith("numeric(") or t.startswith("decimal("):
        m = re.search(r"(numeric|decimal)\((\d+),(\d+)\)", t)
        if m:
            return Numeric(precision=int(m.group(2)), scale=int(m.group(3)))
        return Numeric()
    if t in {"numeric", "decimal"}:
        return Numeric()
    if t in {"datetime", "timestamp"}:
        return DateTime(timezone=True)
    # défaut
    return Text()


# ------------------------- CRUD -------------------------


def fetch_rows(
    table_name: str,
    where: WhereType = None,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    order_by: Optional[Union[str, Sequence[str]]] = None,
    schema: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retourne des lignes sous forme de liste de dicts."""
    tbl = _get_table(table_name, schema)

    if columns:
        for c in columns:
            if c not in tbl.c:
                raise KeyError(f"Colonne inconnue demandée: {c}")
        sel = select(*(tbl.c[c] for c in columns))
    else:
        sel = select(tbl)

    cond = _build_where_clause(tbl, where)
    if cond is not None:
        sel = sel.where(cast(Any, cond))

    if order_by:
        if isinstance(order_by, str):
            order_by = [order_by]
        for ob in order_by:
            desc = ob.strip().lower().endswith(" desc")
            colname = ob.replace(" desc", "").replace(" DESC", "").strip()
            if colname not in tbl.c:
                raise KeyError(f"Colonne inconnue pour order_by: {colname}")
            sel = sel.order_by(tbl.c[colname].desc() if desc else tbl.c[colname].asc())

    if limit is not None:
        sel = sel.limit(limit)

    eng = get_engine()
    with eng.connect() as conn:
        rows = conn.execute(sel).mappings().all()
        return [dict(r) for r in rows]


def insert_row(
    table_name: str,
    data: Mapping[str, Any],
    schema: Optional[str] = None,
    returning: bool = False,
) -> Optional[Dict[str, Any]]:
    """Insère une ligne. Si returning=True retourne la ligne insérée."""
    if not data:
        raise ValueError("'data' ne peut pas être vide")
    tbl = _get_table(table_name, schema)
    stmt = tbl.insert().values(**dict(data))
    if returning:
        stmt = stmt.returning(*tbl.c)
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(stmt)
        if returning:
            row = res.mappings().first()
            return dict(row) if row else None
    return None


def bulk_insert(
    table_name: str,
    rows: Iterable[Mapping[str, Any]],
    schema: Optional[str] = None,
) -> int:
    """Insère en masse. Retourne le nombre de lignes insérées."""
    rows = list(rows)
    if not rows:
        return 0
    tbl = _get_table(table_name, schema)
    stmt = tbl.insert()
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(stmt, rows)
        return res.rowcount or 0


def upsert_row(
    table_name: str,
    data: Mapping[str, Any],
    conflict_columns: Sequence[str],
    update_columns: Optional[Sequence[str]] = None,
    schema: Optional[str] = None,
    returning: bool = False,
) -> Optional[Dict[str, Any]]:
    """UPSERT (INSERT ... ON CONFLICT DO UPDATE) pour PostgreSQL.
    - conflict_columns: colonnes du conflit (index unique/PK)
    - update_columns: colonnes mises à jour en cas de conflit (par défaut toutes hors clés)
    """
    if not data:
        raise ValueError("'data' ne peut pas être vide")
    tbl = _get_table(table_name, schema)
    for c in conflict_columns:
        if c not in tbl.c:
            raise KeyError(f"Colonne de conflit inconnue: {c}")
    ins = pg_insert(tbl).values(**dict(data))
    if update_columns is None:
        update_columns = [c for c in data.keys() if c not in set(conflict_columns)]
    update_dict = {c: ins.excluded[c] for c in update_columns}
    stmt = ins.on_conflict_do_update(
        index_elements=list(conflict_columns), set_=update_dict
    )
    if returning:
        stmt = stmt.returning(*tbl.c)
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(stmt)
        if returning:
            row = res.mappings().first()
            return dict(row) if row else None
    return None


def update_rows(
    table_name: str,
    where: WhereType,
    values: Mapping[str, Any],
    schema: Optional[str] = None,
    returning: bool = False,
) -> Union[int, List[Dict[str, Any]]]:
    """Met à jour des lignes. Retourne le nombre de lignes ou les lignes si returning=True."""
    if not values:
        raise ValueError("'values' ne peut pas être vide")
    tbl = _get_table(table_name, schema)
    cond = _build_where_clause(tbl, where)
    if cond is None:
        raise ValueError("'where' est requis pour update_rows")
    stmt = tbl.update().where(cast(Any, cond)).values(**dict(values))
    if returning:
        stmt = stmt.returning(*tbl.c)
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(stmt)
        if returning:
            return [dict(r) for r in res.mappings().all()]
        return res.rowcount or 0


essential_delete_warning = (
    "ATTENTION: suppression sans WHERE interdit pour éviter les erreurs."
)


def delete_rows(
    table_name: str,
    where: WhereType,
    schema: Optional[str] = None,
    returning: bool = False,
) -> Union[int, List[Dict[str, Any]]]:
    """Supprime des lignes. Retourne le nombre de lignes ou les lignes si returning=True."""
    tbl = _get_table(table_name, schema)
    cond = _build_where_clause(tbl, where)
    if cond is None:
        raise ValueError(essential_delete_warning)
    stmt = tbl.delete().where(cast(Any, cond))
    if returning:
        stmt = stmt.returning(*tbl.c)
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(stmt)
        if returning:
            return [dict(r) for r in res.mappings().all()]
        return res.rowcount or 0


# ------------------------- SQL brut -------------------------


def execute_sql(
    sql: str, params: Optional[Mapping[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Exécute un SQL brut en lecture (ou écriture). Retourne les lignes si SELECT.
    NOTE: utilisez avec prudence et privilégiez les fonctions ci-dessus.
    """
    eng = get_engine()
    with eng.begin() as conn:
        res = conn.execute(text(sql), params or {})
        try:
            rows = res.mappings().all()
            return [dict(r) for r in rows]
        except SQLAlchemyError:
            return []


# ------------------------- DDL (schéma) -------------------------


def create_table(
    table_name: str,
    columns: ColumnsSpec,
    *,
    primary_key: Optional[Sequence[str]] = None,
    uniques: Optional[Sequence[Sequence[str]]] = None,
    indexes: Optional[Sequence[Mapping[str, Any]]] = None,
    if_not_exists: bool = True,
    schema: Optional[str] = None,
) -> None:
    """Crée une table simple.
    - columns: {col_name: type}, type = str ("varchar(100)") ou type SQLAlchemy (String(100))
    - primary_key: [col1, col2]
    - uniques: [[col1], [col2, col3]]
    - indexes: [{"name": "ix_users_email", "columns": ["email"], "unique": True}]
    """
    _validate_identifier(table_name)
    if schema:
        _validate_identifier(schema)

    md = get_metadata()
    eng = get_engine()

    if if_not_exists and _table_exists(table_name, schema):
        return

    cols = []
    for name, typ in columns.items():
        _validate_identifier(name)
        sa_type = _parse_type(typ)
        is_pk = primary_key is not None and name in set(primary_key)
        cols.append(Column(name, sa_type, primary_key=is_pk))

    # Contraintes uniques (déclarées au niveau Table)
    uc_objs: List[UniqueConstraint] = []
    if uniques:
        for uq_cols in uniques:
            for c in uq_cols:
                if c not in columns:
                    raise KeyError(f"Colonne unique inconnue: {c}")
            uc_objs.append(
                UniqueConstraint(*uq_cols, name=f"uq_{table_name}_{'_'.join(uq_cols)}")
            )

    tbl = Table(table_name, md, *cols, *uc_objs, schema=schema)

    # Indexes (seront créés avec create_all car attachés à la table)
    if indexes:
        for ix in indexes:
            ix_cols = list(ix.get("columns", []))
            for c in ix_cols:
                if c not in columns:
                    raise KeyError(f"Colonne d'index inconnue: {c}")
            Index(
                ix.get("name", f"ix_{table_name}_{'_'.join(ix_cols)}"),
                *[tbl.c[c] for c in ix_cols],
                unique=bool(ix.get("unique", False)),
            )

    # Création de la table (et de ses index/contraintes)
    md.create_all(eng, tables=[tbl])


def add_column(
    table_name: str,
    column_name: str,
    col_type: Union[str, Any],
    *,
    nullable: bool = True,
    default: Optional[Any] = None,
    unique: bool = False,
    schema: Optional[str] = None,
) -> None:
    """Ajoute une colonne via ALTER TABLE si elle n'existe pas."""
    _validate_identifier(table_name)
    _validate_identifier(column_name)
    if schema:
        _validate_identifier(schema)

    if _column_exists(table_name, column_name, schema):
        return

    eng = get_engine()
    sa_type = _parse_type(col_type)
    type_sql = sa_type.compile(dialect=eng.dialect)

    tbl_qualified = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    col_def = f'"{column_name}" {type_sql}'
    if default is not None:
        col_def += f" DEFAULT :_default"
    if not nullable:
        col_def += " NOT NULL"

    sql = f"ALTER TABLE {tbl_qualified} ADD COLUMN {col_def}"

    with eng.begin() as conn:
        if default is not None:
            conn.execute(text(sql), {"_default": default})
        else:
            conn.execute(text(sql))
        if unique:
            uq_name = f"uq_{table_name}_{column_name}"
            conn.execute(
                text(
                    f'CREATE UNIQUE INDEX IF NOT EXISTS "{uq_name}" ON {tbl_qualified} ("{column_name}")'
                )
            )


def drop_column(
    table_name: str,
    column_name: str,
    *,
    schema: Optional[str] = None,
    cascade: bool = False,
) -> None:
    _validate_identifier(table_name)
    _validate_identifier(column_name)
    if schema:
        _validate_identifier(schema)
    if not _table_exists(table_name, schema) or not _column_exists(
        table_name, column_name, schema
    ):
        return
    eng = get_engine()
    tbl_qualified = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    cascade_sql = " CASCADE" if cascade else ""
    sql = f'ALTER TABLE {tbl_qualified} DROP COLUMN "{column_name}"{cascade_sql}'
    with eng.begin() as conn:
        conn.execute(text(sql))


def rename_column(
    table_name: str,
    old_name: str,
    new_name: str,
    *,
    schema: Optional[str] = None,
) -> None:
    _validate_identifier(table_name)
    _validate_identifier(old_name)
    _validate_identifier(new_name)
    if schema:
        _validate_identifier(schema)
    if not _column_exists(table_name, old_name, schema):
        raise KeyError(f"La colonne à renommer n'existe pas: {old_name}")
    if _column_exists(table_name, new_name, schema):
        raise KeyError(f"La nouvelle colonne existe déjà: {new_name}")
    eng = get_engine()
    tbl_qualified = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    sql = f'ALTER TABLE {tbl_qualified} RENAME COLUMN "{old_name}" TO "{new_name}"'
    with eng.begin() as conn:
        conn.execute(text(sql))


def alter_column_type(
    table_name: str,
    column_name: str,
    new_type: Union[str, Any],
    *,
    using_expression: Optional[str] = None,
    schema: Optional[str] = None,
) -> None:
    _validate_identifier(table_name)
    _validate_identifier(column_name)
    if schema:
        _validate_identifier(schema)
    if not _column_exists(table_name, column_name, schema):
        raise KeyError(f"Colonne inconnue: {column_name}")
    eng = get_engine()
    sa_type = _parse_type(new_type)
    type_sql = sa_type.compile(dialect=eng.dialect)
    tbl_qualified = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    sql = f'ALTER TABLE {tbl_qualified} ALTER COLUMN "{column_name}" TYPE {type_sql}'
    if using_expression:
        sql += f" USING {using_expression}"
    with eng.begin() as conn:
        conn.execute(text(sql))


__all__ = [
    # Connexion
    "get_engine",
    # CRUD
    "fetch_rows",
    "insert_row",
    "bulk_insert",
    "upsert_row",
    "update_rows",
    "delete_rows",
    # SQL brut
    "execute_sql",
    # DDL
    "create_table",
    "add_column",
    "drop_column",
    "rename_column",
    "alter_column_type",
]

# ========================= Domain: Investors & Stakes =========================

# Noms de tables
INVESTORS_TABLE = "investors"
STAKES_TABLE = "stakes"


def ensure_investment_tables() -> None:
    """Crée les tables `investors` et `stakes` si elles n'existent pas."""
    # investors: id (PK autoincrement), name, email, notes, created_at
    create_table(
        INVESTORS_TABLE,
        columns={
            "id": Integer(),  # PK auto-incrément
            "name": "varchar(255)",
            "email": "varchar(255)",
            "notes": "text",
            "created_at": "timestamp",
        },
        primary_key=["id"],
        uniques=[["email"]],
        indexes=[{"columns": ["email"], "unique": True}],
        if_not_exists=True,
    )

    # stakes: id (PK autoincrement), investor_id, amount, start_date, end_date
    # NB: Pas de contrainte FK via create_table utilitaire, mais index sur investor_id
    create_table(
        STAKES_TABLE,
        columns={
            "id": Integer(),
            "investor_id": "integer",
            "amount": "numeric(20,6)",
            "start_date": "timestamp",
            "end_date": "timestamp",
            "model_name": "varchar(255)",
            "notes": "text",
            "created_at": "timestamp",
            "updated_at": "timestamp",
        },
        primary_key=["id"],
        indexes=[
            {"columns": ["investor_id"], "unique": False},
            {"columns": ["model_name"], "unique": False},
        ],
        if_not_exists=True,
    )

    # Migration douce si la table existe déjà sans la colonne model_name
    try:
        if not _column_exists(STAKES_TABLE, "model_name"):
            add_column(STAKES_TABLE, "model_name", "varchar(255)", nullable=True)
            try:
                execute_sql(
                    'CREATE INDEX IF NOT EXISTS "ix_stakes_model_name" ON "stakes" ("model_name")'
                )
            except Exception:
                pass
    except Exception:
        pass


# ------------------------- Investors CRUD -------------------------


def add_investor(
    name: str, email: Optional[str] = None, notes: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Ajoute un investisseur et retourne la ligne insérée."""
    ensure_investment_tables()
    data: Dict[str, Any] = {"name": name}
    if email:
        data["email"] = email
    if notes:
        data["notes"] = notes
    return insert_row(INVESTORS_TABLE, data, returning=True)


def get_investors(
    where: WhereType = None, order_by: Optional[Union[str, Sequence[str]]] = ("id",)
) -> List[Dict[str, Any]]:
    ensure_investment_tables()
    return fetch_rows(INVESTORS_TABLE, where=where, order_by=order_by)


def get_investor_by_id(investor_id: int) -> Optional[Dict[str, Any]]:
    rows = get_investors(
        where={"id": investor_id},
        order_by=None,
    )
    return rows[0] if rows else None


def update_investor(
    investor_id: int, values: Mapping[str, Any]
) -> Union[int, List[Dict[str, Any]]]:
    ensure_investment_tables()
    # Nettoyer clés interdites
    values = {k: v for k, v in dict(values).items() if k in {"name", "email", "notes"}}
    if not values:
        return 0
    return update_rows(
        INVESTORS_TABLE, where={"id": investor_id}, values=values, returning=True
    )


def delete_investor(investor_id: int) -> Union[int, List[Dict[str, Any]]]:
    ensure_investment_tables()
    # Supprimer d'abord les stakes liés pour éviter orphelins
    delete_rows(STAKES_TABLE, where={"investor_id": investor_id})
    return delete_rows(INVESTORS_TABLE, where={"id": investor_id}, returning=True)


# ------------------------- Stakes CRUD -------------------------


def add_stake(
    investor_id: int,
    amount: Union[int, float],
    start_date: Any,  # datetime/date/str accepté par SQLAlchemy
    end_date: Optional[Any] = None,
    model_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Ajoute un stake pour un investisseur."""
    ensure_investment_tables()
    data: Dict[str, Any] = {
        "investor_id": investor_id,
        "amount": amount,
        "start_date": start_date,
    }
    if end_date is not None:
        data["end_date"] = end_date
    if model_name:
        data["model_name"] = model_name
    if notes:
        data["notes"] = notes
    return insert_row(STAKES_TABLE, data, returning=True)


def get_stakes(
    investor_id: Optional[int] = None,
    model_name: Optional[str] = None,
    order_by: Optional[Union[str, Sequence[str]]] = ("start_date desc",),
) -> List[Dict[str, Any]]:
    ensure_investment_tables()
    if investor_id is not None and model_name is not None:
        where: WhereType = {"investor_id": investor_id, "model_name": model_name}
    elif investor_id is not None:
        where = {"investor_id": investor_id}
    elif model_name is not None:
        where = {"model_name": model_name}
    else:
        where = None
    return fetch_rows(STAKES_TABLE, where=where, order_by=order_by)


def get_stake_by_id(stake_id: int) -> Optional[Dict[str, Any]]:
    rows = fetch_rows(STAKES_TABLE, where={"id": stake_id}, limit=1)
    return rows[0] if rows else None


def update_stake(
    stake_id: int, values: Mapping[str, Any]
) -> Union[int, List[Dict[str, Any]]]:
    ensure_investment_tables()
    allowed = {"investor_id", "amount", "start_date", "end_date", "model_name", "notes"}
    values = {k: v for k, v in dict(values).items() if k in allowed}
    if not values:
        return 0
    return update_rows(
        STAKES_TABLE, where={"id": stake_id}, values=values, returning=True
    )


def delete_stake(stake_id: int) -> Union[int, List[Dict[str, Any]]]:
    ensure_investment_tables()
    return delete_rows(STAKES_TABLE, where={"id": stake_id}, returning=True)


__all__ += [
    # Domain
    "INVESTORS_TABLE",
    "STAKES_TABLE",
    "ensure_investment_tables",
    # Investors
    "add_investor",
    "get_investors",
    "get_investor_by_id",
    "update_investor",
    "delete_investor",
    # Stakes
    "add_stake",
    "get_stakes",
    "get_stake_by_id",
    "update_stake",
    "delete_stake",
]
