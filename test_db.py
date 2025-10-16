"""
Script de démonstration/tests pour les utilitaires de base de données dans db.py.

Ce script exécute un cycle complet:
- Création d'une table de test
- Insert (simple et bulk)
- Lecture (SELECT)
- Upsert
- Update
- Delete
- DDL (add/rename/alter/drop column)
- SQL brut
- Nettoyage (DROP TABLE)

Exécuter: python test_db.py
Prérequis: DATABASE_URL défini dans l'env ou dans .streamlit/secrets.toml
"""

from __future__ import annotations

import sys
import uuid
from pprint import pprint

from db import (
    get_engine,
    create_table,
    add_column,
    rename_column,
    alter_column_type,
    drop_column,
    insert_row,
    bulk_insert,
    upsert_row,
    fetch_rows,
    update_rows,
    delete_rows,
    execute_sql,
)


def show(title: str, data):
    print(f"\n=== {title} ===")
    if isinstance(data, list):
        for row in data:
            pprint(row)
    else:
        pprint(data)


def main() -> int:
    # Vérifier la connexion
    try:
        execute_sql("SELECT 1")
        print("Connexion DB OK.")
    except Exception as e:
        print("Erreur de connexion DB:", e)
        return 1

    table_name = f"test_demo_{uuid.uuid4().hex[:8]}"
    print("Table de test:", table_name)

    try:
        # 1) CREATE TABLE
        create_table(
            table_name,
            columns={
                "id": "integer",
                "email": "varchar(255)",
                "name": "text",
                "balance": "numeric(10,2)",
                "created_at": "timestamp",
            },
            primary_key=["id"],
            uniques=[["email"]],
            indexes=[{"columns": ["email"], "unique": True}],
            if_not_exists=True,
        )
        print("Table créée.")

        # 2) INSERT simple
        r1 = insert_row(
            table_name,
            {
                "id": 1,
                "email": "alice@example.com",
                "name": "Alice",
                "balance": 10.50,
                "created_at": None,
            },
            returning=True,
        )
        show("Insert 1 (returning)", r1)

        # 3) BULK INSERT
        n = bulk_insert(
            table_name,
            [
                {
                    "id": 2,
                    "email": "bob@example.com",
                    "name": "Bob",
                    "balance": 20.0,
                    "created_at": None,
                },
                {
                    "id": 3,
                    "email": "carol@example.com",
                    "name": "Carol",
                    "balance": 30.25,
                    "created_at": None,
                },
            ],
        )
        show("Bulk insert count", n)

        # 4) SELECT
        rows = fetch_rows(table_name, order_by=["id"])
        show("Après insertions", rows)

        # 5) UPSERT (conflit sur email) -> met à jour name & balance
        up = upsert_row(
            table_name,
            {
                "id": 99,  # ignoré sur conflit si non repris dans update_columns
                "email": "alice@example.com",
                "name": "Alice V2",
                "balance": 15.75,
                "created_at": None,
            },
            conflict_columns=["email"],
            update_columns=["name", "balance"],
            returning=True,
        )
        show("Upsert (returning)", up)

        # 6) UPDATE (where)
        updated = update_rows(
            table_name,
            where={"id": 2},
            values={"name": "Bobby", "balance": 22.00},
            returning=True,
        )
        show("Update id=2 (returning)", updated)

        # 7) DELETE (where)
        deleted = delete_rows(table_name, where={"id": 3}, returning=True)
        show("Delete id=3 (returning)", deleted)

        # 8) DDL: ADD COLUMN
        add_column(table_name, "age", "integer", nullable=True, default=0)
        print("Colonne 'age' ajoutée.")

        # 9) DDL: RENAME COLUMN name -> full_name
        rename_column(table_name, "name", "full_name")
        print("Colonne 'name' renommée en 'full_name'.")

        # 10) DDL: ALTER COLUMN TYPE (age -> numeric)
        alter_column_type(
            table_name, "age", "numeric(10,2)", using_expression="age::numeric"
        )
        print("Colonne 'age' convertie en numeric(10,2).")

        # 11) DDL: DROP COLUMN
        drop_column(table_name, "age")
        print("Colonne 'age' supprimée.")

        # 12) SELECT final
        final_rows = fetch_rows(table_name, order_by=["id"])
        show("État final", final_rows)

        # 13) SQL brut: count
        count_rows = execute_sql(f'SELECT COUNT(*) AS cnt FROM "{table_name}"')
        show("Count via SQL brut", count_rows)

        print("\nTous les tests ont réussi.")
        return 0

    finally:
        try:
            execute_sql(f'DROP TABLE IF EXISTS "{table_name}"')
            print("Table nettoyée.")
        except Exception as e:
            print("Échec du nettoyage:", e)


# Permet l'exécution avec `python test_db.py`
if __name__ == "__main__":
    sys.exit(main())
