import logging
from src.db.connection import db
from src.models.schema import ColumnInfo, TableSchema, DatabaseSchema

logger = logging.getLogger(__name__)


def discover_schema() -> DatabaseSchema:
    """Introspect PostgreSQL and build full schema object."""
    tables = []
    for table_name in db.get_table_names():
        # Skip shadow tables created by the cleaning pipeline (Issue #15)
        if table_name.endswith("_cleaned"):
            continue
        row_count = db.get_row_count(table_name)
        columns = _get_columns(table_name, row_count)
        pks = _get_primary_keys(table_name)
        for col in columns:
            if col.name in pks:
                col.is_primary_key = True
        tables.append(TableSchema(
            name=table_name,
            columns=columns,
            row_count=row_count,
        ))

    fks = _get_foreign_keys()
    return DatabaseSchema(tables=tables, foreign_keys=fks)


def check_referential_integrity(fks: list[dict]) -> list[dict]:
    """Check for orphaned foreign key references.
    Returns list of FK relationships with orphan counts."""
    issues = []
    for fk in fks:
        try:
            with db.cursor() as cur:
                cur.execute(
                    f'SELECT COUNT(*) AS orphans FROM "{fk["from_table"]}" ft '
                    f'LEFT JOIN "{fk["to_table"]}" tt '
                    f'ON ft."{fk["from_col"]}" = tt."{fk["to_col"]}" '
                    f'WHERE tt."{fk["to_col"]}" IS NULL '
                    f'AND ft."{fk["from_col"]}" IS NOT NULL'
                )
                row = cur.fetchone()
                orphans = row["orphans"] if row else 0
                if orphans > 0:
                    issues.append({
                        "from_table": fk["from_table"],
                        "from_col": fk["from_col"],
                        "to_table": fk["to_table"],
                        "to_col": fk["to_col"],
                        "orphan_count": orphans,
                    })
        except Exception as e:
            logger.debug("Referential integrity check failed for %s: %s", fk, e)
    return issues


def _get_columns(table_name: str, row_count: int) -> list[ColumnInfo]:
    with db.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "ORDER BY ordinal_position",
            (table_name,),
        )
        cols_raw = cur.fetchall()

        columns = []
        for row in cols_raw:
            name = row["column_name"]
            dtype = row["data_type"]
            nullable = row["is_nullable"] == "YES"

            # Use TABLESAMPLE for large tables to keep value sampling fast
            if row_count > 1_000_000:
                sample_sql = (
                    f'SELECT DISTINCT "{name}" '
                    f'FROM "{table_name}" TABLESAMPLE SYSTEM(1) '
                    f'WHERE "{name}" IS NOT NULL LIMIT 5'
                )
            else:
                sample_sql = (
                    f'SELECT DISTINCT "{name}" '
                    f'FROM "{table_name}" '
                    f'WHERE "{name}" IS NOT NULL LIMIT 5'
                )

            cur.execute(sample_sql)
            sample_values = [str(r[name]) for r in cur.fetchall()]

            columns.append(ColumnInfo(
                name=name,
                dtype=dtype,
                nullable=nullable,
                sample_values=sample_values,
            ))
    return columns


def _get_primary_keys(table_name: str) -> set[str]:
    with db.cursor() as cur:
        cur.execute(
            "SELECT kcu.column_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            "  AND tc.table_schema = kcu.table_schema "
            "WHERE tc.constraint_type = 'PRIMARY KEY' "
            "  AND tc.table_schema = 'public' "
            "  AND tc.table_name = %s",
            (table_name,),
        )
        return {row["column_name"] for row in cur.fetchall()}


def _get_foreign_keys() -> list[dict]:
    """Extract FK relationships from information_schema."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT "
            "  kcu.table_name AS from_table, "
            "  kcu.column_name AS from_col, "
            "  ccu.table_name AS to_table, "
            "  ccu.column_name AS to_col "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            "  AND tc.table_schema = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu "
            "  ON ccu.constraint_name = tc.constraint_name "
            "  AND ccu.table_schema = tc.table_schema "
            "WHERE tc.constraint_type = 'FOREIGN KEY' "
            "  AND tc.table_schema = 'public'"
        )
        return [dict(row) for row in cur.fetchall()]
