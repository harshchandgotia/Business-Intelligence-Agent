import re
import threading
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import pandas as pd
from config.settings import settings


class PostgreSQLManager:
    """Thread-safe PostgreSQL connection pool manager."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._pool = None
        return cls._instance

    def _get_pool(self):
        if self._pool is None:
            self._pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=settings.db_url,
            )
        return self._pool

    @contextmanager
    def cursor(self):
        """Yield a psycopg2 cursor from the pool, with auto commit/rollback."""
        conn = self._get_pool().getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._get_pool().putconn(conn)

    def execute(self, sql: str, params=None) -> list[dict]:
        """Execute SQL, return list of row dicts. Enforces row limit."""
        sql_stripped = sql.strip().rstrip(";")
        sql_upper = sql_stripped.upper()

        # Check if query already has a trailing LIMIT clause
        has_limit = bool(re.search(r"\bLIMIT\s+\d+\s*$", sql_upper))

        if has_limit:
            limited_sql = sql_stripped
        elif sql_upper.lstrip().startswith("WITH"):
            # CTE: cannot wrap in subquery, append LIMIT instead
            limited_sql = f"{sql_stripped} LIMIT {settings.MAX_QUERY_ROWS}"
        else:
            # Simple query: wrap in subquery
            limited_sql = (
                f"SELECT * FROM ({sql_stripped}) sub LIMIT {settings.MAX_QUERY_ROWS}"
            )

        with self.cursor() as cur:
            cur.execute(limited_sql, params or [])
            return [dict(row) for row in cur.fetchall()]

    def execute_df(self, sql: str, params=None) -> pd.DataFrame:
        """Execute SQL, return a DataFrame."""
        with self.cursor() as cur:
            cur.execute(sql, params or [])
            rows = cur.fetchall()
            return pd.DataFrame([dict(r) for r in rows])

    def get_table_names(self) -> list[str]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            )
            return [row["table_name"] for row in cur.fetchall()]

    def get_row_count(self, table: str) -> int:
        with self.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) AS cnt FROM "{table}"')
            return cur.fetchone()["cnt"]

    def close(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None


# Singleton
db = PostgreSQLManager()
