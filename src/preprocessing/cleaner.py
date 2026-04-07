import pandas as pd
from sqlalchemy import create_engine, text
from src.db.connection import db
from src.models.cleaning import CleaningPlan, CleaningAction, CleaningActionType, CleaningLog
from config.settings import settings

_LARGE_TABLE_THRESHOLD = 100_000

# Module-level engine singleton to avoid creating a new pool on every call
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db_url)
    return _engine


class CleaningExecutor:
    """Executes cleaning actions. Never modifies original data."""

    def execute(self, plan: CleaningPlan) -> CleaningLog:
        row_count = db.get_row_count(plan.table_name)
        cleaned_name = f"{plan.table_name}_cleaned"

        if row_count <= _LARGE_TABLE_THRESHOLD:
            return self._execute_pandas(plan, cleaned_name)
        else:
            return self._execute_sql(plan, cleaned_name, row_count)

    # ------------------------------------------------------------------
    # Small tables: pandas round-trip
    # ------------------------------------------------------------------

    def _execute_pandas(self, plan: CleaningPlan, cleaned_name: str) -> CleaningLog:
        engine = _get_engine()
        df = pd.read_sql(f'SELECT * FROM "{plan.table_name}"', engine)
        rows_before = len(df)
        applied = []

        for action in plan.actions:
            df, success = self._apply_action(df, action)
            if success:
                applied.append(action)

        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{cleaned_name}"'))
        df.to_sql(cleaned_name, engine, if_exists="replace", index=False, chunksize=10000)

        return CleaningLog(
            table_name=plan.table_name,
            original_view=plan.table_name,
            cleaned_view=cleaned_name,
            actions_applied=applied,
            rows_before=rows_before,
            rows_after=len(df),
        )

    # ------------------------------------------------------------------
    # Large tables: pure SQL transformations
    # ------------------------------------------------------------------

    def _execute_sql(self, plan: CleaningPlan, cleaned_name: str, rows_before: int) -> CleaningLog:
        engine = _get_engine()
        applied = []

        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{cleaned_name}"'))
            conn.execute(text(
                f'CREATE TABLE "{cleaned_name}" AS SELECT * FROM "{plan.table_name}"'
            ))

            for action in plan.actions:
                try:
                    statements = self._action_to_sql(action, cleaned_name)
                    for sql_template, params in statements:
                        conn.execute(text(sql_template), params)
                    applied.append(action)
                except Exception:
                    pass  # skip failed actions, log silently

        rows_after = db.get_row_count(cleaned_name)

        return CleaningLog(
            table_name=plan.table_name,
            original_view=plan.table_name,
            cleaned_view=cleaned_name,
            actions_applied=applied,
            rows_before=rows_before,
            rows_after=rows_after,
        )

    def _action_to_sql(
        self, action: CleaningAction, table: str
    ) -> list[tuple[str, dict]]:
        """Return a list of (sql_template, params) tuples using parameterized queries."""
        col = action.target_column

        match action.action_type:

            case CleaningActionType.STANDARDIZE_VALUES:
                mapping = action.params.get("mapping", {})
                if not mapping:
                    return []
                # One parameterized UPDATE per mapping entry to avoid SQL injection
                statements = []
                for i, (old_val, new_val) in enumerate(mapping.items()):
                    statements.append((
                        f'UPDATE "{table}" SET "{col}" = :new_{i} WHERE "{col}" = :old_{i}',
                        {f"old_{i}": old_val, f"new_{i}": new_val},
                    ))
                return statements

            case CleaningActionType.FILL_NULLS:
                strategy = action.params.get("strategy", "mode")
                if strategy == "value":
                    val = action.params.get("value", "")
                    return [(
                        f'UPDATE "{table}" SET "{col}" = :fill_value WHERE "{col}" IS NULL',
                        {"fill_value": val},
                    )]
                elif strategy == "median":
                    return [(
                        f'UPDATE "{table}" SET "{col}" = '
                        f'(SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{col}") FROM "{table}") '
                        f'WHERE "{col}" IS NULL',
                        {},
                    )]
                else:  # mode
                    return [(
                        f'UPDATE "{table}" SET "{col}" = '
                        f'(SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL '
                        f'GROUP BY "{col}" ORDER BY COUNT(*) DESC LIMIT 1) '
                        f'WHERE "{col}" IS NULL',
                        {},
                    )]

            case CleaningActionType.REMOVE_DUPLICATES:
                # Get all columns from the table for dedup grouping
                with db.cursor() as cur:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = %s "
                        "ORDER BY ordinal_position",
                        (table,),
                    )
                    all_cols = [row['column_name'] for row in cur.fetchall()]
                if not all_cols:
                    return []
                cols = ", ".join('"' + c + '"' for c in all_cols)
                return [(
                    f'DELETE FROM "{table}" WHERE ctid NOT IN ('
                    f'SELECT MIN(ctid) FROM "{table}" '
                    f'GROUP BY {cols})',
                    {},
                )]

            case CleaningActionType.RENAME_COLUMN:
                new_name = action.params.get("new_name")
                if not new_name or not col:
                    return []
                return [(
                    f'ALTER TABLE "{table}" RENAME COLUMN "{col}" TO "{new_name}"',
                    {},
                )]

            case _:
                return []

    # ------------------------------------------------------------------
    # Pandas action dispatch (small table path)
    # ------------------------------------------------------------------

    def _apply_action(
        self, df: pd.DataFrame, action: CleaningAction
    ) -> tuple[pd.DataFrame, bool]:
        try:
            match action.action_type:

                case CleaningActionType.STANDARDIZE_VALUES:
                    mapping = action.params.get("mapping", {})
                    df[action.target_column] = (
                        df[action.target_column].astype(str).replace(mapping)
                    )

                case CleaningActionType.FILL_NULLS:
                    strategy = action.params.get("strategy", "mode")
                    col = action.target_column
                    if strategy == "mode":
                        df[col] = df[col].fillna(df[col].mode().iloc[0])
                    elif strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "value":
                        df[col] = df[col].fillna(action.params["value"])

                case CleaningActionType.REMOVE_DUPLICATES:
                    df = df.drop_duplicates()

                case CleaningActionType.FIX_TYPES:
                    target_type = action.params.get("target_type", "str")
                    col = action.target_column
                    try:
                        if target_type in ("int", "int64", "float", "float64"):
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            if target_type.startswith("int"):
                                df[col] = df[col].fillna(0).astype(int)
                        elif target_type in ("datetime", "date"):
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                        elif target_type == "str":
                            df[col] = df[col].astype(str)
                        else:
                            df[col] = df[col].astype(target_type)
                    except (ValueError, TypeError):
                        pass  # conversion failed, leave column unchanged

                case CleaningActionType.SEPARATE_COLUMN:
                    new_col = action.params.get("new_column", "flag")
                    threshold = action.params.get("threshold", 0)
                    col = action.target_column
                    # Safe: only supports numeric threshold comparison
                    df[new_col] = df[col].apply(
                        lambda x: x < threshold if isinstance(x, (int, float)) else False
                    )

                case CleaningActionType.RENAME_COLUMN:
                    new_name = action.params.get("new_name")
                    if new_name and action.target_column in df.columns:
                        df = df.rename(columns={action.target_column: new_name})

            return df, True
        except Exception:
            return df, False
