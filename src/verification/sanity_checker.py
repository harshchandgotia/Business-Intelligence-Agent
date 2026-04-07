import pandas as pd
from src.models.health import DataHealthCard
from config.settings import settings


class SanityChecker:
    """Statistical sanity checks on query results. No LLM."""

    def check(
        self,
        data: list[dict],
        health_card: DataHealthCard | None = None,
    ) -> tuple[bool, list[str], bool]:
        """
        Returns: (passed, warnings, needs_preprocessing)
        """
        warnings = []
        needs_preprocessing = False

        if not data:
            warnings.append("Query returned 0 rows")
            return False, warnings, False

        df = pd.DataFrame(data)

        # Check 1: Row count
        if len(df) >= settings.MAX_QUERY_ROWS:
            warnings.append(
                f"Results hit row limit ({settings.MAX_QUERY_ROWS}). "
                "Data may be truncated."
            )

        # Check 2: NULL density
        for col in df.columns:
            null_pct = df[col].isna().mean()
            if null_pct >= 0.5:
                warnings.append(
                    f"Column '{col}' is {null_pct:.0%} NULL — results may be unreliable"
                )
                needs_preprocessing = True

        # Check 3: Single-value columns (useless)
        for col in df.columns:
            if df[col].nunique() <= 1 and len(df) > 1:
                warnings.append(
                    f"Column '{col}' has only 1 unique value — may not be useful"
                )

        # Check 4: Inconsistent string values (triggers preprocessing)
        for col in df.select_dtypes(include=["object"]).columns:
            unique = df[col].dropna().unique()
            if len(unique) > 2:
                # Quick check: are there near-duplicates?
                lower_unique = set(str(v).lower().strip() for v in unique)
                if len(lower_unique) < len(unique) * 0.8:
                    warnings.append(
                        f"Column '{col}' has inconsistent values "
                        f"(e.g. case variations)"
                    )
                    needs_preprocessing = True

        # Check 5: Negative values in likely-positive columns
        for col in df.select_dtypes(include=["number"]).columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["amount", "price", "revenue", "count", "quantity"]):
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    warnings.append(
                        f"Column '{col}' has {neg_count} negative values — "
                        f"may include returns or data errors"
                    )

        passed = len(warnings) == 0
        return passed, warnings, needs_preprocessing
