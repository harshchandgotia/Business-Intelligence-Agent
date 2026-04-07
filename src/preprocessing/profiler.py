import pandas as pd
import numpy as np
from scipy import stats
from fuzzywuzzy import fuzz, process
from src.db.connection import db
from src.models.health import ColumnProfile, DataHealthCard
from config.settings import settings


class DataProfiler:
    """Deterministic data profiling. No LLM calls."""

    def profile_table(self, table_name: str) -> DataHealthCard:
        row_count = db.get_row_count(table_name)

        # For large tables, sample to avoid loading everything into RAM (Issue #15)
        if row_count > 50_000:
            df = db.execute_df(
                f'SELECT * FROM "{table_name}" TABLESAMPLE SYSTEM(10) LIMIT 50000'
            )
        else:
            df = db.execute_df(f'SELECT * FROM "{table_name}"')

        columns = []
        warnings = []

        for col in df.columns:
            profile = self._profile_column(df, col)
            columns.append(profile)

            if profile.null_pct > settings.NULL_WARNING_THRESHOLD:
                warnings.append(
                    f"Column '{col}': {profile.null_pct:.1%} null values"
                )
            if profile.inconsistent_values:
                warnings.append(
                    f"Column '{col}': {len(profile.inconsistent_values)} "
                    f"inconsistent value groups detected"
                )

        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            warnings.append(f"{dup_count} duplicate rows detected")

        # Temporal gap detection for date columns
        temporal_gaps = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
                gaps = self._detect_temporal_gaps(df, col)
                temporal_gaps.extend(gaps)
                for g in gaps:
                    warnings.append(
                        f"Temporal gap in '{col}': {g['gap_start']} to {g['gap_end']} "
                        f"({g['gap_days']} days)"
                    )

        quality = self._compute_quality_score(columns, dup_count, len(df))

        return DataHealthCard(
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            duplicate_row_count=dup_count,
            overall_quality_score=quality,
            columns=columns,
            temporal_gaps=temporal_gaps,
            warnings=warnings,
        )

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = null_count / len(df) if len(df) > 0 else 0
        unique_count = int(series.nunique())

        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
        )

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                profile.min_val = float(clean.min())
                profile.max_val = float(clean.max())
                profile.mean = float(clean.mean())
                profile.median = float(clean.median())
                profile.std = float(clean.std()) if len(clean) > 1 else 0
                profile.skewness = float(clean.skew()) if len(clean) > 2 else 0

                # Outlier detection (Z-score)
                z = np.abs(stats.zscore(clean, nan_policy="omit"))
                outlier_mask = z > settings.OUTLIER_ZSCORE
                profile.outlier_count = int(outlier_mask.sum())
                profile.outlier_indices = list(
                    clean[outlier_mask].index[:20]  # cap at 20
                )

        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Top values
            value_counts = series.value_counts().head(10)
            profile.top_values = [
                {"value": str(v), "count": int(c), "pct": float(c / len(df))}
                for v, c in value_counts.items()
            ]
            # Fuzzy inconsistency detection
            profile.inconsistent_values = self._detect_inconsistencies(series)

        return profile

    def _detect_temporal_gaps(self, df: pd.DataFrame, col: str) -> list[dict]:
        """Detect large gaps in date/time columns."""
        try:
            series = pd.to_datetime(df[col], errors="coerce").dropna().sort_values()
        except Exception:
            return []
        if len(series) < 10:
            return []
        diffs = series.diff().dropna()
        median_gap = diffs.median()
        if median_gap.total_seconds() == 0:
            return []
        threshold = median_gap * 5  # flag gaps 5x the typical interval
        large_gaps = diffs[diffs > threshold]
        results = []
        for i, g in zip(large_gaps.index, large_gaps):
            pos = series.index.get_loc(i)
            if pos == 0:
                continue
            prev_idx = series.index[pos - 1]
            results.append({
                "column": col,
                "gap_start": str(series.loc[prev_idx].date()),
                "gap_end": str(series.loc[i].date()),
                "gap_days": int(g.days),
            })
        return results[:10]  # cap

    def _detect_inconsistencies(self, series: pd.Series) -> list[dict]:
        """Find values that are likely the same but written differently."""
        unique_vals = series.dropna().unique()
        if len(unique_vals) > 500 or len(unique_vals) < 2:
            return []

        clusters = []
        used = set()

        for val in unique_vals:
            if str(val) in used:
                continue
            matches = process.extract(
                str(val), [str(v) for v in unique_vals],
                scorer=fuzz.ratio, limit=10
            )
            group = [
                m[0] for m in matches
                if m[1] >= settings.FUZZY_MATCH_THRESHOLD
                and m[0] != str(val)
                and m[0] not in used
            ]
            if group:
                all_variants = [str(val)] + group
                used.update(all_variants)
                # Suggest the most common variant
                counts = {v: int((series.astype(str) == v).sum()) for v in all_variants}
                suggested = max(counts, key=counts.get)
                clusters.append({
                    "variants": all_variants,
                    "suggested": suggested,
                    "total_affected": sum(counts.values()),
                })

        return clusters

    def _compute_quality_score(
        self, columns: list[ColumnProfile], dup_count: int, total_rows: int
    ) -> float:
        """0-100 quality score. Higher = cleaner data."""
        if total_rows == 0:
            return 0.0

        penalties = 0.0
        max_penalty = 100.0

        # Null penalty (up to 30 points)
        avg_null_pct = np.mean([c.null_pct for c in columns]) if columns else 0
        penalties += min(avg_null_pct * 100, 30)

        # Duplicate penalty (up to 20 points)
        dup_pct = dup_count / total_rows
        penalties += min(dup_pct * 100, 20)

        # Inconsistency penalty (up to 30 points)
        inconsistent_cols = sum(
            1 for c in columns if c.inconsistent_values
        )
        penalties += min((inconsistent_cols / max(len(columns), 1)) * 50, 30)

        # Outlier penalty (up to 20 points)
        total_outliers = sum(c.outlier_count for c in columns)
        outlier_pct = total_outliers / (total_rows * len(columns)) if columns else 0
        penalties += min(outlier_pct * 200, 20)

        return max(0.0, max_penalty - penalties)
