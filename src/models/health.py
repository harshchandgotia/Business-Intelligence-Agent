from pydantic import BaseModel, Field
from typing import Optional


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    # Numeric columns
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None
    outlier_count: int = 0
    outlier_indices: list[int] = []
    # Categorical columns
    top_values: list[dict] = []        # [{value, count, pct}]
    inconsistent_values: list[dict] = []  # [{variants: [...], suggested: "..."}]


class DataHealthCard(BaseModel):
    table_name: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    overall_quality_score: float = Field(ge=0.0, le=100.0)
    columns: list[ColumnProfile]
    temporal_gaps: list[dict] = []     # [{column, gap_start, gap_end}]
    referential_issues: list[dict] = []  # [{column, orphan_count}]
    warnings: list[str] = []
