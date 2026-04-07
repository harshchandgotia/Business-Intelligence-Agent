from pydantic import BaseModel
from typing import Optional


class ColumnInfo(BaseModel):
    name: str
    dtype: str                  # e.g. "VARCHAR", "INTEGER", "DATE"
    nullable: bool
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None  # "table.column" if FK
    sample_values: list[str] = []     # 5 sample values as strings
    description: Optional[str] = None # human-readable description


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnInfo]
    row_count: int
    description: Optional[str] = None


class DatabaseSchema(BaseModel):
    tables: list[TableSchema]
    foreign_keys: list[dict]    # [{from_table, from_col, to_table, to_col}]

    def to_prompt_string(self) -> str:
        """Serialize schema for LLM prompt injection."""
        lines = []
        for t in self.tables:
            cols = ", ".join(
                f"{c.name} ({c.dtype}{'  PK' if c.is_primary_key else ''}"
                f"{'  FK->' + c.references if c.is_foreign_key else ''})"
                for c in t.columns
            )
            lines.append(f"TABLE {t.name} ({t.row_count} rows): {cols}")
        return "\n".join(lines)
