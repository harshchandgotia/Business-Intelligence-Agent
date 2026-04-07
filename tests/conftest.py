"""Shared test fixtures — zero real API calls, zero real DB connections."""
import pytest
from unittest.mock import MagicMock, patch
from src.models.schema import DatabaseSchema, TableSchema, ColumnInfo
from src.models.health import DataHealthCard, ColumnProfile


# ---------------------------------------------------------------------------
# Sample schema
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_schema() -> DatabaseSchema:
    products = TableSchema(
        name="products",
        row_count=200,
        columns=[
            ColumnInfo(name="product_id", dtype="integer", nullable=False, is_primary_key=True),
            ColumnInfo(name="product_name", dtype="character varying", nullable=False),
            ColumnInfo(name="brand", dtype="character varying", nullable=True,
                       sample_values=["Zara", "Nike", "H&M"]),
            ColumnInfo(name="category", dtype="character varying", nullable=True,
                       sample_values=["Tops", "Bottoms", "Dresses"]),
            ColumnInfo(name="color", dtype="character varying", nullable=True,
                       sample_values=["Black", "White", "Blue"]),
            ColumnInfo(name="size", dtype="character varying", nullable=True,
                       sample_values=["S", "M", "L"]),
            ColumnInfo(name="base_price", dtype="numeric", nullable=True),
        ],
    )
    transactions = TableSchema(
        name="transactions",
        row_count=510200,
        columns=[
            ColumnInfo(name="transaction_id", dtype="integer", nullable=False, is_primary_key=True),
            ColumnInfo(name="product_id", dtype="integer", nullable=True,
                       is_foreign_key=True, references="products.product_id"),
            ColumnInfo(name="sale_date", dtype="date", nullable=True),
            ColumnInfo(name="quantity", dtype="integer", nullable=True),
            ColumnInfo(name="unit_price", dtype="numeric", nullable=True),
            ColumnInfo(name="sale_amount", dtype="numeric", nullable=True),
            ColumnInfo(name="purchase_amount", dtype="numeric", nullable=True),
        ],
    )
    return DatabaseSchema(
        tables=[products, transactions],
        foreign_keys=[{
            "from_table": "transactions",
            "from_col": "product_id",
            "to_table": "products",
            "to_col": "product_id",
        }],
    )


# ---------------------------------------------------------------------------
# Sample health card
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_health_card() -> DataHealthCard:
    return DataHealthCard(
        table_name="transactions",
        row_count=510200,
        column_count=7,
        duplicate_row_count=10200,
        overall_quality_score=88.0,
        columns=[
            ColumnProfile(
                name="sale_date",
                dtype="date",
                null_count=5102,
                null_pct=0.01,
                unique_count=2000,
                outlier_count=0,
            ),
            ColumnProfile(
                name="quantity",
                dtype="integer",
                null_count=15306,
                null_pct=0.03,
                unique_count=50,
                outlier_count=120,
            ),
        ],
        warnings=["3% null quantity values", "Inconsistent color casing detected"],
    )


# ---------------------------------------------------------------------------
# Mock LLM — returns canned JSON so no API calls hit during tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.generate.return_value = (
        '{"sql": "SELECT SUM(\\"sale_amount\\") AS total FROM \\"transactions\\"", '
        '"reasoning": "Sum all sale amounts"}'
    )
    client.count_tokens.return_value = 50
    return client


@pytest.fixture
def patch_llm(mock_llm):
    """Patch get_llm() everywhere to return the mock."""
    with patch("src.llm.factory.get_llm", return_value=mock_llm), \
         patch("src.agents.base.get_llm", return_value=mock_llm):
        yield mock_llm


# ---------------------------------------------------------------------------
# Sample SQL results
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sql_rows():
    return [
        {"brand": "Zara", "total_revenue": 1500000.0},
        {"brand": "Nike", "total_revenue": 1200000.0},
        {"brand": "H&M", "total_revenue": 900000.0},
    ]
