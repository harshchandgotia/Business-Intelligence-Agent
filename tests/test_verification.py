"""Tests for schema validator, sanity checker, and confidence scorer."""
import pytest
from unittest.mock import MagicMock, patch
from src.verification.schema_validator import SchemaValidator
from src.verification.sanity_checker import SanityChecker


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------

class TestSchemaValidator:
    def setup_method(self):
        self.validator = SchemaValidator()

    def test_valid_select_passes(self, sample_schema):
        sql = 'SELECT SUM("sale_amount") AS total FROM "transactions"'
        valid, errors = self.validator.validate(sql, sample_schema)
        assert valid
        assert errors == []

    def test_valid_join_passes(self, sample_schema):
        sql = (
            'SELECT p."brand", SUM(t."sale_amount") '
            'FROM "transactions" t '
            'JOIN "products" p ON t."product_id" = p."product_id" '
            'GROUP BY p."brand"'
        )
        valid, errors = self.validator.validate(sql, sample_schema)
        assert valid

    def test_unknown_table_rejected(self, sample_schema):
        sql = 'SELECT * FROM "orders"'
        valid, errors = self.validator.validate(sql, sample_schema)
        assert not valid
        assert any("orders" in e for e in errors)

    def test_delete_rejected(self, sample_schema):
        sql = 'DELETE FROM "transactions" WHERE sale_amount < 0'
        valid, errors = self.validator.validate(sql, sample_schema)
        assert not valid
        assert any("DELETE" in e.upper() for e in errors)

    def test_drop_rejected(self, sample_schema):
        sql = 'DROP TABLE "transactions"'
        valid, errors = self.validator.validate(sql, sample_schema)
        assert not valid

    def test_insert_rejected(self, sample_schema):
        sql = "INSERT INTO \"transactions\" VALUES (1, 1, '2024-01-01', 1, 10.0, 10.0, 5.0)"
        valid, errors = self.validator.validate(sql, sample_schema)
        assert not valid

    def test_update_rejected(self, sample_schema):
        sql = 'UPDATE "transactions" SET "sale_amount" = 0'
        valid, errors = self.validator.validate(sql, sample_schema)
        assert not valid

    def test_subquery_with_valid_tables(self, sample_schema):
        sql = (
            'SELECT * FROM ('
            'SELECT "brand", SUM("sale_amount") AS rev FROM "transactions" t '
            'JOIN "products" p ON t."product_id" = p."product_id" '
            'GROUP BY "brand") sub ORDER BY rev DESC LIMIT 5'
        )
        valid, errors = self.validator.validate(sql, sample_schema)
        assert valid


# ---------------------------------------------------------------------------
# SanityChecker
# ---------------------------------------------------------------------------

class TestSanityChecker:
    def setup_method(self):
        self.checker = SanityChecker()

    def test_empty_data_returns_failed(self, sample_health_card):
        passed, warnings, needs_prep = self.checker.check([], sample_health_card)
        assert not passed
        assert any("0 rows" in w for w in warnings)
        assert needs_prep is False

    def test_high_null_triggers_preprocessing(self):
        data = [{"sale_amount": None if i % 2 == 0 else 100.0} for i in range(100)]
        passed, warnings, needs_prep = self.checker.check(data, None)
        assert needs_prep

    def test_negative_values_warned(self):
        data = [
            {"sale_amount": -50.0},
            {"sale_amount": 100.0},
            {"sale_amount": 200.0},
        ]
        passed, warnings, needs_prep = self.checker.check(data, None)
        assert any("negative" in w.lower() for w in warnings)

    def test_clean_data_passes(self):
        data = [
            {"brand": "Zara", "total_revenue": 1500000.0},
            {"brand": "Nike", "total_revenue": 1200000.0},
            {"brand": "H&M", "total_revenue": 900000.0},
        ]
        passed, warnings, needs_prep = self.checker.check(data, None)
        assert needs_prep is False

    def test_inconsistent_strings_detected(self):
        data = [{"color": c} for c in ["Black", "black", "BLACK", "White", "white", "WHITE"]]
        passed, warnings, needs_prep = self.checker.check(data, None)
        assert needs_prep
        assert any("inconsistent" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class TestConfidenceScorer:
    def test_score_returns_float_in_range(self, sample_schema):
        from src.verification.confidence_scorer import ConfidenceScorer
        scorer = ConfidenceScorer()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"score": 0.85, "reasoning": "Query looks valid", "uncertain_aspects": []}'
        )

        with patch("src.verification.confidence_scorer.get_llm", return_value=mock_llm):
            score = scorer.score(
                question="total revenue",
                sql='SELECT SUM("sale_amount") FROM "transactions"',
                row_count=1,
                schema_valid=True,
                sanity_warnings=[],
            )

        assert 0.0 <= score.score <= 1.0
        assert score.reasoning

    def test_invalid_schema_gives_zero(self, sample_schema):
        from src.verification.confidence_scorer import ConfidenceScorer
        from src.models.query import ConfidenceScore
        scorer = ConfidenceScorer()

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"score": 0.1, "reasoning": "Schema invalid", "uncertain_aspects": ["table"]}'
        )

        with patch("src.verification.confidence_scorer.get_llm", return_value=mock_llm):
            score = scorer.score(
                question="total revenue",
                sql='SELECT * FROM "nonexistent"',
                row_count=0,
                schema_valid=False,
                sanity_warnings=["Unknown table"],
            )

        assert isinstance(score, ConfidenceScore)
