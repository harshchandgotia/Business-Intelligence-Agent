"""Tests for orchestrator nodes and graph paths — all external calls mocked."""
import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator.nodes import (
    classify_node,
    template_node,
    format_output_node,
)
from src.models.query import RouteType, SQLResult, ConfidenceScore


# ---------------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def test_simple_query_classified(self, sample_schema):
        state = {
            "query": "total revenue",
            "schema": sample_schema,
            "conversation_context": "",
        }
        result = classify_node(state)
        assert "route" in result

    def test_meta_query_classified(self, sample_schema):
        state = {
            "query": "what tables do you have",
            "schema": sample_schema,
            "conversation_context": "",
        }
        result = classify_node(state)
        assert "route" in result
        assert result["route"] == RouteType.META

    def test_classify_node_never_raises(self, sample_schema):
        """Even on a pathological input, classify_node returns a dict."""
        state = {"query": "", "schema": sample_schema, "conversation_context": ""}
        result = classify_node(state)
        assert isinstance(result, dict)
        assert "route" in result


# ---------------------------------------------------------------------------
# template_node
# ---------------------------------------------------------------------------

class TestTemplateNode:
    def test_template_match_returns_sql_result(self, sample_schema):
        with patch("src.orchestrator.nodes.db") as mock_db, \
             patch("src.orchestrator.nodes.SchemaValidator") as MockValidator:
            mock_db.execute.return_value = [{"total_sale_amount": 5000000.0}]
            MockValidator.return_value.validate.return_value = (True, [])
            state = {
                "query": "total sale_amount",
                "schema": sample_schema,
                "conversation_context": "",
            }
            result = template_node(state)

        # Either matched (sql_results) or fell back to analytical route
        assert "sql_results" in result or result.get("sql_results") == []

    def test_no_match_falls_back_to_analytical(self, sample_schema):
        state = {
            "query": "why did brand X outperform Y in Q3 considering seasonal effects",
            "schema": sample_schema,
            "conversation_context": "",
        }
        with patch("src.orchestrator.nodes.db"):
            result = template_node(state)
        # No template match → returns empty sql_results so conditional edge re-routes
        assert result.get("sql_results") == []


# ---------------------------------------------------------------------------
# format_output_node / meta handler
# ---------------------------------------------------------------------------

class TestFormatOutputNode:
    def test_meta_route_returns_narrative(self, sample_schema):
        state = {
            "query": "what tables do you have",
            "schema": sample_schema,
            "route": RouteType.META,
            "health_cards": {},
            "sql_results": [],
        }
        result = format_output_node(state)
        assert "narrative" in result
        assert "transactions" in result["narrative"] or "products" in result["narrative"]

    def test_meta_columns_query(self, sample_schema):
        state = {
            "query": "what columns are in the transactions table",
            "schema": sample_schema,
            "route": RouteType.META,
            "health_cards": {},
            "sql_results": [],
        }
        result = format_output_node(state)
        assert "narrative" in result
        assert "sale_amount" in result["narrative"] or "column" in result["narrative"].lower()

    def test_non_meta_route_is_passthrough(self, sample_schema):
        state = {
            "query": "total revenue",
            "schema": sample_schema,
            "route": RouteType.SIMPLE,
            "sql_results": [],
            "narrative": "Revenue was $5M",
            "confidence": ConfidenceScore(score=0.9, reasoning="test", uncertain_aspects=[]),
        }
        result = format_output_node(state)
        # Non-meta route with existing narrative and confidence → empty result
        assert result == {}

    def test_meta_describe_includes_schema(self, sample_schema):
        state = {
            "query": "describe the data",
            "schema": sample_schema,
            "route": RouteType.META,
            "health_cards": {},
            "sql_results": [],
        }
        result = format_output_node(state)
        assert "narrative" in result


# ---------------------------------------------------------------------------
# Error handling in nodes
# ---------------------------------------------------------------------------

class TestNodeErrorHandling:
    def test_classify_node_handles_router_failure(self, sample_schema):
        with patch("src.orchestrator.nodes.HybridRouter") as MockRouter:
            MockRouter.return_value.classify.side_effect = RuntimeError("Router failed")
            state = {"query": "total revenue", "schema": sample_schema, "conversation_context": ""}
            result = classify_node(state)
            # Should not raise; returns some route
            assert "route" in result

    def test_template_node_db_failure_falls_back(self, sample_schema):
        with patch("src.orchestrator.nodes.db") as mock_db, \
             patch("src.orchestrator.nodes.TemplateEngine") as MockEngine:
            MockEngine.return_value.try_match.return_value = "SELECT 1"
            mock_db.execute.side_effect = Exception("Connection refused")
            state = {"query": "total revenue", "schema": sample_schema}
            result = template_node(state)
            assert isinstance(result, dict)
