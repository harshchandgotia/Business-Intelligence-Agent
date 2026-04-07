"""Tests for SQLAgent — all LLM calls are mocked."""
import json
import pytest
from unittest.mock import MagicMock, patch
from src.agents.sql_agent import SQLAgent
from src.models.schema import DatabaseSchema


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_agent(llm_response: str) -> SQLAgent:
    agent = SQLAgent.__new__(SQLAgent)
    agent._llm = MagicMock()
    agent._llm.generate.return_value = llm_response
    agent._llm.count_tokens.return_value = 50
    agent._start_time = None
    return agent


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------

def test_generates_valid_sql(sample_schema):
    payload = json.dumps({
        "sql": 'SELECT SUM("sale_amount") AS total FROM "transactions"',
        "reasoning": "Simple aggregate",
    })
    agent = make_agent(payload)
    result = agent.run(
        question="total revenue",
        schema=sample_schema,
        sub_question_id=0,
        conversation_context="",
    )
    assert "error" not in result
    assert "sql" in result
    assert "SUM" in result["sql"].upper()


def test_system_prompt_blocks_destructive_keywords(sample_schema):
    """Schema validator must reject any SQL with DELETE/DROP etc."""
    from src.verification.schema_validator import SchemaValidator
    validator = SchemaValidator()

    dangerous_sql = 'DELETE FROM "transactions" WHERE sale_amount < 0'
    valid, errors = validator.validate(dangerous_sql, sample_schema)
    assert not valid
    assert any("DELETE" in e or "forbidden" in e.lower() for e in errors)


def test_forbidden_keywords_blocked(sample_schema):
    """DROP TABLE must be caught by schema validator."""
    from src.verification.schema_validator import SchemaValidator
    validator = SchemaValidator()

    sql = 'DROP TABLE "transactions"'
    valid, errors = validator.validate(sql, sample_schema)
    assert not valid


def test_unknown_table_flagged(sample_schema):
    """Referencing a non-existent table must produce a validation error."""
    from src.verification.schema_validator import SchemaValidator
    validator = SchemaValidator()

    sql = 'SELECT * FROM "nonexistent_table"'
    valid, errors = validator.validate(sql, sample_schema)
    assert not valid
    assert any("nonexistent_table" in e for e in errors)


def test_valid_sql_passes_validation(sample_schema):
    from src.verification.schema_validator import SchemaValidator
    validator = SchemaValidator()

    sql = 'SELECT SUM("sale_amount") AS total FROM "transactions"'
    valid, errors = validator.validate(sql, sample_schema)
    assert valid
    assert errors == []


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def test_retry_fires_on_execution_error(sample_schema):
    """execute_with_retry should call _retry when DB raises a recoverable error."""
    first_sql = 'SELECT SUM("revenu") FROM "transactions"'  # typo
    retry_sql = 'SELECT SUM("sale_amount") AS total FROM "transactions"'

    agent = SQLAgent.__new__(SQLAgent)
    agent._start_time = None

    call_count = {"n": 0}

    def fake_generate(prompt, system=None, json_mode=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({"sql": first_sql, "reasoning": "first attempt"})
        return json.dumps({"sql": retry_sql, "reasoning": "fixed"})

    agent._llm = MagicMock()
    agent._llm.generate.side_effect = fake_generate
    agent._llm.count_tokens.return_value = 50

    with patch("src.verification.schema_validator.SchemaValidator") as MockValidator, \
         patch("src.db.connection.PostgreSQLManager.execute") as mock_execute:

        # Validator: first SQL fails (unknown column), retry SQL passes
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [
            (False, ["column 'revenu' does not exist"]),  # first attempt
            (True, []),                                    # retry attempt
        ]
        MockValidator.return_value = mock_validator

        # DB executes successfully on retry
        mock_execute.return_value = [{"total": 5000000.0}]

        result = agent.execute_with_retry(
            question="total revenue",
            schema=sample_schema,
            sub_question_id=0,
            conversation_context="",
        )

    assert "rows" in result
    assert result["retry_count"] >= 1


def test_max_retries_returns_error(sample_schema):
    """After max retries, a structured error dict is returned (no crash)."""
    bad_sql = 'SELECT * FROM "bogus_table"'

    agent = SQLAgent.__new__(SQLAgent)
    agent._start_time = None
    agent._llm = MagicMock()
    agent._llm.generate.return_value = json.dumps({"sql": bad_sql, "reasoning": ""})
    agent._llm.count_tokens.return_value = 50

    with patch("src.verification.schema_validator.SchemaValidator") as MockValidator:

        mock_validator = MagicMock()
        # Always fail validation
        mock_validator.validate.return_value = (False, ["Unknown table: bogus_table"])
        MockValidator.return_value = mock_validator

        result = agent.execute_with_retry(
            question="whatever",
            schema=sample_schema,
            sub_question_id=0,
        )

    assert "error" in result
