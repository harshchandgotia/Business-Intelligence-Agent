import json
import logging
from src.llm.json_utils import extract_json
from src.agents.base import BaseAgent
from src.agents.value_resolver import resolve_values
from src.models.schema import DatabaseSchema
from src.models.query import SQLResult
from config.settings import settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RECOVERABLE_ERRORS = (
    "does not exist",
    "column",
    "relation",
    "syntax error",
    "type",
    "operator does not exist",
    "ambiguous",
)

_SYSTEM_PROMPT = (
    "You are a SQL expert. Generate a single PostgreSQL-compatible SQL query.\n"
    "Respond ONLY with a JSON object: {\"sql\": \"...\", \"reasoning\": \"...\"}\n"
    "Rules:\n"
    "- Use only tables and columns from the schema provided\n"
    "- Always use double quotes around identifiers\n"
    "- Never use DELETE, UPDATE, INSERT, DROP, ALTER, CREATE\n"
    "- For aggregations, always include GROUP BY\n"
    "- Limit results to 1000 rows maximum"
)


class SQLAgent(BaseAgent):
    name = "sql_agent"

    def _execute(
        self,
        question: str,
        schema: DatabaseSchema,
        sub_question_id: int = 0,
        conversation_context: str = "",
    ) -> dict:
        # Fuzzy value resolution — catch misspellings before SQL generation
        value_hints = resolve_values(question, schema)
        extra_context = conversation_context
        if value_hints:
            extra_context += "\nValue corrections: " + "; ".join(value_hints)
            logger.info("Value resolver hints: %s", value_hints)

        # Schema linking — filter to relevant tables/columns
        filtered_schema = _filter_relevant_schema(question, schema)

        prompt = self._load_prompt("sql_generation.txt").format(
            schema=filtered_schema.to_prompt_string(),
            question=question,
            context=extra_context,
        )

        response = self._llm.generate(prompt, system=_SYSTEM_PROMPT, json_mode=True)
        parsed = extract_json(response, fallback={"sql": "SELECT 1", "reasoning": "JSON parse failed"})

        return {
            "sql": parsed["sql"],
            "reasoning": parsed.get("reasoning", ""),
            "sub_question_id": sub_question_id,
            "retry_count": 0,
        }

    def execute_with_retry(
        self,
        question: str,
        schema: DatabaseSchema,
        sub_question_id: int = 0,
        conversation_context: str = "",
    ) -> dict:
        """Generate SQL, then retry up to _MAX_RETRIES times on recoverable errors."""
        from src.verification.schema_validator import SchemaValidator
        from src.db.connection import db

        result = self.run(
            question=question,
            schema=schema,
            sub_question_id=sub_question_id,
            conversation_context=conversation_context,
        )

        if "error" in result:
            return result

        validator = SchemaValidator()
        retries = 0

        for attempt in range(_MAX_RETRIES + 1):
            sql = result.get("sql", "")
            valid, validation_errors = validator.validate(sql, schema)

            if valid:
                try:
                    rows = db.execute(sql)
                    return {
                        "sql": sql,
                        "reasoning": result.get("reasoning", ""),
                        "sub_question_id": sub_question_id,
                        "retry_count": attempt,
                        "rows": rows,
                    }
                except Exception as e:
                    error_msg = str(e)
                    if attempt == _MAX_RETRIES or not _is_recoverable(error_msg):
                        return {
                            "sql": sql,
                            "error": error_msg,
                            "retry_count": attempt,
                            "sub_question_id": sub_question_id,
                        }
                    logger.warning("SQL execution error (retry %d): %s", attempt + 1, error_msg)
                    result = self._retry(question, schema, sql, error_msg, sub_question_id)
            else:
                error_msg = "; ".join(validation_errors)
                if attempt == _MAX_RETRIES:
                    return {
                        "sql": sql,
                        "error": f"Schema validation failed after {attempt+1} attempts: {error_msg}",
                        "retry_count": attempt,
                        "sub_question_id": sub_question_id,
                    }
                logger.warning("Schema validation failed (retry %d): %s", attempt + 1, error_msg)
                result = self._retry(question, schema, sql, error_msg, sub_question_id)

        return {"sql": result.get("sql", ""), "error": "Max retries exceeded", "sub_question_id": sub_question_id}

    def _retry(
        self,
        question: str,
        schema: DatabaseSchema,
        failed_sql: str,
        error: str,
        sub_question_id: int,
    ) -> dict:
        prompt = self._load_prompt("sql_retry.txt").format(
            schema=schema.to_prompt_string(),
            question=question,
            failed_sql=failed_sql,
            error=error,
        )
        # Use light model for retries — lower cost, sufficient quality
        response = self._llm.generate(
            prompt, system=_SYSTEM_PROMPT, json_mode=True,
            model=settings.GROQ_MODEL_LIGHT,
        )
        parsed = extract_json(response, fallback={"sql": "SELECT 1", "reasoning": "JSON parse failed on retry"})
        return {
            "sql": parsed["sql"],
            "reasoning": parsed.get("reasoning", ""),
            "sub_question_id": sub_question_id,
        }


def _is_recoverable(error: str) -> bool:
    error_lower = error.lower()
    return any(kw in error_lower for kw in _RECOVERABLE_ERRORS)


def _filter_relevant_schema(question: str, schema: DatabaseSchema) -> DatabaseSchema:
    """Schema linking: keep only tables/columns that share vocabulary with the question.
    Falls back to full schema if no matches found."""
    q_words = set(question.lower().split())
    q_lower = question.lower()

    relevant_tables = []
    for t in schema.tables:
        table_words = {t.name.lower()} | {c.name.lower() for c in t.columns}
        # Match on table/column names or sample values
        name_match = bool(q_words & table_words)
        value_match = any(
            v.lower() in q_lower
            for c in t.columns
            for v in c.sample_values
        )
        if name_match or value_match:
            relevant_tables.append(t)

    # Fall back to full schema if nothing matched
    if not relevant_tables:
        return schema

    return DatabaseSchema(
        tables=relevant_tables,
        foreign_keys=[
            fk for fk in schema.foreign_keys
            if fk["from_table"] in {t.name for t in relevant_tables}
            or fk["to_table"] in {t.name for t in relevant_tables}
        ],
    )
