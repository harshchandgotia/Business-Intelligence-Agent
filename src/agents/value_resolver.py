"""Fuzzy value resolution for SQL generation.

When a user misspells a filter value (e.g. "Zaara" instead of "Zara"),
this module resolves it against actual database values using fuzzy matching.
"""
import logging
import re
from fuzzywuzzy import process
from src.db.connection import db
from config.settings import settings

logger = logging.getLogger(__name__)


def resolve_values(question: str, schema) -> list[str]:
    """Extract potential filter values from the question and resolve them
    against actual database values. Returns a list of correction hints.

    Example return: ["'Zaara' likely refers to 'Zara' in column brand (table products)"]
    """
    hints = []
    # Extract quoted strings and capitalized words that look like entity names
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
    # Also grab capitalized multi-word names not in common English
    capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
    candidates = list(set(quoted + capitalized))

    if not candidates:
        return hints

    for table in schema.tables:
        # Only check string/text columns with reasonable cardinality
        text_cols = [
            c for c in table.columns
            if c.dtype.lower() in (
                "character varying", "varchar", "text", "name",
            )
        ]
        for col in text_cols:
            try:
                rows = db.execute(
                    f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                    f'WHERE "{col.name}" IS NOT NULL LIMIT 200'
                )
                db_values = [str(r[col.name]) for r in rows if r[col.name]]
                if not db_values:
                    continue

                for candidate in candidates:
                    # Skip if exact match exists
                    if candidate in db_values:
                        continue
                    match, score = process.extractOne(candidate, db_values)
                    if score >= settings.FUZZY_MATCH_THRESHOLD and match != candidate:
                        hints.append(
                            f"'{candidate}' likely refers to '{match}' "
                            f"in column {col.name} (table {table.name})"
                        )
            except Exception as e:
                logger.debug("Value resolution failed for %s.%s: %s", table.name, col.name, e)

    return hints
