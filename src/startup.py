"""App startup: connect to PostgreSQL, seed if empty, discover schema, build health cards."""
import logging
from src.db.connection import db
from src.ingestion.schema_discovery import discover_schema, check_referential_integrity
from src.preprocessing.profiler import DataProfiler
from src.models.schema import DatabaseSchema
from src.models.health import DataHealthCard

logger = logging.getLogger(__name__)


def initialize() -> tuple[DatabaseSchema, dict[str, DataHealthCard]]:
    """
    Run on app startup.
    Returns (schema, health_cards_by_table).
    Seeds the database if no tables are found.
    """
    from config.settings import settings
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a key at https://console.groq.com"
        )
    _ensure_seeded()
    schema = discover_schema()
    health_cards = _build_health_cards(schema)
    logger.info(
        "Startup complete: %d tables discovered", len(schema.tables)
    )
    return schema, health_cards


def _ensure_seeded():
    tables = db.get_table_names()
    if not tables:
        logger.info("No tables found — running seed...")
        try:
            from data.seed import generate
            generate()
        except Exception as e:
            logger.error("Seed failed: %s", e)
            raise RuntimeError(
                "Database is empty and seeding failed. "
                "Run `make seed` manually, then restart."
            ) from e


def _build_health_cards(schema: DatabaseSchema) -> dict[str, DataHealthCard]:
    profiler = DataProfiler()
    cards = {}

    # Check referential integrity across FK relationships
    ref_issues = check_referential_integrity(schema.foreign_keys)

    for table in schema.tables:
        try:
            card = profiler.profile_table(table.name)
            # Attach referential integrity issues for this table
            table_ref_issues = [
                ri for ri in ref_issues if ri["from_table"] == table.name
            ]
            if table_ref_issues:
                card.referential_issues = table_ref_issues
                for ri in table_ref_issues:
                    card.warnings.append(
                        f"Referential integrity: {ri['orphan_count']} orphaned rows "
                        f"in {ri['from_table']}.{ri['from_col']} → {ri['to_table']}.{ri['to_col']}"
                    )
            cards[table.name] = card
        except Exception as e:
            logger.warning("Could not profile table %s: %s", table.name, e)
    return cards
