import logging
from datetime import datetime, timezone
from src.orchestrator.state import GraphState
from src.routing.router import HybridRouter
from src.agents.template_engine import TemplateEngine
from src.agents.decomposer import DecomposerAgent
from src.agents.sql_agent import SQLAgent
from src.agents.trend_agent import TrendAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.narrative_agent import NarrativeAgent
from src.agents.critique_agent import CritiqueAgent
from src.verification.schema_validator import SchemaValidator
from src.verification.sanity_checker import SanityChecker
from src.verification.confidence_scorer import ConfidenceScorer
from src.preprocessing.profiler import DataProfiler
from src.preprocessing.cleaning_agent import CleaningAgent
from src.preprocessing.cleaner import CleaningExecutor
from src.db.connection import db
from src.models.query import SQLResult, ConfidenceScore, RouteType

logger = logging.getLogger(__name__)

# Module-level cached instances (Issue #12, #16)
_router = None
_template_engine = None
_decomposer = None
_sql_agent = None
_trend_agent = None
_anomaly_agent = None
_narrative_agent = None
_critique_agent = None
_confidence_scorer = None


def _get_router():
    global _router
    if _router is None:
        _router = HybridRouter()
    return _router


def _get_template_engine():
    global _template_engine
    if _template_engine is None:
        _template_engine = TemplateEngine()
    return _template_engine


def _get_decomposer():
    global _decomposer
    if _decomposer is None:
        _decomposer = DecomposerAgent()
    return _decomposer


def _get_sql_agent():
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = SQLAgent()
    return _sql_agent


def _get_trend_agent():
    global _trend_agent
    if _trend_agent is None:
        _trend_agent = TrendAgent()
    return _trend_agent


def _get_anomaly_agent():
    global _anomaly_agent
    if _anomaly_agent is None:
        _anomaly_agent = AnomalyAgent()
    return _anomaly_agent


def _get_narrative_agent():
    global _narrative_agent
    if _narrative_agent is None:
        _narrative_agent = NarrativeAgent()
    return _narrative_agent


def _get_critique_agent():
    global _critique_agent
    if _critique_agent is None:
        _critique_agent = CritiqueAgent()
    return _critique_agent


def _get_confidence_scorer():
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------------

def classify_node(state: GraphState) -> dict:
    try:
        router = _get_router()
        route = router.classify(
            state["query"],
            conversation_context=state.get("conversation_context", ""),
        )
        return {"route": route}
    except Exception as e:
        logger.error("classify_node failed: %s", e)
        return {"route": RouteType.SIMPLE, "error": f"Routing failed: {e}"}


# ---------------------------------------------------------------------------
# template_node
# ---------------------------------------------------------------------------

def template_node(state: GraphState) -> dict:
    try:
        engine = _get_template_engine()
        sql = engine.try_match(state["query"], state["schema"])

        if sql is None:
            # No template matched — return empty results so conditional edge re-routes
            return {"sql_results": []}

        # Validate SQL before executing (Issue #6)
        validator = SchemaValidator()
        valid, errors = validator.validate(sql, state["schema"])
        if not valid:
            logger.warning("Template SQL validation failed: %s", errors)
            return {"sql_results": []}

        rows = db.execute(sql)
        return {
            "sql_results": [SQLResult(
                sub_question_id=0,
                sql=sql,
                columns=list(rows[0].keys()) if rows else [],
                row_count=len(rows),
                data=rows,
                execution_time_ms=0,
            )],
        }
    except Exception as e:
        logger.error("template_node failed: %s", e)
        return {"sql_results": [], "error": f"Template engine failed: {e}"}


# ---------------------------------------------------------------------------
# decompose_node  (activated for complex analytical queries, word count > 15)
# ---------------------------------------------------------------------------

def decompose_node(state: GraphState) -> dict:
    query = state["query"]
    route = state.get("route")

    # Only decompose analytical queries that are complex enough
    word_count = len(query.split())
    if route != RouteType.ANALYTICAL or word_count <= 15:
        # Pass through as a single question
        return {"decomposition": None}

    try:
        agent = _get_decomposer()
        result = agent.run(question=query, schema=state["schema"])
        return {"decomposition": result.get("decomposition")}
    except Exception as e:
        logger.error("decompose_node failed: %s", e)
        # Graceful fallback — continue without decomposition
        return {"decomposition": None}


# ---------------------------------------------------------------------------
# sql_node  (uses retry-aware SQLAgent with dependency resolution)
# ---------------------------------------------------------------------------

def _topological_sort(sub_questions):
    """Sort sub-questions respecting depends_on ordering."""
    by_id = {sq.id: sq for sq in sub_questions}
    visited = set()
    order = []

    def visit(sq_id):
        if sq_id in visited:
            return
        visited.add(sq_id)
        sq = by_id.get(sq_id)
        if sq:
            for dep_id in sq.depends_on:
                visit(dep_id)
            order.append(sq)

    for sq in sub_questions:
        visit(sq.id)
    return order


def _build_dependency_context(sq, executed: dict) -> str:
    """Build context string from previously executed dependent sub-questions."""
    if not sq.depends_on:
        return ""
    parts = []
    for dep_id in sq.depends_on:
        if dep_id in executed:
            dep_result = executed[dep_id]
            if dep_result.data:
                summary = f"Results from sub-question {dep_id}: {dep_result.data[:5]}"
                parts.append(summary)
    return "\n" + "\n".join(parts) if parts else ""


def sql_node(state: GraphState) -> dict:
    try:
        agent = _get_sql_agent()
        decomp = state.get("decomposition")
        results = []
        executed = {}  # sq.id -> SQLResult, for dependency context

        sub_questions = decomp.sub_questions if decomp else []
        if not sub_questions:
            # Single question fallback
            from src.models.query import SubQuestion
            sub_questions = [SubQuestion(id=0, text=state["query"])]

        # Topological sort to respect dependencies
        sorted_sqs = _topological_sort(sub_questions)

        # Inject cleaning context if available
        ctx = state.get("conversation_context", "")
        cleaning_ctx = state.get("cleaning_context")
        if cleaning_ctx:
            ctx += f"\nData was cleaned: {cleaning_ctx}"

        for sq in sorted_sqs:
            dep_context = _build_dependency_context(sq, executed)
            t0 = _now()
            outcome = agent.execute_with_retry(
                question=sq.text + dep_context,
                schema=state["schema"],
                sub_question_id=sq.id,
                conversation_context=ctx,
            )
            elapsed = int((_now() - t0).total_seconds() * 1000)

            if "rows" in outcome:
                rows = outcome["rows"]
                sql_result = SQLResult(
                    sub_question_id=sq.id,
                    sql=outcome["sql"],
                    columns=list(rows[0].keys()) if rows else [],
                    row_count=len(rows),
                    data=rows,
                    execution_time_ms=elapsed,
                )
            else:
                sql_result = SQLResult(
                    sub_question_id=sq.id,
                    sql=outcome.get("sql", ""),
                    columns=[], row_count=0, data=[],
                    execution_time_ms=elapsed,
                    error=outcome.get("error", "Unknown error"),
                )

            results.append(sql_result)
            executed[sq.id] = sql_result

        return {"sql_results": results}
    except Exception as e:
        logger.error("sql_node failed: %s", e)
        return {"sql_results": [], "error": f"SQL generation failed: {e}"}


# ---------------------------------------------------------------------------
# sanity_check_node
# ---------------------------------------------------------------------------

def sanity_check_node(state: GraphState) -> dict:
    try:
        checker = SanityChecker()
        all_data = [row for r in state.get("sql_results", []) for row in r.data]
        # Pick the most relevant health card from the dict
        health_cards = state.get("health_cards", {})
        health_card = next(iter(health_cards.values()), None) if health_cards else None
        passed, warnings, needs_prep = checker.check(all_data, health_card)
        return {"needs_preprocessing": needs_prep}
    except Exception as e:
        logger.error("sanity_check_node failed: %s", e)
        return {"needs_preprocessing": False}


# ---------------------------------------------------------------------------
# preprocess_node
# ---------------------------------------------------------------------------

def preprocess_node(state: GraphState) -> dict:
    if state.get("preprocessing_applied"):
        return {}  # prevent infinite loop

    try:
        profiler = DataProfiler()
        # Determine which table to preprocess from SQL results
        table_name = _infer_table_from_results(state)
        health = profiler.profile_table(table_name)

        planner = CleaningAgent()
        plan = planner.generate_plan(health)

        # Auto-approve only non-destructive actions mid-pipeline
        from src.models.cleaning import CleaningActionType
        plan.actions = [
            a for a in plan.actions
            if a.action_type != CleaningActionType.REMOVE_DUPLICATES
        ]

        executor = CleaningExecutor()
        log = executor.execute(plan)

        # Update health cards dict
        health_cards = dict(state.get("health_cards", {}))
        health_cards[table_name] = health

        # Update schema so sql_node uses the cleaned table instead of the original
        import copy
        updated_schema = copy.deepcopy(state.get("schema"))
        if updated_schema:
            for t in updated_schema.tables:
                if t.name == table_name:
                    t.name = log.cleaned_view  # e.g., "transactions_cleaned"
                    break

        # Store cleaning summary so SQL agent knows what changed
        cleaning_summary = ", ".join(
            f"{a.target_column}: {a.description}" for a in plan.actions
        ) if plan.actions else ""

        return {
            "preprocessing_applied": True,
            "health_cards": health_cards,
            "schema": updated_schema,
            "cleaning_context": cleaning_summary,
        }
    except Exception as e:
        logger.error("preprocess_node failed: %s", e)
        return {"preprocessing_applied": True}  # set flag anyway to prevent loop


def _infer_table_from_results(state: GraphState) -> str:
    """Determine which table to preprocess based on SQL results and schema."""
    schema = state.get("schema")
    sql_results = state.get("sql_results", [])

    # Try to extract table name from SQL queries
    if sql_results:
        for r in sql_results:
            sql_lower = r.sql.lower() if r.sql else ""
            for table in (schema.tables if schema else []):
                if table.name.lower() in sql_lower:
                    return table.name

    # Fallback to first table (with warning)
    if schema and schema.tables:
        logger.warning(
            "Could not infer table from SQL results; falling back to '%s'",
            schema.tables[0].name,
        )
        return schema.tables[0].name

    raise ValueError("Cannot determine table to preprocess: no schema tables available")


# ---------------------------------------------------------------------------
# analysis_join_node  (fan-out dispatch point for parallel trend + anomaly)
# ---------------------------------------------------------------------------

def analysis_join_node(state: GraphState) -> dict:
    """Pass-through node that serves as a fan-out dispatch point.
    The graph uses Send() from this node to run trend and anomaly in parallel."""
    return {}


# ---------------------------------------------------------------------------
# trend_node
# ---------------------------------------------------------------------------

def trend_node(state: GraphState) -> dict:
    if state.get("error") and not state.get("sql_results"):
        return {}
    try:
        agent = _get_trend_agent()
        all_data = [row for r in state.get("sql_results", []) for row in r.data]
        result = agent.run(data=all_data, question=state["query"])
        return {"trends": result.get("trends")}
    except Exception as e:
        logger.error("trend_node failed: %s", e)
        return {"trends": None}


# ---------------------------------------------------------------------------
# anomaly_node
# ---------------------------------------------------------------------------

def anomaly_node(state: GraphState) -> dict:
    if state.get("error") and not state.get("sql_results"):
        return {}
    try:
        agent = _get_anomaly_agent()
        all_data = [row for r in state.get("sql_results", []) for row in r.data]
        result = agent.run(data=all_data, question=state["query"])
        return {"anomalies": result.get("anomalies")}
    except Exception as e:
        logger.error("anomaly_node failed: %s", e)
        return {"anomalies": None}


# ---------------------------------------------------------------------------
# narrative_node
# ---------------------------------------------------------------------------

def narrative_node(state: GraphState) -> dict:
    try:
        # If there were errors and no data, produce a graceful message
        if not state.get("sql_results") or all(r.row_count == 0 for r in state.get("sql_results", [])):
            err = state.get("error", "")
            msg = (
                f"I wasn't able to retrieve data for your question. {err}"
                if err
                else "No data was returned for your query."
            )
            return {"narrative": msg, "chart_spec": None}

        agent = _get_narrative_agent()
        all_data = [row for r in state.get("sql_results", []) for row in r.data]
        result = agent.run(
            question=state["query"],
            sql_results=all_data,
            trends=state.get("trends"),
            anomalies=state.get("anomalies"),
            revision_notes=state.get("revision_notes"),
        )
        return {
            "narrative": result["narrative"],
            "chart_spec": result.get("chart_spec"),
        }
    except Exception as e:
        logger.error("narrative_node failed: %s", e)
        return {
            "narrative": f"I encountered an issue generating a narrative for your question. Error: {e}",
            "chart_spec": None,
        }


# ---------------------------------------------------------------------------
# critique_node
# ---------------------------------------------------------------------------

def critique_node(state: GraphState) -> dict:
    try:
        narrative = state.get("narrative", "")
        # Skip critique if narrative already acknowledges an error
        if "wasn't able" in narrative or "encountered an issue" in narrative:
            return {"critique_approved": True, "critique_count": state.get("critique_count", 0) + 1}

        agent = _get_critique_agent()
        all_data = [row for r in state.get("sql_results", []) for row in r.data]
        anomalies = state.get("anomalies")

        result = agent.run(
            narrative=narrative,
            question=state["query"],
            sql_results=all_data,
            anomalies_mentioned="anomal" in narrative.lower(),
            anomalies_found=anomalies.anomaly_count if anomalies else 0,
        )

        count = state.get("critique_count", 0) + 1
        return {
            "critique_approved": result.get("approved", True),
            "critique_count": count,
            "revision_notes": result.get("revision_notes"),
        }
    except Exception as e:
        logger.error("critique_node failed: %s", e)
        return {"critique_approved": True, "critique_count": state.get("critique_count", 0) + 1}


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------

def verify_node(state: GraphState) -> dict:
    try:
        scorer = _get_confidence_scorer()
        sql_results = state.get("sql_results", [])

        if sql_results and sql_results[0].error is None:
            first = sql_results[0]
            confidence = scorer.score(
                question=state["query"],
                sql=first.sql,
                row_count=first.row_count,
                schema_valid=True,
                sanity_warnings=[],
            )
        else:
            confidence = ConfidenceScore(
                score=0.0,
                reasoning="No valid SQL results.",
                uncertain_aspects=["query execution"],
            )

        # Flag for human review when confidence is low or critique loops maxed out
        from config.settings import settings
        requires_review = (
            confidence.score < settings.CONFIDENCE_THRESHOLD
            or state.get("critique_count", 0) >= settings.MAX_CRITIQUE_LOOPS
        )

        return {"confidence": confidence, "requires_human_review": requires_review}
    except Exception as e:
        logger.error("verify_node failed: %s", e)
        return {"confidence": ConfidenceScore(score=0.0, reasoning=str(e), uncertain_aspects=[])}


# ---------------------------------------------------------------------------
# format_output_node  (handles META route)
# ---------------------------------------------------------------------------

def format_output_node(state: GraphState) -> dict:
    route = state.get("route")
    result = {}

    # Surface any accumulated errors in the narrative (Issue #6)
    error = state.get("error")
    narrative = state.get("narrative", "")
    if error and not narrative:
        result["narrative"] = f"An error occurred during analysis: {error}"

    if route == RouteType.META:
        result["narrative"] = _format_meta_response(state)

    if route == RouteType.CLARIFICATION:
        result["narrative"] = (
            "I need more details to answer your question accurately. "
            "Could you be more specific about what data you want to analyze?"
        )

    # Ensure confidence is always set (Issue #3)
    if state.get("confidence") is None:
        result["confidence"] = ConfidenceScore(
            score=1.0 if route == RouteType.META else 0.0,
            reasoning="Meta/format response" if route == RouteType.META else "No SQL executed",
            uncertain_aspects=[],
        )

    return result


def _format_meta_response(state: GraphState) -> str:
    schema = state.get("schema")
    query = state.get("query", "").lower()

    if not schema:
        return "No schema available. Please upload or connect to a data source first."

    lines = []

    if any(kw in query for kw in ("table", "what data", "available")):
        lines.append("**Available tables:**")
        for t in schema.tables:
            lines.append(f"- **{t.name}** — {t.row_count:,} rows, {len(t.columns)} columns")

    if any(kw in query for kw in ("column", "field", "attribute")):
        for t in schema.tables:
            if t.name.lower() in query or "all" in query or "column" in query:
                lines.append(f"\n**{t.name} columns:**")
                for c in t.columns:
                    pk = " (PK)" if c.is_primary_key else ""
                    samples = f" — e.g. {', '.join(c.sample_values[:3])}" if c.sample_values else ""
                    lines.append(f"- `{c.name}` ({c.dtype}){pk}{samples}")

    if any(kw in query for kw in ("describe", "overview", "summary", "about")):
        lines.append("**Database overview:**")
        for t in schema.tables:
            lines.append(f"\n**{t.name}** ({t.row_count:,} rows)")
            col_summary = ", ".join(f"{c.name} ({c.dtype})" for c in t.columns[:8])
            if len(t.columns) > 8:
                col_summary += f" ... and {len(t.columns) - 8} more"
            lines.append(f"Columns: {col_summary}")

        health_cards = state.get("health_cards", {})
        if health_cards:
            all_warnings = []
            for card in health_cards.values():
                all_warnings.extend(card.warnings[:3])
            if all_warnings:
                lines.append("\n**Data quality warnings:**")
                for w in all_warnings[:5]:
                    lines.append(f"- {w}")

    if any(kw in query for kw in ("how much", "how many row", "size", "count")):
        lines.append("**Row counts:**")
        for t in schema.tables:
            lines.append(f"- {t.name}: {t.row_count:,} rows")

    if not lines:
        # Generic schema dump
        lines.append("**Schema summary:**")
        for t in schema.tables:
            lines.append(f"- **{t.name}**: {t.row_count:,} rows | columns: {', '.join(c.name for c in t.columns)}")

    if schema.foreign_keys:
        lines.append("\n**Relationships:**")
        for fk in schema.foreign_keys:
            lines.append(f"- {fk['from_table']}.{fk['from_col']} → {fk['to_table']}.{fk['to_col']}")

    return "\n".join(lines)
