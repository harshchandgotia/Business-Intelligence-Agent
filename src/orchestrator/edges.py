from src.orchestrator.state import GraphState
from src.models.query import RouteType
from config.settings import settings


def route_query(state: GraphState) -> str:
    route = state.get("route")
    if route == RouteType.SIMPLE:
        return "simple"
    elif route == RouteType.ANALYTICAL:
        return "analytical"
    elif route == RouteType.META:
        return "meta"
    else:
        return "clarification"


def template_has_results(state: GraphState) -> str:
    """Check if template_node produced SQL results; if not, re-route to analytical."""
    sql_results = state.get("sql_results", [])
    if sql_results and any(r.row_count > 0 for r in sql_results):
        return "has_results"
    return "no_results"


def should_preprocess(state: GraphState) -> str:
    if state.get("needs_preprocessing") and not state.get("preprocessing_applied"):
        return "preprocess"
    return "continue"


def should_decompose(state: GraphState) -> str:
    """Gate decomposition: only activate for complex analytical queries."""
    decomp = state.get("decomposition")
    if decomp and len(decomp.sub_questions) > 1:
        return "decompose"
    return "skip"


def should_revise(state: GraphState) -> str:
    if (
        not state.get("critique_approved", False)
        and state.get("critique_count", 0) < settings.MAX_CRITIQUE_LOOPS
    ):
        return "revise"
    return "approve"
