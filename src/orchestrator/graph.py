import logging
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from src.orchestrator.state import GraphState
from config.settings import settings

logger = logging.getLogger(__name__)
from src.orchestrator.edges import (
    route_query,
    template_has_results,
    should_decompose,
    should_preprocess,
    should_revise,
)
from src.orchestrator.nodes import (
    classify_node,
    template_node,
    decompose_node,
    sql_node,
    sanity_check_node,
    preprocess_node,
    trend_node,
    anomaly_node,
    analysis_join_node,
    narrative_node,
    critique_node,
    verify_node,
    format_output_node,
)


def _fan_out_analysis(state: GraphState):
    """Fan-out: run trend and anomaly detection in parallel via Send."""
    return [
        Send("trend_analysis", state),
        Send("anomaly_detection", state),
    ]


def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # Add nodes
    g.add_node("classify", classify_node)
    g.add_node("template_sql", template_node)
    g.add_node("decompose", decompose_node)
    g.add_node("sql_execute", sql_node)
    g.add_node("sanity_check", sanity_check_node)
    g.add_node("preprocess", preprocess_node)
    g.add_node("trend_analysis", trend_node)
    g.add_node("anomaly_detection", anomaly_node)
    g.add_node("analysis_join", analysis_join_node)
    g.add_node("narrative", narrative_node)
    g.add_node("critique", critique_node)
    g.add_node("verify", verify_node)
    g.add_node("format_output", format_output_node)

    # Entry
    g.set_entry_point("classify")

    # After classify → route to simple or analytical
    g.add_conditional_edges("classify", route_query, {
        "simple": "template_sql",
        "analytical": "decompose",
        "meta": "format_output",
        "clarification": "format_output",
    })

    # Simple path: if template matched → verify; otherwise fall through to SQL
    g.add_conditional_edges("template_sql", template_has_results, {
        "has_results": "verify",
        "no_results": "decompose",
    })

    # Decompose → decide if decomposition is needed → SQL
    g.add_conditional_edges("decompose", should_decompose, {
        "decompose": "sql_execute",
        "skip": "sql_execute",
    })

    g.add_edge("sql_execute", "sanity_check")

    # Sanity check → preprocess or continue to parallel analysis
    g.add_conditional_edges("sanity_check", should_preprocess, {
        "preprocess": "preprocess",
        "continue": "analysis_join",
    })
    g.add_edge("preprocess", "sql_execute")  # re-run SQL on cleaned data

    # Fan-out: analysis_join dispatches trend + anomaly in parallel
    g.add_conditional_edges("analysis_join", _fan_out_analysis)
    g.add_edge("trend_analysis", "narrative")
    g.add_edge("anomaly_detection", "narrative")

    # Critique loop
    g.add_edge("narrative", "critique")
    g.add_conditional_edges("critique", should_revise, {
        "revise": "narrative",
        "approve": "verify",
    })

    # Final
    g.add_edge("verify", "format_output")
    g.add_edge("format_output", END)

    # Checkpointing for resumable runs
    checkpointer = _get_checkpointer()
    return g.compile(checkpointer=checkpointer)


def _get_checkpointer():
    """Try to create a PostgresSaver checkpointer; fall back to None."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver.from_conn_string(settings.db_url)
        checkpointer.setup()  # create checkpoint tables if needed
        logger.info("LangGraph checkpointing enabled (PostgresSaver)")
        return checkpointer
    except ImportError:
        logger.info("langgraph-checkpoint-postgres not installed; checkpointing disabled")
        return None
    except Exception as e:
        logger.warning("Checkpointing setup failed: %s; continuing without checkpointing", e)
        return None

