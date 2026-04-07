from langgraph.graph import StateGraph, END
from src.orchestrator.state import GraphState
from src.orchestrator.edges import (
    route_query,
    template_has_results,
    # should_decompose,  # decomposition bypassed
    should_preprocess,
    should_revise,
)
from src.orchestrator.nodes import (
    classify_node,
    template_node,
    # decompose_node,  # decomposition bypassed
    sql_node,
    sanity_check_node,
    preprocess_node,
    trend_node,
    anomaly_node,
    narrative_node,
    critique_node,
    verify_node,
    format_output_node,
)


def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # Add nodes
    g.add_node("classify", classify_node)
    g.add_node("template_sql", template_node)
    # g.add_node("decompose", decompose_node)  # decomposition bypassed
    g.add_node("sql_execute", sql_node)
    g.add_node("sanity_check", sanity_check_node)
    g.add_node("preprocess", preprocess_node)
    g.add_node("trend_analysis", trend_node)
    g.add_node("anomaly_detection", anomaly_node)
    g.add_node("narrative", narrative_node)
    g.add_node("critique", critique_node)
    g.add_node("verify", verify_node)
    g.add_node("format_output", format_output_node)

    # Entry
    g.set_entry_point("classify")

    # After classify → route to simple or analytical
    g.add_conditional_edges("classify", route_query, {
        "simple": "template_sql",
        "analytical": "sql_execute",   # skip decomposition, go directly to SQL
        "meta": "format_output",
        "clarification": "format_output",
    })

    # Simple path: if template matched → verify; otherwise fall through to SQL
    g.add_conditional_edges("template_sql", template_has_results, {
        "has_results": "verify",
        "no_results": "sql_execute",   # skip decomposition, go directly to SQL
    })

    # Decomposition bypassed — analytical queries go straight to sql_execute
    # g.add_edge("decompose", "sql_execute")
    g.add_edge("sql_execute", "sanity_check")

    # Sanity check → preprocess or continue
    g.add_conditional_edges("sanity_check", should_preprocess, {
        "preprocess": "preprocess",
        "continue": "trend_analysis",
    })
    g.add_edge("preprocess", "sql_execute")  # re-run SQL on cleaned data

    # Parallel-ish (LangGraph runs sequentially but logically parallel)
    g.add_edge("trend_analysis", "anomaly_detection")
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

    return g.compile()

