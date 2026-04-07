from typing import TypedDict, Optional, Annotated
from src.models.query import (
    RouteType, DecompositionResult, SQLResult, TrendResult,
    AnomalyResult, ConfidenceScore, VerificationResult, ExecutionTrace,
)
from src.models.health import DataHealthCard
from src.models.schema import DatabaseSchema


class GraphState(TypedDict):
    # Input
    query: str
    conversation_context: str

    # Schema
    schema: DatabaseSchema
    health_cards: dict  # table_name -> DataHealthCard

    # Routing
    route: Optional[RouteType]

    # Decomposition
    decomposition: Optional[DecompositionResult]

    # SQL results (accumulated from parallel agents)
    sql_results: list[SQLResult]

    # Analysis
    trends: Optional[TrendResult]
    anomalies: Optional[AnomalyResult]

    # Narrative
    narrative: str
    chart_spec: Optional[dict]

    # Critique loop
    critique_count: int
    critique_approved: bool
    revision_notes: Optional[str]

    # Verification
    verification: Optional[VerificationResult]

    # Preprocessing detour
    needs_preprocessing: bool
    preprocessing_applied: bool
    cleaning_context: Optional[str]  # summary of cleaning actions for SQL context

    # Human review flag
    requires_human_review: bool

    # Output
    confidence: Optional[ConfidenceScore]
    trace: Optional[ExecutionTrace]
    error: Optional[str]
