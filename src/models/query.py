from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
import pandas as pd
from datetime import datetime


class RouteType(str, Enum):
    SIMPLE = "simple"           # deterministic template path
    ANALYTICAL = "analytical"   # full agent pipeline
    META = "meta"               # about the data/system itself
    CLARIFICATION = "clarification"  # need more info from user


class SubQuestion(BaseModel):
    id: int
    text: str
    depends_on: list[int] = []  # IDs of sub-questions this depends on


class DecompositionResult(BaseModel):
    original_query: str
    sub_questions: list[SubQuestion]
    reasoning: str


class SQLResult(BaseModel):
    sub_question_id: int
    sql: str
    columns: list[str]
    row_count: int
    data: list[dict]           # list of row dicts
    execution_time_ms: float
    error: Optional[str] = None


class TrendResult(BaseModel):
    trends: list[dict]         # [{column, direction, magnitude, period, description}]
    seasonal_patterns: list[str]
    summary: str


class AnomalyResult(BaseModel):
    anomalies: list[dict]      # [{column, value, expected_range, severity, explanation}]
    anomaly_count: int
    summary: str


class ConfidenceScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    uncertain_aspects: list[str]


class VerificationResult(BaseModel):
    schema_valid: bool
    schema_errors: list[str]
    sanity_passed: bool
    sanity_warnings: list[str]
    confidence: ConfidenceScore
    needs_preprocessing: bool
    preprocessing_reason: Optional[str] = None


class AgentStep(BaseModel):
    agent_name: str
    started_at: datetime
    ended_at: datetime
    input_summary: str
    output_summary: str
    tokens_used: int
    error: Optional[str] = None


class ExecutionTrace(BaseModel):
    steps: list[AgentStep]
    route_taken: RouteType
    total_time_ms: float
    total_tokens: int
    decomposition: Optional[DecompositionResult] = None


class QueryRequest(BaseModel):
    query: str
    conversation_id: str
    turn_number: int


class QueryResponse(BaseModel):
    narrative: str
    sql_queries: list[str]
    data_tables: list[dict]     # serialized DataFrames
    chart_spec: Optional[dict]  # Plotly figure JSON
    confidence: ConfidenceScore
    anomalies: Optional[AnomalyResult] = None
    trends: Optional[TrendResult] = None
    trace: ExecutionTrace
    needs_user_input: bool = False
    user_prompt: Optional[str] = None  # question for user if needed
