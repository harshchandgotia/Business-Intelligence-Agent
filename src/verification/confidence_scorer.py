import json
from src.llm.json_utils import extract_json
from src.llm.factory import get_llm
from src.models.query import ConfidenceScore
from config.settings import settings


class ConfidenceScorer:
    """Confidence scoring with deterministic fast-paths and LLM fallback."""

    def score(
        self,
        question: str,
        sql: str,
        row_count: int,
        schema_valid: bool,
        sanity_warnings: list[str],
    ) -> ConfidenceScore:
        # Deterministic fast-paths — no LLM needed for obvious cases
        if not schema_valid:
            return ConfidenceScore(
                score=0.1,
                reasoning="Schema validation failed — table or column names may be incorrect.",
                uncertain_aspects=["table/column names"],
            )

        if row_count == 0:
            return ConfidenceScore(
                score=0.2,
                reasoning="Query returned no rows — filter conditions may be too restrictive or values may not exist.",
                uncertain_aspects=["filter conditions", "data availability"],
            )

        if len(sanity_warnings) >= 3:
            return ConfidenceScore(
                score=0.4,
                reasoning=f"Multiple data quality warnings detected: {'; '.join(sanity_warnings[:3])}",
                uncertain_aspects=sanity_warnings[:5],
            )

        if len(sanity_warnings) >= 1:
            # Moderate concern — cap at 0.7 but let LLM refine
            cap = 0.7
        else:
            cap = 1.0

        # LLM scoring for ambiguous cases
        return self._llm_score(question, sql, row_count, sanity_warnings, cap)

    def _llm_score(
        self,
        question: str,
        sql: str,
        row_count: int,
        sanity_warnings: list[str],
        cap: float = 1.0,
    ) -> ConfidenceScore:
        llm = get_llm()

        prompt = (
            f"Original question: {question}\n"
            f"Generated SQL: {sql}\n"
            f"Result row count: {row_count}\n"
            f"Schema validation passed: True\n"
            f"Sanity warnings: {sanity_warnings if sanity_warnings else 'None'}\n\n"
            "Rate your confidence (0.0 to 1.0) that this SQL correctly answers "
            "the question. Respond JSON:\n"
            '{"score": 0.85, "reasoning": "...", "uncertain_aspects": ["..."]}'
        )

        response = llm.generate(
            prompt, json_mode=True, model=settings.GROQ_MODEL_LIGHT,
        )
        parsed = extract_json(response, fallback={
            "score": 0.5,
            "reasoning": "Unable to parse confidence.",
            "uncertain_aspects": [],
        })

        score = max(0.0, min(cap, parsed.get("score", 0.5)))
        return ConfidenceScore(
            score=score,
            reasoning=parsed.get("reasoning", ""),
            uncertain_aspects=parsed.get("uncertain_aspects", []),
        )
