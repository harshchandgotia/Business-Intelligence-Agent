import json
from src.llm.json_utils import extract_json
from src.llm.factory import get_llm
from src.models.query import ConfidenceScore


class ConfidenceScorer:
    """LLM-based confidence scoring."""

    def score(
        self,
        question: str,
        sql: str,
        row_count: int,
        schema_valid: bool,
        sanity_warnings: list[str],
    ) -> ConfidenceScore:
        llm = get_llm()

        prompt = (
            f"Original question: {question}\n"
            f"Generated SQL: {sql}\n"
            f"Result row count: {row_count}\n"
            f"Schema validation passed: {schema_valid}\n"
            f"Sanity warnings: {sanity_warnings if sanity_warnings else 'None'}\n\n"
            "Rate your confidence (0.0 to 1.0) that this SQL correctly answers "
            "the question. Respond JSON:\n"
            '{"score": 0.85, "reasoning": "...", "uncertain_aspects": ["..."]}'
        )

        response = llm.generate(prompt, json_mode=True)
        parsed = extract_json(response, fallback={"score": 0.5, "reasoning": "Unable to parse confidence.", "uncertain_aspects": []})

        return ConfidenceScore(
            score=max(0.0, min(1.0, parsed.get("score", 0.5))),
            reasoning=parsed.get("reasoning", ""),
            uncertain_aspects=parsed.get("uncertain_aspects", []),
        )
