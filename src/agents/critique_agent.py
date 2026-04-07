import json
from src.llm.json_utils import extract_json
from src.agents.base import BaseAgent


class CritiqueAgent(BaseAgent):
    name = "critique_agent"

    def _execute(
        self,
        narrative: str,
        question: str,
        sql_results: list[dict],
        anomalies_mentioned: bool,
        anomalies_found: int,
    ) -> dict:
        system = self._load_prompt("critique.txt")

        prompt = (
            f"Original question: {question}\n\n"
            f"Generated narrative:\n{narrative}\n\n"
            f"Actual data (first 20 rows): {json.dumps(sql_results[:20], default=str)}\n\n"
            f"Anomalies found by detector: {anomalies_found}\n"
            f"Anomalies mentioned in narrative: {anomalies_mentioned}\n\n"
            "Evaluate this report. Respond with JSON:\n"
            "{\n"
            '  "approved": true/false,\n'
            '  "issues": ["issue1", "issue2"],\n'
            '  "revision_notes": "specific instructions for improvement" or null,\n'
            '  "factual_errors": ["error1"] or []\n'
            "}"
        )

        response = self._llm.generate(prompt, system=system, json_mode=True)
        parsed = extract_json(response, fallback={"approved": True, "issues": [], "revision_notes": None, "factual_errors": []})

        return {
            "approved": parsed.get("approved", True),
            "issues": parsed.get("issues", []),
            "revision_notes": parsed.get("revision_notes"),
            "factual_errors": parsed.get("factual_errors", []),
        }
