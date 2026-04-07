import json
import re
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

        # Deterministic pre-check: verify numbers in narrative against actual data
        mismatches = self._verify_numbers_against_data(narrative, sql_results)

        mismatch_context = ""
        if mismatches:
            mismatch_context = (
                f"\n\nPre-detected factual mismatches:\n"
                + "\n".join(f"- {m}" for m in mismatches)
                + "\n"
            )

        prompt = (
            f"Original question: {question}\n\n"
            f"Generated narrative:\n{narrative}\n\n"
            f"Actual data (first 20 rows): {json.dumps(sql_results[:20], default=str)}\n\n"
            f"Anomalies found by detector: {anomalies_found}\n"
            f"Anomalies mentioned in narrative: {anomalies_mentioned}\n"
            f"{mismatch_context}\n"
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

        # If deterministic check found mismatches, force rejection
        if mismatches and parsed.get("approved", True):
            parsed["approved"] = False
            parsed["factual_errors"] = parsed.get("factual_errors", []) + mismatches
            parsed["revision_notes"] = (
                (parsed.get("revision_notes") or "")
                + f" Fix these number mismatches: {'; '.join(mismatches)}"
            ).strip()

        return {
            "approved": parsed.get("approved", True),
            "issues": parsed.get("issues", []),
            "revision_notes": parsed.get("revision_notes"),
            "factual_errors": parsed.get("factual_errors", []),
        }

    def _extract_numbers_from_narrative(self, narrative: str) -> list[float]:
        """Extract significant numbers from narrative text."""
        # Match currency amounts, percentages, and plain numbers
        raw = re.findall(r"[\$€£]?([\d,]+\.?\d*)", narrative)
        numbers = []
        for n in raw:
            try:
                val = float(n.replace(",", ""))
                if val > 0:  # skip 0
                    numbers.append(val)
            except ValueError:
                continue
        return numbers

    def _verify_numbers_against_data(
        self, narrative: str, data: list[dict]
    ) -> list[str]:
        """Check if large numbers in the narrative actually appear in (or can be
        derived from) the query results. Returns list of mismatch descriptions."""
        if not data:
            return []

        # Collect all numeric values from data
        all_data_values = set()
        for row in data:
            for v in row.values():
                if isinstance(v, (int, float)):
                    all_data_values.add(round(float(v), 2))

        # Also add common aggregates (sums per numeric column)
        if len(data) > 1:
            for key in data[0]:
                vals = [row.get(key) for row in data if isinstance(row.get(key), (int, float))]
                if vals:
                    all_data_values.add(round(sum(vals), 2))
                    all_data_values.add(round(sum(vals) / len(vals), 2))  # mean
                    all_data_values.add(round(max(vals), 2))
                    all_data_values.add(round(min(vals), 2))

        if not all_data_values:
            return []

        narrative_numbers = self._extract_numbers_from_narrative(narrative)
        mismatches = []
        for n in narrative_numbers:
            # Only check significant numbers (> 100) to avoid false positives on counts etc.
            if n > 100 and not any(
                abs(n - d) / max(abs(d), 1) < 0.05 for d in all_data_values
            ):
                mismatches.append(f"{n:,.2f} not found in query results (within 5% tolerance)")

        return mismatches[:5]  # cap
