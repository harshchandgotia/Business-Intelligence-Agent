import json
from src.llm.json_utils import extract_json
from src.agents.base import BaseAgent
from src.models.query import TrendResult, AnomalyResult


class NarrativeAgent(BaseAgent):
    name = "narrative_agent"

    def _execute(
        self,
        question: str,
        sql_results: list[dict],
        trends: TrendResult | None = None,
        anomalies: AnomalyResult | None = None,
        revision_notes: str | None = None,
    ) -> dict:
        system = self._load_prompt("narrative.txt")

        context_parts = [
            f"Original question: {question}",
            f"Query results ({len(sql_results)} rows): {json.dumps(sql_results[:30], default=str)}",
        ]

        if trends:
            context_parts.append(f"Trends identified: {trends.summary}")
        if anomalies and anomalies.anomaly_count > 0:
            context_parts.append(f"Anomalies: {anomalies.summary}")
        if revision_notes:
            context_parts.append(f"REVISION REQUIRED — Critique feedback: {revision_notes}")

        prompt = "\n\n".join(context_parts)

        response = self._llm.generate(prompt, system=system)

        # Chart suggestion with narrative context for coherence
        columns = list(sql_results[0].keys()) if sql_results else []
        row_count = len(sql_results)
        chart_prompt = (
            f"Data question: {question}\n"
            f"Narrative written: {response[:500]}\n"
            f"Columns available: {columns}\n"
            f"Row count: {row_count}\n"
            "Suggest the best chart that supports this specific narrative.\n"
            "For ranked lists or top-N results with <=20 rows, prefer 'table'.\n"
            "Respond JSON: "
            '{"chart_type": "bar|line|pie|scatter|heatmap|table|none", '
            '"x_column": "...", "y_column": "...", "group_by": "..." or null}'
        )
        from config.settings import settings
        chart_response = self._llm.generate(
            chart_prompt, json_mode=True, model=settings.GROQ_MODEL_LIGHT,
        )

        try:
            chart_spec = extract_json(chart_response, fallback={"chart_type": "none"})
        except Exception:
            chart_spec = {"chart_type": "none"}

        return {
            "narrative": response,
            "chart_spec": chart_spec,
        }
