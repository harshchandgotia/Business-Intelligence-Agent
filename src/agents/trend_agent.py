import json
from src.llm.json_utils import extract_json
import pandas as pd
from src.agents.base import BaseAgent
from src.models.query import TrendResult


class TrendAgent(BaseAgent):
    name = "trend_agent"

    def _execute(self, data: list[dict], question: str) -> dict:
        df = pd.DataFrame(data)
        stats_summary = self._compute_stats(df)

        system = (
            "You are a data trend analyst. Given query results and statistics,\n"
            "identify meaningful trends, patterns, and changes.\n"
            "Respond with JSON:\n"
            "{\n"
            '  "trends": [{"column": "...", "direction": "increasing|decreasing|stable",\n'
            '              "magnitude": "...", "period": "...", "description": "..."}],\n'
            '  "seasonal_patterns": ["..."],\n'
            '  "summary": "2-3 sentence trend summary"\n'
            "}"
        )

        prompt = (
            f"Question: {question}\n\n"
            f"Data statistics:\n{stats_summary}\n\n"
            f"Sample rows (first 20):\n{json.dumps(data[:20], default=str)}\n\n"
            "Identify trends and patterns."
        )

        response = self._llm.generate(prompt, system=system, json_mode=True)
        parsed = extract_json(response, fallback={
            "trends": [], "seasonal_patterns": [], "summary": "Unable to parse trend analysis."
        })

        return {"trends": TrendResult(**parsed)}

    def _compute_stats(self, df: pd.DataFrame) -> str:
        """Deterministic stats summary for the LLM."""
        lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"]

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                lines.append(
                    f"{col}: min={df[col].min()}, max={df[col].max()}, "
                    f"mean={df[col].mean():.2f}, std={df[col].std():.2f}"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                lines.append(f"{col}: range {df[col].min()} to {df[col].max()}")

        return "\n".join(lines)
