import json
from src.llm.json_utils import extract_json
import numpy as np
import pandas as pd
from scipy import stats
from src.agents.base import BaseAgent
from src.models.query import AnomalyResult
from config.settings import settings


class AnomalyAgent(BaseAgent):
    name = "anomaly_agent"

    def _execute(self, data: list[dict], question: str) -> dict:
        df = pd.DataFrame(data)
        anomalies = self._detect_anomalies(df)

        if not anomalies:
            return {
                "anomalies": AnomalyResult(
                    anomalies=[], anomaly_count=0, summary="No anomalies detected."
                )
            }

        # LLM explains the anomalies in context
        system = (
            "You explain data anomalies in business context.\n"
            "Respond with JSON: {\"explanations\": [{\"index\": 0, \"explanation\": \"...\"}], "
            "\"summary\": \"...\"}"
        )

        prompt = (
            f"Question: {question}\n"
            f"Anomalies found:\n{json.dumps(anomalies, default=str)}\n"
            "Explain each anomaly's potential business significance."
        )

        response = self._llm.generate(prompt, system=system, json_mode=True)
        parsed = extract_json(response, fallback={"explanations": [], "summary": "Unable to parse anomaly explanations."})

        # Merge explanations back
        for i, anomaly in enumerate(anomalies):
            for expl in parsed.get("explanations", []):
                if expl.get("index") == i:
                    anomaly["explanation"] = expl["explanation"]

        return {
            "anomalies": AnomalyResult(
                anomalies=anomalies,
                anomaly_count=len(anomalies),
                summary=parsed.get("summary", ""),
            )
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> list[dict]:
        """Deterministic anomaly detection. No LLM."""
        anomalies = []

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 5:
                continue

            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            outlier_mask = z_scores > settings.OUTLIER_ZSCORE

            for idx in series[outlier_mask].index:
                anomalies.append({
                    "column": col,
                    "row_index": int(idx),
                    "value": float(df.loc[idx, col]),
                    "z_score": float(z_scores[idx]),
                    "expected_range": f"{series.mean() - 2*series.std():.2f} to "
                                     f"{series.mean() + 2*series.std():.2f}",
                    "severity": "high" if z_scores[idx] > 4 else "medium",
                })

            # Period-over-period check (if data looks like time series)
            if len(series) >= 4:
                pct_changes = series.pct_change().dropna()
                large_changes = pct_changes[pct_changes.abs() > 0.5]  # >50% change
                for idx in large_changes.index:
                    anomalies.append({
                        "column": col,
                        "row_index": int(idx),
                        "value": float(df.loc[idx, col]),
                        "pct_change": float(pct_changes[idx]),
                        "severity": "high" if abs(pct_changes[idx]) > 1.0 else "medium",
                    })

        return anomalies[:20]  # cap
