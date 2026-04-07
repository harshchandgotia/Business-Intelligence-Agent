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
        """Deterministic anomaly detection using Z-score + IQR. No LLM."""
        anomalies = []
        seen_indices = {}  # (col, row_index) -> anomaly dict, for dedup

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 5:
                continue

            # Method 1: Z-score
            z_scores = np.abs(stats.zscore(series))
            outlier_mask = z_scores > settings.OUTLIER_ZSCORE

            for idx in series[outlier_mask].index:
                key = (col, int(idx))
                seen_indices[key] = {
                    "column": col,
                    "row_index": int(idx),
                    "value": float(df.loc[idx, col]),
                    "z_score": float(z_scores[idx]),
                    "expected_range": f"{series.mean() - 2*series.std():.2f} to "
                                     f"{series.mean() + 2*series.std():.2f}",
                    "method": "z-score",
                    "severity": "high" if z_scores[idx] > 4 else "medium",
                }

            # Method 2: IQR (more robust for skewed distributions like revenue)
            iqr_outliers = self._detect_iqr_outliers(series, col)
            for anomaly in iqr_outliers:
                key = (anomaly["column"], anomaly["row_index"])
                if key in seen_indices:
                    # Both methods flagged it — upgrade severity and note both
                    seen_indices[key]["method"] = "z-score+IQR"
                    seen_indices[key]["severity"] = "high"
                else:
                    seen_indices[key] = anomaly

            # Method 3: Period-over-period check (if data looks like time series)
            if len(series) >= 4:
                pct_changes = series.pct_change().dropna()
                large_changes = pct_changes[pct_changes.abs() > 0.5]  # >50% change
                for idx in large_changes.index:
                    key = (col, int(idx))
                    if key not in seen_indices:
                        seen_indices[key] = {
                            "column": col,
                            "row_index": int(idx),
                            "value": float(df.loc[idx, col]),
                            "pct_change": float(pct_changes[idx]),
                            "method": "period-over-period",
                            "severity": "high" if abs(pct_changes[idx]) > 1.0 else "medium",
                        }

        anomalies = list(seen_indices.values())
        # Sort by severity (high first) then by absolute value
        anomalies.sort(key=lambda a: (0 if a["severity"] == "high" else 1, -abs(a["value"])))
        return anomalies[:20]  # cap

    def _detect_iqr_outliers(self, series: pd.Series, col: str) -> list[dict]:
        """IQR-based outlier detection — more robust for skewed distributions."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return []
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (series < lower) | (series > upper)
        results = []
        for idx in series[mask].index:
            val = float(series[idx])
            results.append({
                "column": col,
                "row_index": int(idx),
                "value": val,
                "expected_range": f"{lower:.2f} to {upper:.2f}",
                "method": "IQR",
                "severity": "high" if val > Q3 + 3 * IQR or val < Q1 - 3 * IQR else "medium",
            })
        return results
