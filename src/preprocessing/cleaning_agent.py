import json
import os
from src.llm.json_utils import extract_json
from src.llm.factory import get_llm
from src.models.health import DataHealthCard
from src.models.cleaning import CleaningPlan, CleaningAction, CleaningActionType
from config.settings import settings


class CleaningAgent:
    """LLM decides what to clean. Execution is deterministic."""

    def __init__(self):
        self._llm = get_llm()

    def generate_plan(self, health_card: DataHealthCard) -> CleaningPlan:
        prompt = self._build_prompt(health_card)
        response = self._llm.generate(prompt, json_mode=True)
        return self._parse_plan(response, health_card.table_name)

    def _build_prompt(self, card: DataHealthCard) -> str:
        prompt_path = os.path.join(settings.PROMPTS_DIR, "cleaning_plan.txt")
        with open(prompt_path) as f:
            template = f.read()

        issues = []
        for col in card.columns:
            if col.inconsistent_values:
                issues.append(
                    f"Column '{col.name}': inconsistent values — "
                    f"{json.dumps(col.inconsistent_values[:3])}"
                )
            if col.null_pct > 0.05:
                issues.append(
                    f"Column '{col.name}': {col.null_pct:.1%} nulls"
                )
        if card.duplicate_row_count > 0:
            issues.append(
                f"{card.duplicate_row_count} duplicate rows"
            )

        return template.format(
            table_name=card.table_name,
            row_count=card.row_count,
            quality_score=card.overall_quality_score,
            issues="\n".join(issues) if issues else "No major issues found.",
        )

    def _parse_plan(self, response: str, table_name: str) -> CleaningPlan:
        """Parse LLM JSON response into CleaningPlan."""
        try:
            data = extract_json(response, fallback={"actions": []})
            actions = [CleaningAction(**a) for a in data.get("actions", [])]
        except Exception:
            actions = []

        requires_approval = any(a.is_destructive for a in actions)
        return CleaningPlan(
            table_name=table_name,
            actions=actions,
            estimated_quality_improvement=0.0,
            requires_user_approval=requires_approval,
        )
