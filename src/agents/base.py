from abc import ABC, abstractmethod
from datetime import datetime, timezone
from src.llm.factory import get_llm, LLMClient
from src.models.query import AgentStep
from config.settings import settings


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self):
        self._llm: LLMClient = get_llm()
        self._start_time: datetime | None = None

    def run(self, **kwargs) -> dict:
        self._start_time = datetime.now(timezone.utc)
        try:
            result = self._execute(**kwargs)
            return result
        except Exception as e:
            return {"error": str(e)}

    @abstractmethod
    def _execute(self, **kwargs) -> dict:
        ...

    def _load_prompt(self, filename: str) -> str:
        import os
        prompt_path = os.path.join(settings.PROMPTS_DIR, filename)
        with open(prompt_path) as f:
            return f.read()

    def _build_step(self, input_summary: str, output_summary: str, tokens: int = 0) -> AgentStep:
        now = datetime.now(timezone.utc)
        if tokens == 0 and (input_summary or output_summary):
            tokens = self._llm.count_tokens(input_summary + " " + output_summary)
        return AgentStep(
            agent_name=self.name,
            started_at=self._start_time or now,
            ended_at=now,
            input_summary=input_summary,
            output_summary=output_summary,
            tokens_used=tokens,
        )
