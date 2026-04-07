from collections import deque
from config.settings import settings


class ConversationMemory:
    """Sliding window conversation history."""

    def __init__(self):
        self._history: deque[dict] = deque(
            maxlen=settings.MAX_CONVERSATION_TURNS
        )

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        sql_queries: list[str] | None = None,
        confidence: float = 0.0,
    ):
        self._history.append({
            "turn": len(self._history) + 1,
            "query": user_message,
            "sql": sql_queries or [],
            "narrative": assistant_response[:500],  # truncated
            "confidence": confidence,
        })

    def get_context_string(self) -> str:
        """Serialize history for LLM prompt injection."""
        if not self._history:
            return "No previous conversation."

        lines = []
        for turn in self._history:
            lines.append(
                f"Turn {turn['turn']}: User asked: \"{turn['query']}\" → "
                f"Answered with confidence {turn['confidence']:.2f}"
            )
        return "\n".join(lines)

    def get_last_sql(self) -> list[str]:
        if self._history:
            return self._history[-1].get("sql", [])
        return []

    def clear(self):
        self._history.clear()

