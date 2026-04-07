from src.models.query import RouteType
from src.routing.regex_router import RegexRouter
from src.routing.keyword_router import KeywordRouter
from src.routing.llm_router import LLMRouter


class HybridRouter:
    """3-tier query classification: Regex → Keyword → LLM."""

    def __init__(self):
        self._regex = RegexRouter()
        self._keyword = KeywordRouter()
        self._llm = LLMRouter()

    def classify(self, query: str, conversation_context: str = "") -> RouteType:
        # Tier 1: exact patterns
        result = self._regex.classify(query)
        if result is not None:
            return result

        # Tier 2: keyword scoring
        result = self._keyword.classify(query)
        if result is not None:
            return result

        # Tier 3: LLM fallback (with conversation context)
        return self._llm.classify(query, conversation_context=conversation_context)
