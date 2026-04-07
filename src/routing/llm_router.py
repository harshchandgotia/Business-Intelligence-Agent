from src.llm.factory import get_llm
from src.models.query import RouteType


class LLMRouter:
    SYSTEM = (
        "Classify the user's data question into exactly one category.\n"
        "Respond with ONLY one word: SIMPLE, ANALYTICAL, META, or CLARIFICATION.\n\n"
        "SIMPLE: direct lookups, aggregations, top-N, counts.\n"
        "ANALYTICAL: trends, comparisons, why-questions, multi-step analysis.\n"
        "META: questions about the data structure, schema, columns.\n"
        "CLARIFICATION: too vague or ambiguous to answer."
    )

    # Ordered by priority (most specific first)
    _KEYWORDS = [
        ("CLARIFICATION", RouteType.CLARIFICATION),
        ("ANALYTICAL", RouteType.ANALYTICAL),
        ("SIMPLE", RouteType.SIMPLE),
        ("META", RouteType.META),
    ]

    def classify(self, query: str, conversation_context: str = "") -> RouteType:
        llm = get_llm()
        prompt = query
        if conversation_context and conversation_context != "No previous conversation.":
            prompt = f"Conversation context:\n{conversation_context}\n\nCurrent question: {query}"
        response = llm.generate(prompt, system=self.SYSTEM).strip().upper()

        # Try exact match first
        mapping = {
            "SIMPLE": RouteType.SIMPLE,
            "ANALYTICAL": RouteType.ANALYTICAL,
            "META": RouteType.META,
            "CLARIFICATION": RouteType.CLARIFICATION,
        }
        if response in mapping:
            return mapping[response]

        # Fallback: search for keyword anywhere in the response
        for keyword, route_type in self._KEYWORDS:
            if keyword in response:
                return route_type

        return RouteType.ANALYTICAL  # safe default

