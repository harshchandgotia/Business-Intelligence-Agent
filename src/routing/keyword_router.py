from src.models.query import RouteType


class KeywordRouter:
    # Single-word keywords — matched via set intersection
    ANALYTICAL_WORDS = {
        "compare", "trend", "analyze", "why", "correlation",
        "predict", "forecast", "breakdown", "versus", "vs",
        "relationship", "impact", "cause", "decline", "increase",
        "pattern", "seasonal", "anomaly", "unusual", "spike",
    }

    SIMPLE_WORDS = {
        "total", "sum", "count", "average", "list", "show",
        "get", "top", "bottom", "maximum", "minimum", "latest", "recent",
    }

    # Multi-word phrases — matched via substring search
    ANALYTICAL_PHRASES = [
        "year over year", "month over month", "compared to",
    ]

    SIMPLE_PHRASES = [
        "how many", "what is", "what are", "how much",
    ]

    def classify(self, query: str) -> RouteType | None:
        q_lower = query.lower()
        words = set(q_lower.split())

        analytical_score = len(words & self.ANALYTICAL_WORDS)
        simple_score = len(words & self.SIMPLE_WORDS)

        # Check multi-word phrases against the full query string
        analytical_score += sum(1 for p in self.ANALYTICAL_PHRASES if p in q_lower)
        simple_score += sum(1 for p in self.SIMPLE_PHRASES if p in q_lower)

        if analytical_score >= 2:
            return RouteType.ANALYTICAL
        if simple_score >= 1 and analytical_score == 0:
            return RouteType.SIMPLE

        return None  # pass to LLM
