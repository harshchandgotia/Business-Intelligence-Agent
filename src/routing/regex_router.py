import re
from src.models.query import RouteType


class RegexRouter:
    SIMPLE_PATTERNS = [
        r"^(total|sum|count|average|avg|max|min)\s+(of\s+)?\w+",
        r"^how many\s+",
        r"^(show|list|get)\s+(me\s+)?(all|the)\s+\w+",
        r"^what (is|are) the (total|top|bottom)\s+",
        r"^top \d+\s+",
    ]

    META_PATTERNS = [
        r"^what (tables|columns|data)",
        r"^(describe|show)\s+(the\s+)?(schema|structure|tables)",
        r"^how much data",
        r"^what does .+ (mean|contain|look like)",
    ]

    def classify(self, query: str) -> RouteType | None:
        q = query.lower().strip()

        for pattern in self.META_PATTERNS:
            if re.match(pattern, q):
                return RouteType.META

        for pattern in self.SIMPLE_PATTERNS:
            if re.match(pattern, q):
                return RouteType.SIMPLE

        return None
