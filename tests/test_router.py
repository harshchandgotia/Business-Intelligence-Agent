import pytest
import json
from src.routing.router import HybridRouter
from src.models.query import RouteType


@pytest.fixture
def router():
    return HybridRouter()


@pytest.fixture
def test_queries():
    with open("tests/fixtures/test_queries.json") as f:
        return json.load(f)


def test_simple_queries_routed_correctly(router, test_queries):
    simple = [q for q in test_queries if q["expected_route"] == "simple"]
    for q in simple:
        result = router.classify(q["query"])
        assert result == RouteType.SIMPLE, (
            f"Query '{q['query']}' expected SIMPLE, got {result}"
        )


def test_analytical_queries_routed_correctly(router, test_queries):
    analytical = [q for q in test_queries if q["expected_route"] == "analytical"]
    for q in analytical:
        result = router.classify(q["query"])
        assert result == RouteType.ANALYTICAL, (
            f"Query '{q['query']}' expected ANALYTICAL, got {result}"
        )


def test_meta_queries_routed_correctly(router, test_queries):
    meta = [q for q in test_queries if q["expected_route"] == "meta"]
    for q in meta:
        result = router.classify(q["query"])
        assert result == RouteType.META, (
            f"Query '{q['query']}' expected META, got {result}"
        )
