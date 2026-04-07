"""Resilient JSON extraction from LLM responses.

LLMs frequently return JSON wrapped in markdown fences, with preamble text,
or with minor formatting issues. This module handles all those cases.
"""
import logging
import json_repair

logger = logging.getLogger(__name__)


def extract_json(text: str, fallback: dict | None = None) -> dict:
    """Extract and parse a JSON object from an LLM response.

    Uses json_repair to handle markdown fences, missing braces, trailing commas,
    and other common LLM output issues.

    Args:
        text: Raw LLM response text.
        fallback: Dict to return if parsing fails. Defaults to empty dict.

    Returns:
        Parsed dict from the JSON content.
    """
    if fallback is None:
        fallback = {}

    if not text or not text.strip():
        logger.warning("extract_json: received empty text")
        return fallback

    text = text.strip()

    try:
        # json_repair handles markdown fences and formatting errors automatically
        result = json_repair.loads(text)
        if isinstance(result, dict):
            return result
    except Exception as e:
        logger.warning("extract_json: json_repair failed (%s). Input: %.200s", e, text)

    return fallback
