import logging
from pydantic import BaseModel
from config.settings import settings

logger = logging.getLogger(__name__)

_LLM_TIMEOUT = 30.0  # seconds


class LLMClient:
    """Unified LLM interface."""

    def generate(
        self,
        prompt: str,
        system: str = None,
        json_mode: bool = False,
        model: str = None,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)  # rough estimate


class GroqClient(LLMClient):
    def __init__(self):
        from groq import Groq
        self._client = Groq(api_key=settings.GROQ_API_KEY, timeout=_LLM_TIMEOUT)

    def generate(
        self,
        prompt: str,
        system: str = None,
        json_mode: bool = False,
        model: str = None,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model = model or settings.GROQ_MODEL

        # Build response_format: structured schema > json_mode > None
        response_format = None
        if response_schema is not None:
            response_format = {
                "type": "json_object",
                "schema": response_schema.model_json_schema(),
            }
        elif json_mode:
            response_format = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
        )
        return response.choices[0].message.content


_llm_instance = None


def get_llm() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://console.groq.com"
            )
        _llm_instance = GroqClient()
    return _llm_instance


def reset_llm():
    """Clear the cached LLM instance."""
    global _llm_instance
    _llm_instance = None
