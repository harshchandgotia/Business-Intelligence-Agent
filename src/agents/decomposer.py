import json
from src.llm.json_utils import extract_json
from src.agents.base import BaseAgent
from src.models.query import DecompositionResult, SubQuestion
from src.models.schema import DatabaseSchema
from config.settings import settings


class DecomposerAgent(BaseAgent):
    name = "decomposer"

    def _execute(self, question: str, schema: DatabaseSchema) -> dict:
        system = (
            "You decompose complex data questions into independent sub-questions.\n"
            "Each sub-question should be answerable with a single SQL query.\n"
            f"Maximum {settings.MAX_DECOMPOSITION_SUBTASKS} sub-questions.\n"
            "If the question is already simple enough, return it as a single sub-question.\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "sub_questions": [\n'
            '    {"id": 1, "text": "...", "depends_on": []},\n'
            '    {"id": 2, "text": "...", "depends_on": [1]}\n'
            "  ],\n"
            '  "reasoning": "..."\n'
            "}"
        )

        prompt = (
            f"Schema:\n{schema.to_prompt_string()}\n\n"
            f"Question: {question}\n\n"
            "Decompose this into sub-questions."
        )

        response = self._llm.generate(prompt, system=system, json_mode=True)
        parsed = extract_json(response, fallback={
            "sub_questions": [{"id": 1, "text": question, "depends_on": []}],
            "reasoning": "Fallback: JSON parse failed, treating as single question",
        })

        sub_questions = [SubQuestion(**sq) for sq in parsed["sub_questions"]]

        return {
            "decomposition": DecompositionResult(
                original_query=question,
                sub_questions=sub_questions,
                reasoning=parsed.get("reasoning", ""),
            )
        }
