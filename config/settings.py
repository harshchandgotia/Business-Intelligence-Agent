from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # LLM (Groq only)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Database — prefer DATABASE_URL; fall back to individual fields
    DATABASE_URL: Optional[str] = None
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "bi_agent"
    DB_PASSWORD: str = "password"
    DB_NAME: str = "bi_agent_db"

    @property
    def db_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # Limits
    MAX_QUERY_ROWS: int = 1000
    MAX_CONVERSATION_TURNS: int = 10
    MAX_CRITIQUE_LOOPS: int = 2
    MAX_DECOMPOSITION_SUBTASKS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.4    # below this, ask user for clarification

    # Preprocessing
    NULL_WARNING_THRESHOLD: float = 0.05  # 5% nulls triggers warning
    OUTLIER_ZSCORE: float = 3.0
    FUZZY_MATCH_THRESHOLD: int = 85       # fuzzywuzzy score

    # Paths (resolved from project root)
    PROMPTS_DIR: str = str(_PROJECT_ROOT / "config" / "prompts")

    class Config:
        env_file = str(_PROJECT_ROOT / ".env")
        extra = "ignore"  # ignore leftover env vars (e.g. old GEMINI_* keys)


settings = Settings()

