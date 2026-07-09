from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.qwen"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_base_url: str = Field(default="http://localhost:1234/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="lm-studio", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="qwen2.5-7b-instruct", alias="OPENAI_MODEL")
    llm_temperature: float = Field(default=0.35, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1200, alias="LLM_MAX_TOKENS")
    llm_json_retries: int = Field(default=2, alias="LLM_JSON_RETRIES")
    llm_timeout_seconds: float = Field(default=120.0, alias="LLM_TIMEOUT_SECONDS")
    meeting_storage_dir: str = Field(default="./data/meetings", alias="MEETING_STORAGE_DIR")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
