from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Configuration
    APP_NAME: str = "inu-agent-core"
    APP_ENV: str = "development"
    APP_PORT: int = 8000
    DEBUG: bool = False

    # GPU / LLM Inference Cluster (LiteLLM / vLLM - OpenAI-Compatible)
    LLM_BASE_URL: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible LiteLLM/vLLM base URL",
    )
    LLM_API_KEY: str = Field(
        default="not-set",
        description="LiteLLM or vLLM API key",
    )
    LLM_MODEL_NAME: str = Field(
        default="gemma-27b",
        description="Default served model name",
    )
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT_SECONDS: float = 60.0

    # Redis Cache & Session
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # Internal Service Endpoints
    INU_PORTAL_SERVER_URL: str = "http://localhost:8080"
    INU_INTERNAL_S2S_SECRET: str = Field(
        default="",
        description="Shared secret for internal service-to-service communication",
    )

    # Security & CORS
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]


settings = Settings()
