from typing import Optional, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    PROJECT_NAME: str = "Research-Mate"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "CHANGE_THIS_TO_A_SECURE_SECRET_KEY"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ENVIRONMENT: str = "development"  # development | production
    AUTO_CREATE_TABLES: bool = True
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000"
    ALLOWED_HOSTS: str = "localhost,127.0.0.1"
    
    # Database
    DATABASE_URL: Optional[str] = None
    DB_USER: Optional[str] = None
    DB_PASS: Optional[str] = None
    DB_NAME: Optional[str] = None
    INSTANCE_CONNECTION_NAME: Optional[str] = None
    USE_CLOUD_SQL_IN_DEV: bool = False
    
    # Gemini (Vertex AI)
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_API_KEY: Optional[str] = None

    # OpenAI-compatible provider (optional fallback/alternative)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    GOOGLE_OAUTH_CLIENT_ID: Optional[str] = None

    # Workflow controls
    USE_LANGGRAPH: bool = True
    MAX_REPORT_REVISIONS: int = 2
    TEXTBOOK_DATA_DIR: str = "app/data/textbook"

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"development", "production"}:
            raise ValueError("ENVIRONMENT must be 'development' or 'production'")
        return normalized

    @property
    def cors_allow_origins_list(self) -> List[str]:
        value = (self.CORS_ALLOW_ORIGINS or "").strip()
        if not value:
            return []
        if value == "*":
            return ["*"]
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @property
    def allowed_hosts_list(self) -> List[str]:
        value = (self.ALLOWED_HOSTS or "").strip()
        if not value:
            return ["*"]
        if value == "*":
            return ["*"]
        return [host.strip() for host in value.split(",") if host.strip()]

settings = Settings()
