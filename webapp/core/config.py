"""
Core configuration settings for the Smart Legal Assistant Web UI.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Core application settings
    APP_ENV: str = Field(default="dev", description="Application environment")
    AUTH_TOKEN: Optional[str] = Field(default=None, description="Optional authentication token")
    RAG_CONFIG_PATH: str = Field(
        default="phase_4_llm_rag/Rag_config.json", 
        description="Path to RAG engine configuration file"
    )
    
    # Additional application settings (for compatibility with existing app.py)
    APP_NAME: str = "Smart Legal Assistant - Web UI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    STATIC_DIR: str = "webapp/static"
    TEMPLATES_DIR: str = "webapp/templates"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000"
    ]
    
    @field_validator('RAG_CONFIG_PATH')
    @classmethod
    def validate_rag_config_path(cls, v: str) -> str:
        """Validate that RAG config path exists, log warning if missing."""
        if v:
            config_path = Path(v)
            if not config_path.exists():
                logger.warning(
                    f"RAG config file not found: {v}. "
                    "The RAG service may not function properly until this file is available."
                )
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()


# Global settings instance for backward compatibility
settings = get_settings()
