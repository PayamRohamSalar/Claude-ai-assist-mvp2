"""
Core configuration settings for the Smart Legal Assistant Web UI.
"""

import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    APP_ENV: str = "development"
    APP_NAME: str = "Smart Legal Assistant - Web UI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # RAG Engine settings
    RAG_CONFIG_PATH: str = "phase_4_llm_rag/Rag_config.json"
    
    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # For development frontends
    ]
    
    # Static and template paths
    STATIC_DIR: str = "webapp/static"
    TEMPLATES_DIR: str = "webapp/templates"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
