"""
Core configuration settings for the Smart Legal Assistant Web UI.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
    
    # Optional LLM-related environment overrides (ENV > config > default)
    OLLAMA_BASE_URL: Optional[str] = Field(default=None, description="Override Ollama base URL")
    OLLAMA_MODEL_NAME: Optional[str] = Field(default=None, description="Override Ollama model name")
    # Note: We intentionally do not add OLLAMA_TIMEOUT as a required setting here;
    # it will still be honored from environment inside effective_llm_settings.
    
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


def _pick_with_source(env_key: str, cfg: Dict[str, Any], cfg_key: str, default: Any) -> Tuple[Any, str]:
    """Pick a value with precedence and report its source.
    
    Precedence: ENV > config > default
    Returns (value, source) where source in {"ENV", "config", "default"}
    """
    env_val = os.getenv(env_key)
    if env_val not in (None, ""):
        return env_val, "ENV"
    if cfg and cfg_key in cfg and cfg[cfg_key] not in (None, ""):
        return cfg[cfg_key], "config"
    return default, "default"


def _safe_int(val: Any, fallback: int) -> int:
    try:
        return int(val)
    except Exception:
        return fallback


def effective_llm_settings(config_dict: Dict[str, Any]) -> Tuple[str, str, int]:
    """Resolve effective LLM base_url, model, and timeout using ENV > config > default.
    
    Args:
        config_dict: Parsed JSON configuration (e.g., Rag_config.json)
    Returns:
        (base_url, model, timeout_seconds)
    """
    cfg_llm = (config_dict or {}).get("llm", {})

    # Defaults align with engine expectations
    default_base_url = "http://localhost:11434"
    default_model = "qwen2.5:7b-instruct"
    default_timeout = 60

    base_url, _ = _pick_with_source("OLLAMA_BASE_URL", cfg_llm, "base_url", default_base_url)
    model, _ = _pick_with_source("OLLAMA_MODEL_NAME", cfg_llm, "model", default_model)
    # Timeout can come from ENV OLLAMA_TIMEOUT or config llm.timeout_s
    timeout_env = os.getenv("OLLAMA_TIMEOUT")
    if timeout_env not in (None, ""):
        timeout = _safe_int(timeout_env, default_timeout)
    else:
        timeout = _safe_int(cfg_llm.get("timeout_s", default_timeout), default_timeout)

    return base_url.rstrip('/'), str(model).strip(), timeout


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()


def _load_config_from_path(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def debug_effective_llm() -> Dict[str, Any]:
    """Return a debug dictionary for health endpoint with effective LLM settings."""
    s = get_settings()
    cfg = _load_config_from_path(s.RAG_CONFIG_PATH) if s.RAG_CONFIG_PATH else {}

    cfg_llm = cfg.get("llm", {})

    base_url, model, timeout = effective_llm_settings(cfg)

    # Determine sources for base_url and model for diagnostics
    base_url_source = "ENV" if s.OLLAMA_BASE_URL not in (None, "") else ("config" if "base_url" in cfg_llm else "default")
    model_source = "ENV" if s.OLLAMA_MODEL_NAME not in (None, "") else ("config" if "model" in cfg_llm else "default")
    timeout_source = "ENV" if os.getenv("OLLAMA_TIMEOUT") not in (None, "") else ("config" if "timeout_s" in cfg_llm else "default")

    return {
        "base_url": base_url,
        "model": model,
        "timeout": timeout,
        "sources": {
            "base_url": base_url_source,
            "model": model_source,
            "timeout": timeout_source,
        },
        "rag_config_present": bool(cfg),
        "rag_config_path": s.RAG_CONFIG_PATH,
    }


# Global settings instance for backward compatibility
settings = get_settings()
