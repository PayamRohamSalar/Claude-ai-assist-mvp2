"""
Health check API routes for the Smart Legal Assistant.
"""

from datetime import datetime, timezone
from fastapi import APIRouter

from webapp.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/")
async def health_check():
    """Return structured health diagnostics for the API and RAG engine."""
    try:
        # Import here to avoid circular imports
        from webapp.services.rag_service import get_rag_service
        from webapp.core.config import debug_effective_llm, get_settings

        settings = get_settings()
        rag_service = get_rag_service()

        # Gather diagnostics from service
        svc_diag = rag_service.get_engine_diagnostics() if hasattr(rag_service, 'get_engine_diagnostics') else {}
        engine_ready = bool(svc_diag.get("engine_ready", False))
        model_from_engine = svc_diag.get("model")
        base_url_from_engine = svc_diag.get("base_url")
        vector_store = svc_diag.get("vector_store") or "none"
        db_path = svc_diag.get("db_path")

        # Compute effective LLM settings (ENV > config > default)
        llm_debug = debug_effective_llm()
        effective_model = llm_debug.get("model") or model_from_engine
        effective_base_url = llm_debug.get("base_url") or base_url_from_engine

        status = "ok" if engine_ready else "down"
        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            "status": status,
            "engine_ready": engine_ready,
            "model": effective_model,
            "base_url": effective_base_url,
            "vector_store": vector_store or "none",
            "db_path": db_path,
            "time": now_iso,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "down",
            "engine_ready": False,
            "model": None,
            "base_url": None,
            "vector_store": "none",
            "db_path": None,
            "time": datetime.now(timezone.utc).isoformat(),
            "message": "سامانه در دسترس نیست. لطفاً اتصال به سرویس‌ها و فایل تنظیمات را بررسی کنید.",
        }
