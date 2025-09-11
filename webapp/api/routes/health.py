"""
Health check API routes for the Smart Legal Assistant.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from webapp.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    timestamp: datetime
    version: str = "0.1.0"


class DetailedHealthResponse(HealthResponse):
    """Detailed health check response with system info."""
    components: dict = {}


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="سامانه پاسخگوی حقوقی هوشمند فعال است",
        timestamp=datetime.now()
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with system components status."""
    try:
        # Import here to avoid circular imports
        from webapp.services.rag_service import get_rag_service
        from webapp.core.config import get_settings
        
        settings = get_settings()
        rag_service = get_rag_service()
        
        # Check RAG service availability
        engine_ready = False
        engine_error = None
        base_url = "unknown"
        model = "unknown"
        vector_store = "unknown"
        db_path = "unknown"
        
        try:
            engine_ready = rag_service.is_available()
            
            # Get effective configuration from engine if available
            if hasattr(rag_service, '_engine') and rag_service._engine:
                if hasattr(rag_service._engine, 'llm_client'):
                    base_url = getattr(rag_service._engine.llm_client, 'base_url', 'unknown')
                    model = getattr(rag_service._engine.llm_client, 'model', 'unknown')
                
                if hasattr(rag_service._engine, 'vector_store') and rag_service._engine.vector_store:
                    vector_store = rag_service._engine.vector_store.get('type', 'unknown')
                    
                if hasattr(rag_service._engine, 'config'):
                    db_path = rag_service._engine.config.get('database_path', 'unknown')
                    
        except Exception as e:
            engine_error = str(e)
            logger.warning(f"RAG engine health check failed: {e}")
        
        # Determine component statuses
        components = {
            "api": "healthy",
            "rag_engine": "healthy" if engine_ready else "unhealthy",
            "vector_store": vector_store,
            "llm_model": model,
            "base_url": base_url,
            "database": db_path,
            "config_path": settings.RAG_CONFIG_PATH
        }
        
        if engine_error:
            components["engine_error"] = engine_error
        
        overall_status = "healthy" if engine_ready else "degraded"
        message = "تمام اجزای سامانه در وضعیت مطلوب قرار دارند" if engine_ready else "برخی از اجزای سامانه مشکل دارند"
        
        return DetailedHealthResponse(
            status=overall_status,
            message=message,
            timestamp=datetime.now(),
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="برخی از اجزای سامانه در دسترس نیستند"
        )
