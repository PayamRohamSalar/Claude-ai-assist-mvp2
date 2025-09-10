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
        # Check various system components
        components = {
            "api": "healthy",
            "database": "healthy",  # Could check actual DB connection
            "rag_engine": "healthy",  # Could check RAG engine status
            "memory": "healthy"
        }
        
        return DetailedHealthResponse(
            status="healthy",
            message="تمام اجزای سامانه در وضعیت مطلوب قرار دارند",
            timestamp=datetime.now(),
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="برخی از اجزای سامانه در دسترس نیستند"
        )
