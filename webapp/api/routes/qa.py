"""
Question-Answering API routes for the Smart Legal Assistant.
"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

logger = logging.getLogger(__name__)

router = APIRouter()


class QuestionRequest(BaseModel):
    """Request model for legal questions."""
    question: str = Field(..., min_length=5, max_length=1000, description="سوال حقوقی به زبان فارسی")
    template: str = Field(default="default", description="نوع قالب پاسخ")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="تعداد اسناد مرتبط")
    filters: Optional[Dict[str, str]] = Field(default=None, description="فیلترهای جستجو")
    session_id: Optional[str] = Field(default=None, description="شناسه جلسه")


class Citation(BaseModel):
    """Citation information for a legal reference."""
    document_title: Optional[str] = None
    document_uid: Optional[str] = None
    article_number: Optional[str] = None
    note_label: Optional[str] = None


class QuestionResponse(BaseModel):
    """Response model for legal questions."""
    answer: str = Field(..., description="پاسخ به زبان فارسی")
    citations: List[Citation] = Field(default=[], description="منابع و استنادها")
    retrieved_chunks: int = Field(..., description="تعداد اسناد بازیابی شده")
    processing_time: float = Field(..., description="زمان پردازش به ثانیه")
    session_id: str = Field(..., description="شناسه جلسه")
    request_id: str = Field(..., description="شناسه درخواست")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="پیام خطا")
    message: str = Field(..., description="توضیحات خطا")
    request_id: str = Field(..., description="شناسه درخواست")


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks
):
    """Ask a legal question and get an AI-powered answer."""
    request_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing question request {request_id}: {request.question[:50]}...")
        
        # Import RAG engine here to avoid import issues during startup
        from phase_4_llm_rag.rag_engine import LegalRAGEngine
        
        # Initialize RAG engine
        engine = LegalRAGEngine()
        
        import time
        start_time = time.time()
        
        # Get answer from RAG engine
        result = engine.answer(
            question=request.question,
            top_k=request.top_k,
            template_name=request.template,
            filters=request.filters
        )
        
        processing_time = time.time() - start_time
        
        # Format citations
        citations = []
        for citation in result.get("citations", []):
            citations.append(Citation(
                document_title=citation.get("document_title"),
                document_uid=citation.get("document_uid"),
                article_number=citation.get("article_number"),
                note_label=citation.get("note_label")
            ))
        
        response = QuestionResponse(
            answer=result.get("answer", "پاسخی دریافت نشد."),
            citations=citations,
            retrieved_chunks=result.get("retrieved_chunks", 0),
            processing_time=processing_time,
            session_id=session_id,
            request_id=request_id
        )
        
        # Log analytics in background
        background_tasks.add_task(
            log_qa_analytics,
            request_id,
            request.question,
            response.answer,
            processing_time
        )
        
        logger.info(f"Successfully processed request {request_id} in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "خطا در پردازش سوال",
                "message": "متأسفانه در حال حاضر امکان پاسخگویی وجود ندارد. لطفاً مجدداً تلاش کنید.",
                "request_id": request_id
            }
        )


@router.get("/templates")
async def get_question_templates():
    """Get available question templates."""
    return {
        "templates": [
            {
                "name": "default",
                "display_name": "پاسخ عادی",
                "description": "پاسخ مستقیم و کامل به سوال"
            },
            {
                "name": "compare",
                "display_name": "مقایسه",
                "description": "مقایسه بین دو متن یا مفهوم حقوقی"
            },
            {
                "name": "draft",
                "display_name": "پیش‌نویس",
                "description": "تهیه پیش‌نویس متن حقوقی"
            }
        ]
    }


async def log_qa_analytics(request_id: str, question: str, answer: str, processing_time: float):
    """Log analytics data for QA requests (background task)."""
    try:
        # Here you could log to analytics database, metrics system, etc.
        logger.info(f"Analytics: {request_id} | Q_len: {len(question)} | A_len: {len(answer)} | Time: {processing_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to log analytics for {request_id}: {e}")
