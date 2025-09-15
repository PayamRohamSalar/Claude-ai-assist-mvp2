"""
Question-Answering API routes for the Smart Legal Assistant.
"""

import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from webapp.core.logging import get_logger

logger = get_logger(__name__)

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
    title: Optional[str] = None  # Alias for document_title for frontend compatibility
    document_uid: Optional[str] = None
    article_number: Optional[str] = None
    note_label: Optional[str] = None


class RetrievedChunk(BaseModel):
    """Information about a retrieved chunk."""
    document_title: str = Field(..., description="عنوان سند")
    document_uid: str = Field(..., description="شناسه سند")
    article_number: Optional[str] = Field(default="", description="شماره ماده")
    note_label: Optional[str] = Field(default="", description="برچسب تبصره")
    text: str = Field(..., description="متن بخش بازیابی شده")
    similarity_score: Optional[float] = Field(default=None, description="امتیاز شباهت")


class QuestionResponse(BaseModel):
    """Response model for legal questions."""
    answer: str = Field(..., description="پاسخ به زبان فارسی")
    citations: List[Citation] = Field(default=[], description="منابع و استنادها")
    retrieved_chunks: int = Field(..., description="تعداد اسناد بازیابی شده")
    chunks_data: List[RetrievedChunk] = Field(default=[], description="داده‌های بخش‌های بازیابی شده")
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
        
        # Import RAG service
        from webapp.services.rag_service import get_rag_service, ServiceError
        
        import time
        start_time = time.time()
        
        # Get RAG service and process question
        rag_service = get_rag_service()
        result = rag_service.answer(
            question=request.question,
            top_k=request.top_k,
            template=request.template
        )
        
        processing_time = time.time() - start_time
        
        # Format citations for response model
        citations = []
        for citation in result.get("citations", []):
            title = citation.get("title", "")
            citations.append(Citation(
                document_title=title,  # Use normalized title
                title=title,  # Also set title field for frontend compatibility
                document_uid=citation.get("document_uid", ""),
                article_number=citation.get("article_number", ""),
                note_label=citation.get("note_label", "")
            ))
        
        # Format retrieved chunks for response model
        chunks_data = []
        for chunk in result.get("chunks_data", []):
            chunks_data.append(RetrievedChunk(
                document_title=chunk.get("document_title", ""),
                document_uid=chunk.get("document_uid", ""),
                article_number=chunk.get("article_number", ""),
                note_label=chunk.get("note_label", ""),
                text=chunk.get("text", ""),
                similarity_score=chunk.get("similarity_score")
            ))
        
        response = QuestionResponse(
            answer=result.get("answer", "پاسخی دریافت نشد."),
            citations=citations,
            retrieved_chunks=len(result.get("citations", [])),  # Use citation count
            chunks_data=chunks_data,
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
        
    except ServiceError as e:
        logger.error(f"Service error processing request {request_id} [trace_id: {e.trace_id}]: {e.technical_details}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "خطا در سرویس",
                "message": e.user_message,
                "request_id": request_id,
                "trace_id": e.trace_id
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error processing request {request_id}: {e}")
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
