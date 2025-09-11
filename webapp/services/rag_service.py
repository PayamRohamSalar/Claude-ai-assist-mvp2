"""
RAG Service for the Smart Legal Assistant Web UI.
Handles loading and interfacing with the Legal RAG Engine.
"""

import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path

from webapp.core.config import get_settings
from webapp.core.logging import get_logger

logger = get_logger(__name__)


class ServiceError(Exception):
    """Custom exception for service-level errors with Persian user messages."""
    
    def __init__(self, user_message: str, technical_details: str = None, trace_id: str = None):
        """
        Initialize ServiceError with user-friendly Persian message.
        
        Args:
            user_message: Persian message for end users
            technical_details: Technical details for logging
            trace_id: Trace ID for debugging
        """
        self.user_message = user_message
        self.technical_details = technical_details
        self.trace_id = trace_id or str(uuid.uuid4())
        
        super().__init__(user_message)


class RAGService:
    """Service for handling RAG engine operations with proper error handling."""
    
    def __init__(self):
        """Initialize RAG service with lazy loading."""
        self._engine = None
        self._settings = get_settings()
        self._engine_loaded = False
        self._engine_ready = False
        self._last_error = None
    
    def _load_engine(self) -> None:
        """Load the Legal RAG Engine once."""
        if self._engine_loaded:
            return
            
        trace_id = str(uuid.uuid4())
        
        try:
            logger.info(f"[{trace_id}] Loading RAG engine from: {self._settings.RAG_CONFIG_PATH}")
            
            # Check if config file exists
            config_path = Path(self._settings.RAG_CONFIG_PATH)
            if not config_path.exists():
                raise ServiceError(
                    user_message="خطا در پیکربندی: فایل تنظیمات موتور جستجو یافت نشد. لطفاً مدیر سیستم را در جریان قرار دهید.",
                    technical_details=f"RAG config file not found at: {config_path}",
                    trace_id=trace_id
                )
            
            # Import and initialize RAG engine
            try:
                from phase_4_llm_rag.rag_engine import LegalRAGEngine
                self._engine = LegalRAGEngine()
                logger.info(f"[{trace_id}] RAG engine loaded successfully")
                
                # Test if engine is ready with a lightweight ping
                self._test_engine_readiness(trace_id)
                
            except ImportError as e:
                raise ServiceError(
                    user_message="خطا در بارگذاری: موتور جستجو در دسترس نیست. لطفاً نصب سیستم را بررسی کنید.",
                    technical_details=f"Failed to import LegalRAGEngine: {str(e)}",
                    trace_id=trace_id
                )
            
            except Exception as e:
                # Check for specific error types that indicate missing components
                error_msg = str(e).lower()
                
                if "vector" in error_msg or "faiss" in error_msg or "index" in error_msg:
                    raise ServiceError(
                        user_message="خطا در بازیابی: بردارها در دسترس نیستند. لطفاً فازهای ۳ و ۴ را بررسی کنید.",
                        technical_details=f"Vector store initialization failed: {str(e)}",
                        trace_id=trace_id
                    )
                elif "model" in error_msg or "llm" in error_msg or "connection" in error_msg:
                    raise ServiceError(
                        user_message="خطا در اتصال: مدل زبانی در دسترس نیست. لطفاً اتصال به سرویس Ollama را بررسی کنید.",
                        technical_details=f"LLM connection failed: {str(e)}",
                        trace_id=trace_id
                    )
                else:
                    raise ServiceError(
                        user_message="خطا در راه‌اندازی: سیستم پاسخگویی آماده نیست. لطفاً با پشتیبانی تماس بگیرید.",
                        technical_details=f"RAG engine initialization failed: {str(e)}",
                        trace_id=trace_id
                    )
            
            self._engine_loaded = True
            
        except ServiceError:
            # Re-raise ServiceError as-is but cache error info
            self._last_error = str(e)
            raise
        except Exception as e:
            logger.error(f"[{trace_id}] Unexpected error loading RAG engine: {str(e)}")
            self._last_error = str(e)
            raise ServiceError(
                user_message="خطای غیرمنتظره در راه‌اندازی سیستم. لطفاً مجدداً تلاش کنید.",
                technical_details=f"Unexpected error: {str(e)}",
                trace_id=trace_id
            )
    
    def _test_engine_readiness(self, trace_id: str) -> None:
        """Test if the RAG engine is ready to process queries."""
        try:
            # Test if LLM client can ping Ollama
            if hasattr(self._engine, 'llm_client') and hasattr(self._engine.llm_client, 'ping'):
                logger.info(f"[{trace_id}] Testing LLM client connectivity...")
                if not self._engine.llm_client.ping():
                    raise ServiceError(
                        user_message="خطا در تولید پاسخ: امکان اتصال به سرور Ollama وجود ندارد",
                        technical_details=f"Ollama ping failed for base_url: {getattr(self._engine.llm_client, 'base_url', 'unknown')}",
                        trace_id=trace_id
                    )
                logger.info(f"[{trace_id}] ✅ LLM client connectivity test passed")
            
            # Test if vector store is accessible
            if hasattr(self._engine, 'vector_store') and self._engine.vector_store:
                logger.info(f"[{trace_id}] Vector store loaded: {self._engine.vector_store.get('type', 'unknown')}")
            
            self._engine_ready = True
            logger.info(f"[{trace_id}] ✅ RAG engine readiness check passed")
            
        except ServiceError:
            raise
        except Exception as e:
            logger.warning(f"[{trace_id}] Engine readiness test failed: {e}")
            raise ServiceError(
                user_message="خطا در آماده‌سازی سیستم: سرویس‌های مورد نیاز در دسترس نیستند",
                technical_details=f"Engine readiness test failed: {str(e)}",
                trace_id=trace_id
            )
    
    def answer(self, question: str, top_k: int = 5, template: str = "default") -> Dict[str, Any]:
        """
        Get answer for a legal question using the RAG engine.
        
        Args:
            question: The legal question to answer
            top_k: Number of documents to retrieve (default: 5)
            template: Template type for answer formatting (default: "default")
            
        Returns:
            Dictionary with answer and normalized citations
            
        Raises:
            ServiceError: When engine is unavailable or processing fails
        """
        trace_id = str(uuid.uuid4())
        import time
        start_time = time.time()
        
        try:
            # Ensure engine is loaded and ready
            self._load_engine()
            
            if self._engine is None:
                raise ServiceError(
                    user_message="خطای داخلی: موتور جستجو در دسترس نیست.",
                    technical_details="RAG engine is None after loading",
                    trace_id=trace_id
                )
            
            if not self._engine_ready:
                raise ServiceError(
                    user_message="خطای سیستم: موتور جستجو آماده نیست. لطفاً با پشتیبانی تماس بگیرید.",
                    technical_details=f"RAG engine not ready. Last error: {self._last_error}",
                    trace_id=trace_id
                )
            
            logger.info(f"[{trace_id}] Processing question with {top_k} documents, template: {template}")
            logger.debug(f"[{trace_id}] Question: {question[:100]}...")
            
            # Call the RAG engine
            result = self._engine.answer(
                question=question,
                top_k=top_k,
                template_name=template
            )
            
            # Process and normalize the result
            normalized_result = {
                "answer": result.get("answer", "پاسخی دریافت نشد."),
                "citations": self._normalize_citations(result.get("citations", []))
            }
            
            processing_time = time.time() - start_time
            logger.info(f"[{trace_id}] Successfully processed question in {processing_time:.2f}s, got {len(normalized_result['citations'])} citations")
            
            return normalized_result
            
        except ServiceError:
            # Re-raise ServiceError as-is
            raise
        except Exception as e:
            logger.error(f"[{trace_id}] Error processing question: {str(e)}")
            
            # Check for specific runtime errors with better Persian messages
            error_msg = str(e).lower()
            
            if "امکان اتصال به سرور ollama وجود ندارد" in str(e):
                raise ServiceError(
                    user_message="خطا در تولید پاسخ: امکان اتصال به سرور Ollama وجود ندارد",
                    technical_details=f"Ollama connection error: {str(e)}",
                    trace_id=trace_id
                )
            elif "connection" in error_msg or "timeout" in error_msg:
                raise ServiceError(
                    user_message="خطا در ارتباط: اتصال به سرویس‌ها قطع شده است. لطفاً مجدداً تلاش کنید.",
                    technical_details=f"Connection error during processing: {str(e)}",
                    trace_id=trace_id
                )
            elif "memory" in error_msg or "resource" in error_msg:
                raise ServiceError(
                    user_message="خطای منابع: حافظه سیستم کافی نیست. لطفاً چند دقیقه صبر کنید.",
                    technical_details=f"Resource error: {str(e)}",
                    trace_id=trace_id
                )
            else:
                raise ServiceError(
                    user_message="خطا در پردازش: امکان پردازش سوال وجود ندارد. لطفاً مجدداً تلاش کنید.",
                    technical_details=f"Processing error: {str(e)}",
                    trace_id=trace_id
                )
    
    def _normalize_citations(self, engine_citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize citations from engine output to include human-readable titles and links.
        
        Args:
            engine_citations: Raw citations from the RAG engine
            
        Returns:
            List of normalized citation dictionaries
        """
        normalized_citations = []
        
        for citation in engine_citations:
            try:
                # Extract basic citation info
                document_uid = citation.get("document_uid", "")
                article_number = citation.get("article_number", "")
                note_label = citation.get("note_label", "")
                document_title = citation.get("document_title", "")
                
                # Create human-readable title
                title_parts = []
                if document_title:
                    title_parts.append(document_title)
                if article_number:
                    title_parts.append(f"ماده {article_number}")
                if note_label:
                    title_parts.append(f"تبصره {note_label}")
                
                title = " - ".join(title_parts) if title_parts else "سند نامشخص"
                
                # Create client link
                link_parts = []
                if document_uid:
                    link_parts.append(f"/doc/{document_uid}")
                    query_params = []
                    if article_number:
                        query_params.append(f"a={article_number}")
                    if note_label:
                        query_params.append(f"n={note_label}")
                    if query_params:
                        link_parts.append("?" + "&".join(query_params))
                
                link = "".join(link_parts) if link_parts else None
                
                # Create normalized citation
                normalized_citation = {
                    "document_uid": document_uid,
                    "article_number": article_number,
                    "note_label": note_label,
                    "title": title,
                    "link": link
                }
                
                # Add any additional fields from the original citation
                for key, value in citation.items():
                    if key not in ["document_uid", "article_number", "note_label", "document_title"]:
                        normalized_citation[key] = value
                
                normalized_citations.append(normalized_citation)
                
            except Exception as e:
                logger.warning(f"Failed to normalize citation {citation}: {str(e)}")
                # Include the original citation if normalization fails
                citation_copy = citation.copy()
                citation_copy.setdefault("title", "خطا در نمایش منبع")
                citation_copy.setdefault("link", None)
                normalized_citations.append(citation_copy)
        
        return normalized_citations
    
    def is_available(self) -> bool:
        """
        Check if the RAG service is available.
        
        Returns:
            True if service is available, False otherwise
        """
        try:
            self._load_engine()
            return self._engine is not None
        except Exception as e:
            logger.warning(f"RAG service availability check failed: {str(e)}")
            return False


# Global RAG service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get the global RAG service instance (singleton)."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
