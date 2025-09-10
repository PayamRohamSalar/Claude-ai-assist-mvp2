"""
Smart Legal Assistant - Main FastAPI Web Application
Provides the main web interface for the legal assistant system.
"""

import logging
import uuid
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import local modules
from webapp.core.config import settings
from webapp.core.logging import get_logger
from webapp.api.routes import health, qa

# Get configured logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"📁 RAG Engine config path: {settings.RAG_CONFIG_PATH}")
    logger.info(f"🌍 Environment: {settings.APP_ENV}")
    logger.info(f"🔧 Debug mode: {settings.DEBUG}")
    logger.info(f"📂 Static files directory: {settings.STATIC_DIR}")
    logger.info(f"📄 Templates directory: {settings.TEMPLATES_DIR}")
    
    # Ensure directories exist
    static_path = Path(settings.STATIC_DIR)
    templates_path = Path(settings.TEMPLATES_DIR)
    
    if not static_path.exists():
        logger.warning(f"Static directory does not exist: {static_path}")
    if not templates_path.exists():
        logger.warning(f"Templates directory does not exist: {templates_path}")
    
    # Log successful startup
    logger.info("✅ سامانه پاسخگوی حقوقی هوشمند با موفقیت راه‌اندازی شد")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Smart Legal Assistant")
    logger.info("✅ سامانه پاسخگوی حقوقی هوشمند با موفقیت خاموش شد")


# Initialize FastAPI application
app = FastAPI(
    title="Smart Legal Assistant - Web UI",
    version="0.1.0",
    description="سامانه پاسخگوی حقوقی هوشمند - رابط کاربری وب",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    health.router,
    prefix="/api/health",
    tags=["health"],
    responses={404: {"description": "Not found"}}
)

app.include_router(
    qa.router,
    prefix="/api",
    tags=["qa"],
    responses={404: {"description": "Not found"}}
)

# Mount static files
if Path(settings.STATIC_DIR).exists():
    app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
    logger.info(f"📁 Static files mounted from: {settings.STATIC_DIR}")
else:
    logger.warning(f"⚠️ Static directory not found: {settings.STATIC_DIR}")

# Setup Jinja2 templates
if Path(settings.TEMPLATES_DIR).exists():
    templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)
    logger.info(f"📄 Templates configured from: {settings.TEMPLATES_DIR}")
else:
    logger.warning(f"⚠️ Templates directory not found: {settings.TEMPLATES_DIR}")
    templates = None


# Global exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions with Persian user-friendly messages."""
    
    trace_id = str(uuid.uuid4())
    
    # Log the full exception with trace ID
    logger.error(
        f"Unhandled exception [trace_id: {trace_id}]: {str(exc)}",
        extra={
            "trace_id": trace_id,
            "path": str(request.url),
            "method": request.method,
            "client": request.client.host if request.client else "unknown",
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "خطای داخلی سرور",
            "message": "متأسفانه خطایی در سرور رخ داده است. لطفاً دوباره تلاش کنید یا با پشتیبانی تماس بگیرید.",
            "trace_id": trace_id
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors with Persian user-friendly messages."""
    
    trace_id = str(uuid.uuid4())
    
    # Log validation error with trace ID
    logger.warning(
        f"Request validation error [trace_id: {trace_id}]: {str(exc)}",
        extra={
            "trace_id": trace_id,
            "path": str(request.url),
            "method": request.method,
            "client": request.client.host if request.client else "unknown",
            "validation_errors": exc.errors()
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "خطا در اعتبارسنجی داده‌ها",
            "message": "داده‌های ارسالی معتبر نیستند. لطفاً ورودی‌های خود را بررسی کنید.",
            "details": exc.errors(),
            "trace_id": trace_id
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions with Persian user-friendly messages."""
    
    trace_id = str(uuid.uuid4())
    
    # Log HTTP exception with trace ID
    logger.warning(
        f"HTTP exception [trace_id: {trace_id}]: {exc.status_code} - {exc.detail}",
        extra={
            "trace_id": trace_id,
            "path": str(request.url),
            "method": request.method,
            "client": request.client.host if request.client else "unknown",
            "status_code": exc.status_code
        }
    )
    
    # Map common HTTP status codes to Persian messages
    persian_messages = {
        404: "صفحه یا منبع مورد نظر یافت نشد.",
        403: "شما دسترسی لازم برای این عملیات را ندارید.",
        401: "لطفاً ابتدا وارد سیستم شوید.",
        400: "درخواست نامعتبر است.",
        405: "روش درخواست مجاز نیست.",
        429: "تعداد درخواست‌های شما از حد مجاز تجاوز کرده است.",
        503: "سرویس در حال حاضر در دسترس نیست."
    }
    
    persian_message = persian_messages.get(
        exc.status_code, 
        "خطایی در پردازش درخواست شما رخ داده است."
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"خطای HTTP {exc.status_code}",
            "message": persian_message,
            "trace_id": trace_id
        }
    )


# Root route
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main index page using Jinja2Templates."""
    
    if templates is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "خطای پیکربندی",
                "message": "قالب‌های صفحه پیکربندی نشده‌اند."
            }
        )
    
    try:
        context = {
            "request": request,
            "title": "دستیار حقوقی هوشمند",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV
        }
        
        return templates.TemplateResponse("index.html", context)
        
    except Exception as e:
        logger.error(f"Error rendering index template: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "خطا در نمایش صفحه",
                "message": "متأسفانه صفحه قابل نمایش نیست. لطفاً دوباره تلاش کنید."
            }
        )


# Additional utility endpoints
@app.get("/api/info")
async def app_info():
    """Get application information."""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "debug": settings.DEBUG,
        "persian_name": "سامانه پاسخگوی حقوقی هوشمند"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "webapp.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
