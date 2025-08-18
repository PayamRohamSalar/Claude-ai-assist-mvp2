# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\shared_utils\logger.py

"""
Legal Assistant AI - Persian Logging System
Provides Persian-aware logging functionality with proper datetime formatting
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
from enum import Enum

from .constants import (
    LOGS_DIR, LOG_LEVELS, LOG_FORMAT, LOG_DATE_FORMAT,
    PROJECT_NAME, english_to_persian_digits
)


class LogLevel(Enum):
    """Log levels with Persian descriptions"""
    DEBUG = ("DEBUG", "اشکال‌زدایی")
    INFO = ("INFO", "اطلاعات")
    WARNING = ("WARNING", "هشدار") 
    ERROR = ("ERROR", "خطا")
    CRITICAL = ("CRITICAL", "بحرانی")


class PersianFormatter(logging.Formatter):
    """Custom formatter for Persian logging with Jalali date support"""
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        
    def format(self, record):
        """Format log record with Persian elements"""
        
        # Convert timestamp to Persian
        persian_time = self._format_persian_datetime(record.created)
        
        # Create formatted message
        if hasattr(record, 'persian_msg') and record.persian_msg:
            message = record.persian_msg
        else:
            message = record.getMessage()
            
        # Get Persian log level
        level_info = self._get_persian_level(record.levelname)
        
        # Format the final message
        formatted_msg = f"{persian_time} - {record.name} - {level_info} - {message}"
        
        # Add exception info if present
        if record.exc_info:
            formatted_msg += f"\n{self.formatException(record.exc_info)}"
            
        return formatted_msg
    
    def _format_persian_datetime(self, timestamp: float) -> str:
        """Convert Unix timestamp to Persian formatted datetime"""
        dt = datetime.fromtimestamp(timestamp)
        # Format as Persian date/time
        formatted = dt.strftime("%Y/%m/%d %H:%M:%S")
        return english_to_persian_digits(formatted)
    
    def _get_persian_level(self, level_name: str) -> str:
        """Get Persian description for log level"""
        for level in LogLevel:
            if level.value[0] == level_name:
                return f"{level.value[1]} ({level_name})"
        return level_name


class LegalLogger:
    """
    Main logging class for Legal Assistant AI
    Provides Persian-aware logging with structured output
    """
    
    def __init__(self, name: str = PROJECT_NAME, level: str = "INFO"):
        self.name = name
        self.level = level
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the logger"""
        
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(LOG_LEVELS.get(self.level, logging.INFO))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        persian_formatter = PersianFormatter()
        standard_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        
        # Console handler (for development)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(persian_formatter)
        logger.addHandler(console_handler)
        
        # File handler for general logs
        general_log_file = LOGS_DIR / f"{self.name.replace(' ', '_')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(persian_formatter)
        logger.addHandler(file_handler)
        
        # Error file handler (errors only)
        error_log_file = LOGS_DIR / f"{self.name.replace(' ', '_')}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(persian_formatter)
        logger.addHandler(error_handler)
        
        # JSON handler for structured logging
        json_log_file = LOGS_DIR / f"{self.name.replace(' ', '_')}_structured.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(self._get_json_formatter())
        logger.addHandler(json_handler)
        
        return logger
    
    def _get_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging"""
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add Persian message if available
                if hasattr(record, 'persian_msg'):
                    log_entry['persian_message'] = record.persian_msg
                    
                # Add extra fields if available
                if hasattr(record, 'extra_data'):
                    log_entry.update(record.extra_data)
                    
                return json.dumps(log_entry, ensure_ascii=False)
                
        return JSONFormatter()
    
    def debug(self, message: str, persian_msg: str = None, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, persian_msg, **kwargs)
    
    def info(self, message: str, persian_msg: str = None, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, persian_msg, **kwargs)
    
    def warning(self, message: str, persian_msg: str = None, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, persian_msg, **kwargs)
    
    def error(self, message: str, persian_msg: str = None, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, persian_msg, **kwargs)
    
    def critical(self, message: str, persian_msg: str = None, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, persian_msg, **kwargs)
    
    def _log(self, level: int, message: str, persian_msg: str = None, **kwargs):
        """Internal logging method"""
        extra = {'persian_msg': persian_msg}
        if kwargs:
            extra['extra_data'] = kwargs
        self.logger.log(level, message, extra=extra)
    
    def log_document_processing(self, document_name: str, status: str, 
                              details: Dict[str, Any] = None):
        """Specialized logging for document processing"""
        persian_status = self._translate_status(status)
        message = f"Document processing: {document_name} - {status}"
        persian_msg = f"پردازش سند: {document_name} - {persian_status}"
        
        extra_data = {
            'document_name': document_name,
            'processing_status': status,
            'details': details or {}
        }
        
        if status.lower() in ['error', 'failed']:
            self.error(message, persian_msg, **extra_data)
        else:
            self.info(message, persian_msg, **extra_data)
    
    def log_search_query(self, query: str, results_count: int, 
                        processing_time: float):
        """Specialized logging for search operations"""
        message = f"Search query executed: {query[:50]}... - {results_count} results in {processing_time:.2f}s"
        persian_msg = f"جستجو انجام شد: {query[:50]}... - {results_count} نتیجه در {processing_time:.2f} ثانیه"
        
        extra_data = {
            'query': query,
            'results_count': results_count,
            'processing_time': processing_time
        }
        
        self.info(message, persian_msg, **extra_data)
    
    def log_llm_interaction(self, model_name: str, prompt_length: int, 
                           response_length: int, success: bool):
        """Specialized logging for LLM interactions"""
        status = "successful" if success else "failed"
        persian_status = "موفق" if success else "ناموفق"
        
        message = f"LLM interaction with {model_name}: {status}"
        persian_msg = f"تعامل با مدل {model_name}: {persian_status}"
        
        extra_data = {
            'model_name': model_name,
            'prompt_length': prompt_length,
            'response_length': response_length,
            'success': success
        }
        
        if success:
            self.info(message, persian_msg, **extra_data)
        else:
            self.error(message, persian_msg, **extra_data)
    
    def _translate_status(self, status: str) -> str:
        """Translate common status messages to Persian"""
        translations = {
            'started': 'شروع شده',
            'processing': 'در حال پردازش',
            'completed': 'تکمیل شده',
            'success': 'موفق',
            'failed': 'ناموفق',
            'error': 'خطا',
            'warning': 'هشدار',
            'pending': 'در انتظار',
            'cancelled': 'لغو شده'
        }
        return translations.get(status.lower(), status)


# Global logger instance
_global_logger: Optional[LegalLogger] = None

def get_logger(name: str = None, level: str = "INFO") -> LegalLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name (defaults to project name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        LegalLogger instance
    """
    global _global_logger
    
    if _global_logger is None or (name and name != _global_logger.name):
        logger_name = name or PROJECT_NAME
        _global_logger = LegalLogger(logger_name, level)
        
    return _global_logger

def log_system_startup():
    """Log system startup information"""
    logger = get_logger()
    logger.info(
        "Legal Assistant AI system starting up",
        persian_msg="سیستم دستیار حقوقی هوشمند در حال راه‌اندازی",
        version=PROJECT_NAME,
        logs_directory=str(LOGS_DIR)
    )

def log_system_shutdown():
    """Log system shutdown information"""
    logger = get_logger()
    logger.info(
        "Legal Assistant AI system shutting down",
        persian_msg="سیستم دستیار حقوقی هوشمند در حال خاموش شدن"
    )

# Convenience functions for common operations
def log_debug(message: str, persian_msg: str = None, **kwargs):
    """Convenience function for debug logging"""
    get_logger().debug(message, persian_msg, **kwargs)

def log_info(message: str, persian_msg: str = None, **kwargs):
    """Convenience function for info logging"""
    get_logger().info(message, persian_msg, **kwargs)

def log_warning(message: str, persian_msg: str = None, **kwargs):
    """Convenience function for warning logging"""
    get_logger().warning(message, persian_msg, **kwargs)

def log_error(message: str, persian_msg: str = None, **kwargs):
    """Convenience function for error logging"""
    get_logger().error(message, persian_msg, **kwargs)

def log_critical(message: str, persian_msg: str = None, **kwargs):
    """Convenience function for critical logging"""
    get_logger().critical(message, persian_msg, **kwargs)