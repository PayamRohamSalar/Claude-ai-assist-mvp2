"""
Logging configuration for the Smart Legal Assistant Web UI.
"""

import logging
import os
import sys
from typing import Dict, Optional


# Global registry to track configured loggers
_configured_loggers: Dict[str, logging.Logger] = {}
_logging_configured = False


def configure_logging() -> None:
    """Configure the root logging settings with console handler and ISO timestamps."""
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Get log level from environment variable (default: INFO)
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Map string to logging level
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = log_level_map.get(log_level_str, logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter with ISO timestamp format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'  # ISO 8601 format
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is configured
    configure_logging()
    
    # Check if logger is already configured
    if name in _configured_loggers:
        return _configured_loggers[name]
    
    # Create new logger
    logger = logging.getLogger(name)
    
    # Store in registry
    _configured_loggers[name] = logger
    
    return logger
