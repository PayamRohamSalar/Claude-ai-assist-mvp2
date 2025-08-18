# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\shared_utils\__init__.py

"""
Legal Assistant AI - Shared Utilities Package
Provides common utilities for the Legal Assistant AI system
"""

__version__ = "1.0.0"
__author__ = "Claude AI Assistant"

# Import core modules
from .constants import *
from .logger import get_logger, log_info, log_warning, log_error, log_critical
from .config_manager import get_config, get_config_manager, get_config_value
from .file_utils import (
    read_document, get_document_reader, get_file_manager, 
    create_directory, copy_file, get_file_info, validate_file_type,
    FileInfo, DocumentReader, FileManager
)

# Re-export commonly used classes and functions
__all__ = [
    # Constants
    'PROJECT_NAME', 'DocumentType', 'ApprovalAuthority', 'Messages',
    'PromptTemplates', 'PERSIAN_DIGITS', 'ENGLISH_DIGITS',
    
    # Logger functions
    'get_logger', 'log_info', 'log_warning', 'log_error', 'log_critical',
    
    # Config functions  
    'get_config', 'get_config_manager', 'get_config_value',
    
    # File utility functions
    'read_document', 'get_document_reader', 'get_file_manager',
    'create_directory', 'copy_file', 'get_file_info', 'validate_file_type',
    
    # File utility classes
    'FileInfo', 'DocumentReader', 'FileManager',
    
    # Utility functions
    'persian_to_english_digits', 'english_to_persian_digits',
    'get_section_name', 'validate_file_extension'
]