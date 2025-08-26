"""
Phase 2 Database Package

This package handles database operations for the Legal Assistant AI project,
including schema creation, data import, database management, and CLI interface.

Modules:
    database_creator: Database initialization and schema management
    data_importer: JSON document import and processing
    cli: Command-line interface for database operations
    schema.sql: SQLite schema definition with FTS5 support
"""

from .database_creator import init_database, check_database_health, DatabaseCreationError
from .data_importer import import_documents, compute_document_uid, ImportStats, DocumentImportError

__all__ = [
    'init_database', 
    'check_database_health', 
    'DatabaseCreationError',
    'import_documents',
    'compute_document_uid',
    'ImportStats',
    'DocumentImportError'
]