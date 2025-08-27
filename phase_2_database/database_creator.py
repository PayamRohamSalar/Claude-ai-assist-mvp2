"""
Database Creator Module for Legal Assistant AI (Phase 2)

This module handles the creation and initialization of the SQLite database
for storing Persian legal documents with full-text search capabilities.

Author: Legal Assistant AI Team
Version: 2.0
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
from typing import Optional

# Import shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared_utils.config_manager import get_config
from shared_utils.logger import get_logger


class DatabaseCreationError(Exception):
    """Custom exception for database creation errors."""
    pass


def init_database(db_path: Optional[str] = None, recreate: bool = False) -> sqlite3.Connection:
    """
    Initialize the Legal Assistant AI database with the defined schema.
    
    This function creates and initializes the SQLite database using the schema
    defined in schema.sql. It supports full-text search with Persian language
    support and maintains referential integrity.
    
    Args:
        db_path (Optional[str]): Path to the database file. If None, uses
                                configuration default or environment variable.
        recreate (bool): If True, drop existing database and recreate it.
    
    Returns:
        sqlite3.Connection: Active database connection with proper configuration.
    
    Raises:
        DatabaseCreationError: If database initialization fails.
        FileNotFoundError: If schema.sql file is not found.
        sqlite3.Error: If SQLite operations fail.
    """
    logger = get_logger(__name__)
    
    try:
        # Determine database path
        if db_path is None:
            # Check environment variable first
            db_path = os.environ.get('LEGAL_AI_DB_PATH')
            
            # Fall back to configuration
            if db_path is None:
                try:
                    config = get_config()
                    db_path = config.get('database', {}).get('path', 'data/db/legal_assistant.db')
                except Exception as e:
                    logger.warning(f"Could not load config, using default path: {e}")
                    db_path = 'data/db/legal_assistant.db'
        
        # Convert to absolute path and ensure directory exists
        db_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(db_path)
        
        logger.info(f"Initializing database at: {db_path}")
        logger.info(f"درحال راه‌اندازی پایگاه داده در مسیر: {db_path}")
        
        # Create directory if it doesn't exist
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
            logger.info(f"پوشه پایگاه داده ایجاد شد: {db_dir}")
        
        # Handle recreate flag
        if recreate and os.path.exists(db_path):
            logger.info("Recreate flag set, removing existing database...")
            logger.info("پرچم بازسازی تنظیم شده، حذف پایگاه داده موجود...")
            os.remove(db_path)
            logger.info("Existing database removed")
            logger.info("پایگاه داده موجود حذف شد")
        
        # Establish database connection
        logger.info("Establishing database connection...")
        logger.info("درحال برقراری اتصال با پایگاه داده...")
        
        connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=30.0
        )
        
        # Set row factory for easier data access
        connection.row_factory = sqlite3.Row
        
        # Enable WAL mode for better concurrency
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA cache_size=10000")
        connection.execute("PRAGMA temp_store=MEMORY")
        
        logger.info("Database connection established successfully")
        logger.info("اتصال پایگاه داده با موفقیت برقرار شد")
        
        # Load and execute schema
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        logger.info("Reading database schema...")
        logger.info("درحال خواندن طرحواره پایگاه داده...")
        
        with open(schema_path, 'r', encoding='utf-8') as schema_file:
            schema_sql = schema_file.read()
        
        if not schema_sql.strip():
            raise DatabaseCreationError("Schema file is empty")
        
        logger.info("Executing database schema...")
        logger.info("درحال اجرای طرحواره پایگاه داده...")
        
        # Execute schema with transaction
        with connection:
            connection.executescript(schema_sql)
        
        # Verify database structure
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['documents', 'chapters', 'articles', 'notes', 'clauses']
        missing_tables = [table for table in expected_tables if table not in tables]
        
        if missing_tables:
            raise DatabaseCreationError(f"Missing tables after schema execution: {missing_tables}")
        
        # Check FTS tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'")
        fts_tables = [row[0] for row in cursor.fetchall()]
        
        expected_fts_tables = ['documents_fts', 'articles_fts', 'notes_fts', 'clauses_fts']
        missing_fts_tables = [table for table in expected_fts_tables if table not in fts_tables]
        
        if missing_fts_tables:
            logger.warning(f"Missing FTS tables: {missing_fts_tables}")
            logger.warning(f"جداول جستجوی متنی ناقص: {missing_fts_tables}")
        
        logger.info("Database schema executed successfully")
        logger.info("طرحواره پایگاه داده با موفقیت اجرا شد")
        
        # Log database statistics
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        logger.info(f"Database initialized with {doc_count} documents")
        logger.info(f"پایگاه داده با {doc_count} سند راه‌اندازی شد")
        
        logger.info("Database initialization completed successfully")
        logger.info("راه‌اندازی پایگاه داده با موفقیت تکمیل شد")
        
        return connection
        
    except FileNotFoundError as e:
        error_msg = f"Schema file not found: {e}"
        logger.error(error_msg)
        logger.error(f"فایل طرحواره یافت نشد: {e}")
        raise DatabaseCreationError(error_msg) from e
        
    except sqlite3.Error as e:
        error_msg = f"SQLite error during database initialization: {e}"
        logger.error(error_msg)
        logger.error(f"خطای SQLite در راه‌اندازی پایگاه داده: {e}")
        raise DatabaseCreationError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error during database initialization: {e}"
        logger.error(error_msg)
        logger.error(f"خطای غیرمنتظره در راه‌اندازی پایگاه داده: {e}")
        raise DatabaseCreationError(error_msg) from e


def check_database_health(connection: sqlite3.Connection) -> bool:
    """
    Check the health and integrity of the database.
    
    Args:
        connection (sqlite3.Connection): Active database connection.
    
    Returns:
        bool: True if database is healthy, False otherwise.
    """
    logger = get_logger(__name__)
    
    try:
        cursor = connection.cursor()
        
        # Check database integrity
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        
        if integrity_result != 'ok':
            logger.error(f"Database integrity check failed: {integrity_result}")
            logger.error(f"بررسی یکپارچگی پایگاه داده ناموفق: {integrity_result}")
            return False
        
        # Check FTS tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'")
        fts_tables = [row[0] for row in cursor.fetchall()]
        
        for table in fts_tables:
            try:
                cursor.execute(f"INSERT INTO {table}({table}) VALUES('integrity-check')")
                connection.commit()
            except sqlite3.Error as e:
                logger.error(f"FTS table {table} health check failed: {e}")
                logger.error(f"بررسی سلامت جدول FTS {table} ناموفق: {e}")
                return False
        
        logger.info("Database health check passed")
        logger.info("بررسی سلامت پایگاه داده موفق بود")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database health check error: {e}")
        logger.error(f"خطا در بررسی سلامت پایگاه داده: {e}")
        return False


def main() -> None:
    """
    Main entry point for CLI execution.
    
    Accepts optional database path and flags as command line arguments.
    Usage: python database_creator.py [--db-path PATH] [--recreate]
    """
    logger = get_logger(__name__)
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Initialize Legal Assistant AI Database")
        parser.add_argument('--db-path', type=str, help='Path to the database file')
        parser.add_argument('--recreate', action='store_true', 
                          help='Drop existing database and recreate it')
        
        args = parser.parse_args()
        
        if args.db_path:
            logger.info(f"Using database path from command line: {args.db_path}")
            logger.info(f"استفاده از مسیر پایگاه داده از خط فرمان: {args.db_path}")
        
        if args.recreate:
            logger.info("Recreate flag enabled")
            logger.info("پرچم بازسازی فعال است")
        
        # Initialize database
        logger.info("Starting database initialization from CLI...")
        logger.info("شروع راه‌اندازی پایگاه داده از خط فرمان...")
        
        connection = init_database(args.db_path, args.recreate)
        
        # Perform health check
        if check_database_health(connection):
            logger.info("Database is healthy and ready for use")
            logger.info("پایگاه داده سالم و آماده استفاده است")
        else:
            logger.warning("Database health check indicated potential issues")
            logger.warning("بررسی سلامت پایگاه داده مشکلاتی را نشان داد")
        
        # Close connection
        connection.close()
        logger.info("Database connection closed")
        logger.info("اتصال پایگاه داده بسته شد")
        
        print("[SUCCESS] Database initialization completed successfully!")
        print("[SUCCESS] Database initialization completed successfully! (راه‌اندازی پایگاه داده با موفقیت تکمیل شد!)")
        
    except DatabaseCreationError as e:
        logger.error(f"Database creation failed: {e}")
        logger.error(f"ایجاد پایگاه داده ناموفق: {e}")
        print(f"[ERROR] Database initialization failed: {e}")
        print(f"[ERROR] Database initialization failed: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"خطای غیرمنتظره: {e}")
        print(f"[ERROR] Unexpected error: {e}")
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()