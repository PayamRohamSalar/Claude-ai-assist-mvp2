#!/usr/bin/env python3
"""
Command-Line Interface for Legal Assistant AI Database (Phase 2)

This module provides CLI commands for database initialization, data import,
and validation of the Legal Assistant AI system.

Commands:
    init-db: Initialize the database schema
    import-data: Import JSON documents into the database
    validate: Validate database contents and show statistics

Author: Legal Assistant AI Team
Version: 2.0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared_utils.logger import get_logger

# Import database modules
try:
    from .database_creator import init_database, check_database_health, DatabaseCreationError
    from .data_importer import import_documents, DocumentImportError
except ImportError:
    from database_creator import init_database, check_database_health, DatabaseCreationError
    from data_importer import import_documents, DocumentImportError


def handle_init_db(args: argparse.Namespace) -> bool:
    """
    Handle the init-db command to initialize the database.
    
    Creates the SQLite database with proper schema, indexes, and FTS tables
    for storing Persian legal documents.
    
    Args:
        args: Parsed command line arguments containing db_path
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    logger = get_logger(__name__)
    
    try:
        print("شروع راه‌اندازی پایگاه داده...")
        print("Initializing database...")
        
        # Initialize database
        connection = init_database(args.db_path)
        
        # Perform health check
        if check_database_health(connection):
            logger.info("Database health check passed")
        else:
            logger.warning("Database health check indicated potential issues")
            print("⚠️  هشدار: بررسی سلامت پایگاه داده مشکلاتی نشان داد")
            
        connection.close()
        
        # Success message
        db_path = args.db_path if args.db_path else "مسیر پیش‌فرض"
        print(f"✅ پایگاه داده با موفقیت راه‌اندازی شد!")
        print(f"✅ Database successfully initialized!")
        print(f"📍 مسیر پایگاه داده: {db_path}")
        
        logger.info("Database initialization completed successfully")
        return True
        
    except DatabaseCreationError as e:
        error_msg = f"خطا در ایجاد پایگاه داده: {e}"
        print(f"❌ {error_msg}")
        print(f"❌ Database creation error: {e}")
        logger.error(f"Database creation failed: {e}")
        return False
        
    except Exception as e:
        error_msg = f"خطای غیرمنتظره: {e}"
        print(f"❌ {error_msg}")
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error during database initialization: {e}")
        return False


def handle_import_data(args: argparse.Namespace) -> bool:
    """
    Handle the import-data command to import JSON documents.
    
    Imports structured JSON documents from Phase 1 processing into
    the SQLite database with proper relationships and metadata.
    
    Args:
        args: Parsed command line arguments containing input_dir and db_path
        
    Returns:
        bool: True if import successful, False otherwise
    """
    logger = get_logger(__name__)
    
    try:
        # Validate input directory
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"❌ پوشه ورودی وجود ندارد: {args.input_dir}")
            print(f"❌ Input directory does not exist: {args.input_dir}")
            return False
            
        if not input_path.is_dir():
            print(f"❌ مسیر ورودی پوشه نیست: {args.input_dir}")
            print(f"❌ Input path is not a directory: {args.input_dir}")
            return False
        
        print("شروع وارد کردن داده‌ها...")
        print("Starting data import...")
        
        # Initialize database connection
        print("راه‌اندازی اتصال پایگاه داده...")
        connection = init_database(args.db_path)
        
        # Import documents
        print(f"وارد کردن اسناد از پوشه: {args.input_dir}")
        print(f"Importing documents from: {args.input_dir}")
        
        results = import_documents(args.input_dir, connection)
        connection.close()
        
        # Display Persian summary
        print("\n" + "="*50)
        print("📊 خلاصه وارد کردن داده‌ها")
        print("📊 Data Import Summary")
        print("="*50)
        
        print(f"📁 تعداد فایل‌های پردازش شده: {results['documents_processed']}")
        print(f"📁 Files processed: {results['documents_processed']}")
        
        print(f"📄 اسناد وارد شده: {results['documents_inserted']}")
        print(f"📄 Documents inserted: {results['documents_inserted']}")
        
        print(f"🔄 اسناد به‌روزرسانی شده: {results['documents_updated']}")
        print(f"🔄 Documents updated: {results['documents_updated']}")
        
        print(f"❌ اسناد ناموفق: {results['documents_failed']}")
        print(f"❌ Documents failed: {results['documents_failed']}")
        
        print(f"📈 درصد موفقیت: {results['success_rate']:.1f}%")
        print(f"📈 Success rate: {results['success_rate']:.1f}%")
        
        print(f"\n📚 ساختار داده‌ها:")
        print(f"📚 Data structure:")
        print(f"  • فصل‌ها: {results['chapters_inserted']}")
        print(f"  • Chapters: {results['chapters_inserted']}")
        print(f"  • مواد: {results['articles_inserted']}")
        print(f"  • Articles: {results['articles_inserted']}")
        print(f"  • تبصره‌ها: {results['notes_inserted']}")
        print(f"  • Notes: {results['notes_inserted']}")
        print(f"  • بندها: {results['clauses_inserted']}")
        print(f"  • Clauses: {results['clauses_inserted']}")
        
        # Show errors if any
        if results['errors']:
            print(f"\n⚠️  خطاها ({len(results['errors'])}):")
            print(f"⚠️  Errors ({len(results['errors'])}):")
            for i, error in enumerate(results['errors'][:3], 1):
                print(f"  {i}. {error}")
            if len(results['errors']) > 3:
                print(f"  ... و {len(results['errors']) - 3} خطای دیگر")
                print(f"  ... and {len(results['errors']) - 3} more errors")
        
        # Final status
        if results['documents_failed'] == 0:
            print(f"\n✅ وارد کردن داده‌ها با موفقیت کامل شد!")
            print(f"✅ Data import completed successfully!")
        else:
            print(f"\n⚠️  وارد کردن داده‌ها با برخی خطاها کامل شد")
            print(f"⚠️  Data import completed with some errors")
        
        logger.info("Data import process completed")
        return results['documents_failed'] == 0
        
    except DocumentImportError as e:
        error_msg = f"خطا در وارد کردن داده‌ها: {e}"
        print(f"❌ {error_msg}")
        print(f"❌ Data import error: {e}")
        logger.error(f"Data import failed: {e}")
        return False
        
    except Exception as e:
        error_msg = f"خطای غیرمنتظره: {e}"
        print(f"❌ {error_msg}")
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error during data import: {e}")
        return False


def handle_validate(args: argparse.Namespace) -> bool:
    """
    Handle the validate command to verify database contents.
    
    Performs validation checks on the database including record counts,
    data integrity, and FTS table consistency.
    
    Args:
        args: Parsed command line arguments containing db_path
        
    Returns:
        bool: True if validation successful, False otherwise
    """
    logger = get_logger(__name__)
    
    try:
        print("شروع اعتبارسنجی پایگاه داده...")
        print("Starting database validation...")
        
        # Initialize database connection
        connection = init_database(args.db_path)
        cursor = connection.cursor()
        
        print("\n" + "="*50)
        print("🔍 نتایج اعتبارسنجی پایگاه داده")
        print("🔍 Database Validation Results")
        print("="*50)
        
        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"📄 تعداد اسناد: {doc_count}")
        print(f"📄 Documents: {doc_count}")
        
        # Count chapters
        cursor.execute("SELECT COUNT(*) FROM chapters")
        chapter_count = cursor.fetchone()[0]
        print(f"📚 تعداد فصل‌ها: {chapter_count}")
        print(f"📚 Chapters: {chapter_count}")
        
        # Count articles
        cursor.execute("SELECT COUNT(*) FROM articles")
        article_count = cursor.fetchone()[0]
        print(f"📝 تعداد مواد: {article_count}")
        print(f"📝 Articles: {article_count}")
        
        # Count notes
        cursor.execute("SELECT COUNT(*) FROM notes")
        note_count = cursor.fetchone()[0]
        print(f"📋 تعداد تبصره‌ها: {note_count}")
        print(f"📋 Notes: {note_count}")
        
        # Count clauses
        cursor.execute("SELECT COUNT(*) FROM clauses")
        clause_count = cursor.fetchone()[0]
        print(f"🔸 تعداد بندها: {clause_count}")
        print(f"🔸 Clauses: {clause_count}")
        
        # Check FTS tables
        print(f"\n🔍 بررسی جداول جستجوی متنی (FTS):")
        print(f"🔍 Full-Text Search (FTS) Tables:")
        
        fts_tables = ['documents_fts', 'articles_fts', 'notes_fts', 'clauses_fts']
        fts_healthy = True
        
        for table in fts_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  • {table}: {count} رکورد")
                print(f"  • {table}: {count} records")
            except Exception as e:
                print(f"  ❌ {table}: خطا - {e}")
                print(f"  ❌ {table}: Error - {e}")
                fts_healthy = False
        
        # Sample document information
        if doc_count > 0:
            print(f"\n📄 نمونه اسناد:")
            print(f"📄 Sample documents:")
            
            cursor.execute("""
                SELECT document_uid, title, document_type 
                FROM documents 
                LIMIT 3
            """)
            
            for i, row in enumerate(cursor.fetchall(), 1):
                title = row[1] if row[1] else "بدون عنوان"
                title = title[:40] + "..." if len(title) > 40 else title
                print(f"  {i}. {row[0]} - {title} ({row[2]})")
        
        # Database integrity check
        print(f"\n🔍 بررسی یکپارچگی پایگاه داده:")
        print(f"🔍 Database integrity check:")
        
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        
        if integrity_result == 'ok':
            print(f"  ✅ یکپارچگی پایگاه داده تأیید شد")
            print(f"  ✅ Database integrity verified")
        else:
            print(f"  ❌ مشکل در یکپارچگی: {integrity_result}")
            print(f"  ❌ Integrity issue: {integrity_result}")
            fts_healthy = False
        
        connection.close()
        
        # Final validation result
        total_records = doc_count + chapter_count + article_count + note_count + clause_count
        
        print(f"\n" + "="*50)
        if total_records > 0 and fts_healthy:
            print(f"✅ اعتبارسنجی موفق - پایگاه داده سالم است")
            print(f"✅ Validation successful - Database is healthy")
            print(f"📊 مجموع رکوردها: {total_records}")
            print(f"📊 Total records: {total_records}")
        elif total_records > 0:
            print(f"⚠️  اعتبارسنجی با هشدار - داده‌ها موجود اما مشکلاتی یافت شد")
            print(f"⚠️  Validation with warnings - Data exists but issues found")
        else:
            print(f"⚠️  پایگاه داده خالی است")
            print(f"⚠️  Database is empty")
        
        logger.info("Database validation completed")
        return total_records > 0 and fts_healthy
        
    except Exception as e:
        error_msg = f"خطا در اعتبارسنجی: {e}"
        print(f"❌ {error_msg}")
        print(f"❌ Validation error: {e}")
        logger.error(f"Database validation failed: {e}")
        return False


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser with subcommands.
    
    Returns:
        argparse.ArgumentParser: Configured parser with all subcommands
    """
    parser = argparse.ArgumentParser(
        prog='legal-ai-db',
        description='Legal AI Database CLI - Interface for Persian legal document database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

  # Initialize database
  python cli.py init-db

  # Initialize with specific path  
  python cli.py init-db --db-path my_database.db

  # Import data
  python cli.py import-data --input-dir data/processed_phase_1

  # Validate database
  python cli.py validate

For help on individual commands, use: python cli.py <command> --help
        """
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        title='Available commands',
        description='Commands for database management',
        help='Use <command> --help for help on individual commands'
    )
    
    # init-db command
    init_parser = subparsers.add_parser(
        'init-db',
        help='Initialize database schema',
        description='Create SQLite database with proper schema for Persian legal documents'
    )
    init_parser.add_argument(
        '--db-path',
        type=str,
        help='Database file path',
        metavar='PATH'
    )
    
    # import-data command  
    import_parser = subparsers.add_parser(
        'import-data',
        help='Import JSON documents into database',
        description='Import processed JSON documents from Phase 1 into the database'
    )
    import_parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing JSON files',
        metavar='DIR'
    )
    import_parser.add_argument(
        '--db-path',
        type=str,
        help='Database file path',
        metavar='PATH'
    )
    
    # validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate database contents',
        description='Check database contents and display statistics'
    )
    validate_parser.add_argument(
        '--db-path',
        type=str,
        help='Database file path',
        metavar='PATH'
    )
    
    return parser


def main() -> None:
    """
    Main entry point for the CLI application.
    
    Parses command line arguments and dispatches to appropriate handlers.
    """
    logger = get_logger(__name__)
    
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        # Check if a command was provided
        if not args.command:
            parser.print_help()
            print("\n❌ لطفاً یکی از دستورات موجود را انتخاب کنید")
            print("❌ Please specify a command")
            sys.exit(1)
        
        # Dispatch to appropriate handler
        success = False
        
        if args.command == 'init-db':
            success = handle_init_db(args)
        elif args.command == 'import-data':
            success = handle_import_data(args)
        elif args.command == 'validate':
            success = handle_validate(args)
        else:
            print(f"❌ دستور نامعتبر: {args.command}")
            print(f"❌ Invalid command: {args.command}")
            parser.print_help()
            sys.exit(1)
        
        # Exit with appropriate status
        if success:
            logger.info(f"CLI command '{args.command}' completed successfully")
            sys.exit(0)
        else:
            logger.error(f"CLI command '{args.command}' failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  عملیات توسط کاربر متوقف شد")
        print(f"⏹️  Operation interrupted by user")
        logger.info("CLI operation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] CLI error: {e}")
        print(f"[ERROR] Unexpected CLI error: {e}")
        logger.error(f"Unexpected CLI error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()