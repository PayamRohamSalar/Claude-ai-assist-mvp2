"""
Data Importer Module for Legal Assistant AI (Phase 2)

This module handles importing canonical JSON files from Phase 1 
into the SQLite database with proper structure and relationships.

Author: Legal Assistant AI Team
Version: 2.0
"""

import json
import os
import sys
import sqlite3
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

# Import shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared_utils.logger import get_logger
from shared_utils.config_manager import get_config


@dataclass
class ImportStats:
    """Statistics for document import operation."""
    total_files: int = 0
    documents_processed: int = 0
    documents_inserted: int = 0
    documents_updated: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chapters_inserted: int = 0
    articles_inserted: int = 0
    notes_inserted: int = 0
    clauses_inserted: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DocumentImportError(Exception):
    """Custom exception for document import errors."""
    pass


def compute_document_uid(title: str, approval_date: str) -> str:
    """
    Compute a stable, unique identifier for a document.
    
    Uses SHA-256 hash of title and approval date to create a deterministic
    UID that can be used to identify documents across imports.
    
    Args:
        title (str): Document title
        approval_date (str): Document approval date
        
    Returns:
        str: Truncated SHA-256 hash as document UID
    """
    # Normalize inputs
    title_norm = (title or "").strip().lower()
    date_norm = (approval_date or "").strip()
    
    # Create hash input
    hash_input = f"{title_norm}|{date_norm}".encode('utf-8')
    
    # Generate SHA-256 hash and truncate to 16 characters
    sha256_hash = hashlib.sha256(hash_input).hexdigest()
    return sha256_hash[:16]


def insert_document(conn: sqlite3.Connection, doc_data: Dict[str, Any], 
                   document_uid: str, source_file: str, logger) -> int:
    """
    Insert a document record into the database.
    
    Args:
        conn: Database connection
        doc_data: Document metadata dictionary
        document_uid: Unique document identifier
        source_file: Source JSON file path
        logger: Logger instance
        
    Returns:
        int: Document ID of inserted record
        
    Raises:
        sqlite3.Error: If database operation fails
    """
    cursor = conn.cursor()
    
    # Extract document metadata
    metadata = doc_data.get('metadata', {})
    
    cursor.execute("""
        INSERT INTO documents (
            document_uid, title, document_type, section,
            approval_authority, approval_date, effective_date,
            document_number, subject, keywords, confidence_score,
            source_file, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        document_uid,
        metadata.get('title'),
        metadata.get('document_type'),
        metadata.get('section'),
        metadata.get('approval_authority'),
        metadata.get('approval_date'),
        metadata.get('effective_date'),
        metadata.get('document_number'),
        metadata.get('subject'),
        json.dumps(metadata.get('keywords', []) if metadata.get('keywords') else []),
        metadata.get('confidence_score', 0.0),
        source_file,
        datetime.now().isoformat(),
        datetime.now().isoformat()
    ))
    
    document_id = cursor.lastrowid
    logger.debug(f"Inserted document with ID: {document_id}")
    return document_id


def clear_document_structure(conn: sqlite3.Connection, document_id: int, logger) -> None:
    """
    Clear existing structure (chapters, articles, notes, clauses) for a document.
    
    Args:
        conn: Database connection
        document_id: ID of document to clear
        logger: Logger instance
    """
    cursor = conn.cursor()
    
    # Delete chapters and all children (CASCADE will handle articles, notes, clauses)
    cursor.execute("DELETE FROM chapters WHERE document_id = ?", (document_id,))
    rows_deleted = cursor.rowcount
    logger.debug(f"Cleared {rows_deleted} chapters for document ID: {document_id}")


def update_document(conn: sqlite3.Connection, document_id: int, 
                   doc_data: Dict[str, Any], source_file: str, logger) -> None:
    """
    Update an existing document record.
    
    Args:
        conn: Database connection
        document_id: ID of document to update
        doc_data: Updated document metadata
        source_file: Source JSON file path
        logger: Logger instance
    """
    cursor = conn.cursor()
    metadata = doc_data.get('metadata', {})
    
    # Clear existing document structure to ensure idempotence
    clear_document_structure(conn, document_id, logger)
    
    cursor.execute("""
        UPDATE documents SET 
            title = ?, document_type = ?, section = ?,
            approval_authority = ?, approval_date = ?, effective_date = ?,
            document_number = ?, subject = ?, keywords = ?, 
            confidence_score = ?, source_file = ?, updated_at = ?
        WHERE id = ?
    """, (
        metadata.get('title'),
        metadata.get('document_type'),
        metadata.get('section'),
        metadata.get('approval_authority'),
        metadata.get('approval_date'),
        metadata.get('effective_date'),
        metadata.get('document_number'),
        metadata.get('subject'),
        json.dumps(metadata.get('keywords', []) if metadata.get('keywords') else []),
        metadata.get('confidence_score', 0.0),
        source_file,
        datetime.now().isoformat(),
        document_id
    ))
    
    logger.debug(f"Updated document with ID: {document_id}")


def insert_chapter(conn: sqlite3.Connection, document_id: int, 
                  chapter_data: Dict[str, Any], chapter_index: int, logger) -> int:
    """
    Insert a chapter record.
    
    Args:
        conn: Database connection
        document_id: Parent document ID
        chapter_data: Chapter data dictionary
        chapter_index: Chapter index/order
        logger: Logger instance
        
    Returns:
        int: Chapter ID of inserted record
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO chapters (document_id, chapter_index, chapter_title)
        VALUES (?, ?, ?)
    """, (
        document_id,
        chapter_index,
        chapter_data.get('title', chapter_data.get('chapter_title'))
    ))
    
    chapter_id = cursor.lastrowid
    logger.debug(f"Inserted chapter with ID: {chapter_id}")
    return chapter_id


def insert_article(conn: sqlite3.Connection, chapter_id: int, 
                  article_data: Dict[str, Any], logger) -> int:
    """
    Insert an article record.
    
    Args:
        conn: Database connection
        chapter_id: Parent chapter ID
        article_data: Article data dictionary
        logger: Logger instance
        
    Returns:
        int: Article ID of inserted record
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO articles (chapter_id, article_number, article_text)
        VALUES (?, ?, ?)
    """, (
        chapter_id,
        article_data.get('number', article_data.get('article_number')),
        article_data.get('text', article_data.get('article_text'))
    ))
    
    article_id = cursor.lastrowid
    logger.debug(f"Inserted article with ID: {article_id}")
    return article_id


def insert_note(conn: sqlite3.Connection, article_id: int, 
               note_data: Dict[str, Any], logger) -> int:
    """
    Insert a note (تبصره) record.
    
    Args:
        conn: Database connection
        article_id: Parent article ID
        note_data: Note data dictionary
        logger: Logger instance
        
    Returns:
        int: Note ID of inserted record
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO notes (article_id, note_label, note_text)
        VALUES (?, ?, ?)
    """, (
        article_id,
        note_data.get('label', note_data.get('note_label')),
        note_data.get('text', note_data.get('note_text'))
    ))
    
    note_id = cursor.lastrowid
    logger.debug(f"Inserted note with ID: {note_id}")
    return note_id


def insert_clause(conn: sqlite3.Connection, note_id: int, 
                 clause_data: Dict[str, Any], logger) -> int:
    """
    Insert a clause (بند) record.
    
    Args:
        conn: Database connection
        note_id: Parent note ID
        clause_data: Clause data dictionary
        logger: Logger instance
        
    Returns:
        int: Clause ID of inserted record
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO clauses (note_id, clause_label, clause_text)
        VALUES (?, ?, ?)
    """, (
        note_id,
        clause_data.get('label', clause_data.get('clause_label')),
        clause_data.get('text', clause_data.get('clause_text'))
    ))
    
    clause_id = cursor.lastrowid
    logger.debug(f"Inserted clause with ID: {clause_id}")
    return clause_id


def check_document_exists(conn: sqlite3.Connection, document_uid: str) -> Optional[int]:
    """
    Check if a document with the given UID already exists.
    
    Args:
        conn: Database connection
        document_uid: Document UID to check
        
    Returns:
        Optional[int]: Document ID if exists, None otherwise
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM documents WHERE document_uid = ?", (document_uid,))
    result = cursor.fetchone()
    return result[0] if result else None


def process_document_structure(conn: sqlite3.Connection, document_id: int,
                             doc_data: Dict[str, Any], stats: ImportStats, logger) -> None:
    """
    Process and insert the hierarchical structure of a document.
    
    Args:
        conn: Database connection
        document_id: Document ID
        doc_data: Document data dictionary
        stats: Import statistics tracker
        logger: Logger instance
    """
    # Get document structure
    structure = doc_data.get('structure', {})
    chapters = structure.get('chapters', [])
    
    # If no chapters, create a default chapter for articles
    if not chapters and 'articles' in doc_data:
        chapters = [{'title': 'محتوای اصلی', 'articles': doc_data['articles']}]
    elif not chapters and 'content' in doc_data:
        chapters = [{'title': 'محتوای اصلی', 'articles': doc_data['content'].get('articles', [])}]
    
    for chapter_index, chapter_data in enumerate(chapters, 1):
        try:
            chapter_id = insert_chapter(conn, document_id, chapter_data, chapter_index, logger)
            stats.chapters_inserted += 1
            
            # Process articles in chapter
            articles = chapter_data.get('articles', [])
            for article_data in articles:
                try:
                    article_id = insert_article(conn, chapter_id, article_data, logger)
                    stats.articles_inserted += 1
                    
                    # Process notes (تبصره) in article
                    notes = article_data.get('notes', article_data.get('تبصره‌ها', []))
                    for note_data in notes:
                        try:
                            note_id = insert_note(conn, article_id, note_data, logger)
                            stats.notes_inserted += 1
                            
                            # Process clauses (بند) in note
                            clauses = note_data.get('clauses', note_data.get('بندها', []))
                            for clause_data in clauses:
                                try:
                                    insert_clause(conn, note_id, clause_data, logger)
                                    stats.clauses_inserted += 1
                                except Exception as e:
                                    error_msg = f"Failed to insert clause: {e}"
                                    logger.error(error_msg)
                                    stats.errors.append(error_msg)
                                    
                        except Exception as e:
                            error_msg = f"Failed to insert note: {e}"
                            logger.error(error_msg)
                            stats.errors.append(error_msg)
                            
                except Exception as e:
                    error_msg = f"Failed to insert article: {e}"
                    logger.error(error_msg)
                    stats.errors.append(error_msg)
                    
        except Exception as e:
            error_msg = f"Failed to insert chapter: {e}"
            logger.error(error_msg)
            stats.errors.append(error_msg)


def import_documents(input_dir: Union[str, Path], 
                    conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Import documents from JSON files into the SQLite database.
    
    This function processes all JSON files in the specified directory,
    extracts document metadata and structure, and inserts them into
    the database with proper relationships and foreign keys.
    
    Args:
        input_dir: Directory containing JSON files to import
        conn: SQLite database connection
        
    Returns:
        Dict[str, Any]: Import statistics and results
        
    Raises:
        DocumentImportError: If import process fails critically
    """
    logger = get_logger(__name__)
    stats = ImportStats()
    
    # Convert input_dir to Path object
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise DocumentImportError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise DocumentImportError(f"Input path is not a directory: {input_dir}")
    
    # Get all JSON files in sorted order for determinism
    json_files = sorted(input_path.glob("*.json"))
    stats.total_files = len(json_files)
    
    if not json_files:
        logger.warning(f"No JSON files found in directory: {input_dir}")
        logger.warning(f"هیچ فایل JSON در پوشه یافت نشد: {input_dir}")
        return stats.__dict__
    
    logger.info(f"Starting import of {stats.total_files} JSON files")
    logger.info(f"شروع وارد کردن {stats.total_files} فایل JSON")
    
    for json_file in json_files:
        try:
            logger.info(f"Processing file: {json_file.name}")
            logger.info(f"در حال پردازش فایل: {json_file.name}")
            
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Skip non-document files (like metadata files)
            if not isinstance(doc_data, dict) or 'metadata' not in doc_data:
                logger.info(f"Skipping non-document file: {json_file.name}")
                continue
            
            stats.documents_processed += 1
            
            # Extract metadata for UID computation
            metadata = doc_data.get('metadata', {})
            title = metadata.get('title', json_file.stem)
            approval_date = metadata.get('approval_date', '')
            
            # Compute document UID
            document_uid = compute_document_uid(title, approval_date)
            
            # Start transaction for this document
            conn.execute('BEGIN')
            
            try:
                # Check if document already exists
                existing_doc_id = check_document_exists(conn, document_uid)
                
                if existing_doc_id:
                    logger.info(f"Document already exists, updating: {title}")
                    logger.info(f"سند قبلاً وجود دارد، در حال به‌روزرسانی: {title}")
                    
                    update_document(conn, existing_doc_id, doc_data, str(json_file), logger)
                    document_id = existing_doc_id
                    stats.documents_updated += 1
                else:
                    logger.info(f"Inserting new document: {title}")
                    logger.info(f"در حال وارد کردن سند جدید: {title}")
                    
                    document_id = insert_document(conn, doc_data, document_uid, 
                                                str(json_file), logger)
                    stats.documents_inserted += 1
                
                # Process document structure (chapters, articles, notes, clauses)
                process_document_structure(conn, document_id, doc_data, stats, logger)
                
                # Commit transaction
                conn.commit()
                
                logger.info(f"Successfully processed: {json_file.name}")
                logger.info(f"با موفقیت پردازش شد: {json_file.name}")
                
            except Exception as e:
                # Rollback transaction on error
                conn.rollback()
                error_msg = f"Failed to import document {json_file.name}: {e}"
                logger.error(error_msg)
                logger.error(f"خطا در وارد کردن سند {json_file.name}: {e}")
                stats.errors.append(error_msg)
                stats.documents_failed += 1
                
        except Exception as e:
            error_msg = f"Failed to process file {json_file.name}: {e}"
            logger.error(error_msg)
            logger.error(f"خطا در پردازش فایل {json_file.name}: {e}")
            stats.errors.append(error_msg)
            stats.documents_failed += 1
    
    # Generate final statistics
    logger.info("Import process completed")
    logger.info("فرآیند وارد کردن تکمیل شد")
    
    logger.info(f"Files processed: {stats.documents_processed}")
    logger.info(f"Documents inserted: {stats.documents_inserted}")
    logger.info(f"Documents updated: {stats.documents_updated}")
    logger.info(f"Documents failed: {stats.documents_failed}")
    logger.info(f"Chapters inserted: {stats.chapters_inserted}")
    logger.info(f"Articles inserted: {stats.articles_inserted}")
    logger.info(f"Notes inserted: {stats.notes_inserted}")
    logger.info(f"Clauses inserted: {stats.clauses_inserted}")
    
    # Convert stats to dictionary and add summary
    result = stats.__dict__.copy()
    result['import_completed_at'] = datetime.now().isoformat()
    result['success_rate'] = (
        (stats.documents_inserted + stats.documents_updated) / 
        max(stats.documents_processed, 1) * 100
    )
    
    # Optionally write validation report
    try:
        config = get_config()
        # Handle different config object types
        if hasattr(config, 'get'):
            report_path = config.get('database', {}).get('validation_report_path', 
                                                        'data/db/validation_report.json')
        else:
            report_path = 'data/db/validation_report.json'
        
        # Ensure directory exists
        report_dir = Path(report_path).parent
        report_dir.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report written to: {report_path}")
        logger.info(f"گزارش اعتبارسنجی در فایل ذخیره شد: {report_path}")
        
    except Exception as e:
        logger.warning(f"Could not write validation report: {e}")
        logger.warning(f"امکان نوشتن گزارش اعتبارسنجی وجود نداشت: {e}")
    
    return result


def main() -> None:
    """
    Main entry point for CLI execution.
    
    Usage: python data_importer.py <input_directory> [database_path]
    """
    logger = get_logger(__name__)
    
    if len(sys.argv) < 2:
        print("Usage: python data_importer.py <input_directory> [database_path]")
        print("استفاده: python data_importer.py <پوشه_ورودی> [مسیر_پایگاه_داده]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Import database creator
        try:
            from .database_creator import init_database
        except ImportError:
            from database_creator import init_database
        
        # Initialize database
        logger.info("Initializing database connection...")
        logger.info("در حال راه‌اندازی اتصال پایگاه داده...")
        
        conn = init_database(db_path)
        
        # Import documents
        logger.info("Starting document import...")
        logger.info("شروع وارد کردن اسناد...")
        
        results = import_documents(input_dir, conn)
        
        # Close connection
        conn.close()
        
        # Print summary
        print(f"\n=== Import Summary ===")
        print(f"Files processed: {results['documents_processed']}")
        print(f"Documents inserted: {results['documents_inserted']}")
        print(f"Documents updated: {results['documents_updated']}")
        print(f"Documents failed: {results['documents_failed']}")
        print(f"Success rate: {results['success_rate']:.1f}%")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more")
        
        print("\n[SUCCESS] Document import completed!")
        print("[SUCCESS] وارد کردن اسناد با موفقیت تکمیل شد!")
        
    except Exception as e:
        logger.error(f"Import process failed: {e}")
        logger.error(f"فرآیند وارد کردن ناموفق بود: {e}")
        print(f"[ERROR] Import failed: {e}")
        print(f"[ERROR] وارد کردن ناموفق بود: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()