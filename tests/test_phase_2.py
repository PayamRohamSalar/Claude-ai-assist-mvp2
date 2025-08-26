"""
Unit tests for Phase 2 database functionality.

This module tests the database creation and data import functionality
for the Legal Assistant AI project Phase 2.

Author: Legal Assistant AI Team
Version: 2.0
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import sys
# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from phase_2_database import database_creator, data_importer
except ImportError:
    # Fallback for different import paths
    import phase_2_database.database_creator as database_creator
    import phase_2_database.data_importer as data_importer


class TestPhase2Database(unittest.TestCase):
    """
    Test suite for Phase 2 database functionality.
    
    Tests database initialization, document import, and idempotence.
    """

    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="legal_ai_test_")
        self.test_data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Set up temporary database path
        self.temp_db_path = os.path.join(self.temp_dir, "test_database.db")
        
        # Sample JSON data for testing
        self.sample_doc1 = {
            "metadata": {
                "title": "قانون آزمایشی اول",
                "document_type": "قانون",
                "approval_authority": "مجلس شورای اسلامی",
                "approval_date": "1400/01/01",
                "effective_date": "1400/02/01",
                "section_name": "بخش آزمایشی",
                "document_number": "001",
                "subject": "موضوع آزمایشی",
                "keywords": ["آزمایش", "قانون"],
                "confidence_score": 0.95
            },
            "structure": {
                "chapters": [
                    {
                        "title": "فصل اول",
                        "articles": [
                            {
                                "number": "1",
                                "text": "این ماده اول قانون آزمایشی است.",
                                "notes": [
                                    {
                                        "label": "تبصره 1",
                                        "text": "این تبصره اول است.",
                                        "clauses": [
                                            {
                                                "label": "بند الف",
                                                "text": "این بند الف است."
                                            }
                                        ]
                                    }
                                ]
                            },
                            {
                                "number": "2", 
                                "text": "این ماده دوم قانون آزمایشی است.",
                                "notes": []
                            }
                        ]
                    }
                ],
                "totals": {
                    "chapters": 1,
                    "articles": 2,
                    "notes": 1
                }
            }
        }
        
        self.sample_doc2 = {
            "metadata": {
                "title": "قانون آزمایشی دوم",
                "document_type": "آیین‌نامه",
                "approval_authority": "هیئت وزیران",
                "approval_date": "1400/03/01",
                "effective_date": "1400/04/01",
                "section_name": "بخش آزمایشی",
                "document_number": "002",
                "subject": "موضوع آزمایشی دوم",
                "keywords": ["آزمایش", "آیین‌نامه"],
                "confidence_score": 0.88
            },
            "structure": {
                "chapters": [
                    {
                        "title": "فصل واحد",
                        "articles": [
                            {
                                "number": "1",
                                "text": "این ماده واحد آیین‌نامه آزمایشی است.",
                                "notes": []
                            }
                        ]
                    }
                ],
                "totals": {
                    "chapters": 1,
                    "articles": 1,
                    "notes": 0
                }
            }
        }
        
        # Create test JSON files
        self.test_file1 = os.path.join(self.test_data_dir, "test_doc1.json")
        self.test_file2 = os.path.join(self.test_data_dir, "test_doc2.json")
        
        with open(self.test_file1, 'w', encoding='utf-8') as f:
            json.dump(self.sample_doc1, f, ensure_ascii=False, indent=2)
            
        with open(self.test_file2, 'w', encoding='utf-8') as f:
            json.dump(self.sample_doc2, f, ensure_ascii=False, indent=2)
    
    def tearDown(self):
        """Clean up test environment after each test."""
        # Close any open database connections
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except:
                pass
        
        # Remove temporary directory and all its contents
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            # On Windows, sometimes files are still locked
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(self.temp_dir)
            except OSError:
                pass  # Best effort cleanup
    
    def _setup_test_database(self, db_path=":memory:"):
        """Helper method to set up test database with schema."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Load and execute schema directly
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'phase_2_database', 'schema.sql')
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        with conn:
            conn.executescript(schema_sql)
        
        return conn
    
    def test_database_initialization(self):
        """Test database initialization with schema creation."""
        # Test in-memory database
        conn_memory = self._setup_test_database(":memory:")
        self.assertIsInstance(conn_memory, sqlite3.Connection)
        
        # Verify tables exist
        cursor = conn_memory.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['documents', 'chapters', 'articles', 'notes', 'clauses']
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} not found in database")
        
        # Verify FTS tables exist
        fts_tables = [t for t in tables if '_fts' in t]
        expected_fts_tables = ['documents_fts', 'articles_fts', 'notes_fts', 'clauses_fts']
        for fts_table in expected_fts_tables:
            self.assertIn(fts_table, fts_tables, f"FTS table {fts_table} not found")
        
        conn_memory.close()
        
        # Test file-based database
        conn_file = self._setup_test_database(self.temp_db_path)
        self.assertIsInstance(conn_file, sqlite3.Connection)
        self.assertTrue(os.path.exists(self.temp_db_path))
        
        conn_file.close()
    
    def test_document_import_initial(self):
        """Test initial document import functionality."""
        # Initialize database
        self.conn = self._setup_test_database(":memory:")
        
        # Import documents
        results = data_importer.import_documents(self.test_data_dir, self.conn)
        
        # Verify import results
        self.assertIsInstance(results, dict)
        self.assertGreater(results['documents_processed'], 0)
        self.assertEqual(results['documents_inserted'], 2)
        self.assertEqual(results['documents_updated'], 0)
        self.assertEqual(results['documents_failed'], 0)
        
        # Verify document counts
        cursor = self.conn.cursor()
        
        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        self.assertEqual(doc_count, 2, "Expected 2 documents in database")
        
        # Count chapters
        cursor.execute("SELECT COUNT(*) FROM chapters")
        chapter_count = cursor.fetchone()[0]
        self.assertEqual(chapter_count, 2, "Expected 2 chapters in database")
        
        # Count articles
        cursor.execute("SELECT COUNT(*) FROM articles")
        article_count = cursor.fetchone()[0]
        self.assertEqual(article_count, 3, "Expected 3 articles in database")
        
        # Count notes
        cursor.execute("SELECT COUNT(*) FROM notes")
        note_count = cursor.fetchone()[0]
        self.assertEqual(note_count, 1, "Expected 1 note in database")
        
        # Count clauses
        cursor.execute("SELECT COUNT(*) FROM clauses")
        clause_count = cursor.fetchone()[0]
        self.assertEqual(clause_count, 1, "Expected 1 clause in database")
        
        # Verify specific document data
        cursor.execute("SELECT title, document_type FROM documents ORDER BY title")
        docs = cursor.fetchall()
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0][0], "قانون آزمایشی اول")
        self.assertEqual(docs[0][1], "قانون")
        self.assertEqual(docs[1][0], "قانون آزمایشی دوم")
        self.assertEqual(docs[1][1], "آیین‌نامه")
    
    def test_document_import_idempotence(self):
        """Test that re-importing the same documents doesn't create duplicates."""
        # Initialize database
        self.conn = self._setup_test_database(":memory:")
        
        # First import
        results1 = data_importer.import_documents(self.test_data_dir, self.conn)
        
        # Get initial counts
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        initial_doc_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM chapters")
        initial_chapter_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM articles")
        initial_article_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM notes")
        initial_note_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM clauses")
        initial_clause_count = cursor.fetchone()[0]
        
        # Second import (should be idempotent)
        results2 = data_importer.import_documents(self.test_data_dir, self.conn)
        
        # Get final counts
        cursor.execute("SELECT COUNT(*) FROM documents")
        final_doc_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM chapters")
        final_chapter_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM articles")
        final_article_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM notes")
        final_note_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM clauses")
        final_clause_count = cursor.fetchone()[0]
        
        # Verify counts haven't increased
        self.assertEqual(initial_doc_count, final_doc_count, 
                        "Document count should not increase on re-import")
        self.assertEqual(initial_chapter_count, final_chapter_count,
                        "Chapter count should not increase on re-import")
        self.assertEqual(initial_article_count, final_article_count,
                        "Article count should not increase on re-import")
        self.assertEqual(initial_note_count, final_note_count,
                        "Note count should not increase on re-import")
        self.assertEqual(initial_clause_count, final_clause_count,
                        "Clause count should not increase on re-import")
        
        # Verify second import shows updates rather than inserts
        self.assertEqual(results2['documents_inserted'], 0, 
                        "No new documents should be inserted on re-import")
        self.assertEqual(results2['documents_updated'], 2,
                        "Both documents should be updated on re-import")
    
    def test_database_health_check(self):
        """Test database health check functionality."""
        # Initialize database
        conn = self._setup_test_database(":memory:")
        
        # Test basic functionality - just verify we can query tables
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 0, "Fresh database should have no documents")
        
        conn.close()
    
    def test_document_uid_generation(self):
        """Test document UID generation for consistency."""
        # Test UID generation
        uid1 = data_importer.compute_document_uid("قانون آزمایشی", "1400/01/01")
        uid2 = data_importer.compute_document_uid("قانون آزمایشی", "1400/01/01")
        uid3 = data_importer.compute_document_uid("قانون متفاوت", "1400/01/01")
        
        # Same inputs should produce same UID
        self.assertEqual(uid1, uid2, "Same inputs should produce same UID")
        
        # Different inputs should produce different UIDs
        self.assertNotEqual(uid1, uid3, "Different inputs should produce different UIDs")
        
        # UID should be 16 characters long
        self.assertEqual(len(uid1), 16, "UID should be 16 characters long")
    
    def test_import_with_malformed_json(self):
        """Test import behavior with malformed or non-document JSON files."""
        # Create a malformed JSON file
        malformed_file = os.path.join(self.test_data_dir, "malformed.json")
        with open(malformed_file, 'w', encoding='utf-8') as f:
            f.write('{"invalid": json')
        
        # Create a non-document JSON file (metadata file)
        metadata_file = os.path.join(self.test_data_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({"processing_info": "not a document"}, f)
        
        # Initialize database and import
        self.conn = self._setup_test_database(":memory:")
        results = data_importer.import_documents(self.test_data_dir, self.conn)
        
        # Should still process the valid documents, skip invalid ones
        self.assertEqual(results['documents_inserted'], 2, 
                        "Should insert valid documents despite invalid files")
        self.assertGreater(len(results['errors']), 0, 
                          "Should report errors for malformed files")
    
    def test_empty_directory_import(self):
        """Test import behavior with empty directory."""
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        # Initialize database and import from empty directory
        self.conn = self._setup_test_database(":memory:")
        results = data_importer.import_documents(empty_dir, self.conn)
        
        # Should handle empty directory gracefully
        self.assertEqual(results['total_files'], 0)
        self.assertEqual(results['documents_processed'], 0)
        self.assertEqual(results['documents_inserted'], 0)
    
    def test_full_text_search_setup(self):
        """Test that FTS tables are properly set up and functional."""
        # Initialize database and import data
        self.conn = self._setup_test_database(":memory:")
        data_importer.import_documents(self.test_data_dir, self.conn)
        
        cursor = self.conn.cursor()
        
        # Test FTS search on documents
        cursor.execute("SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH ?", ("آزمایشی",))
        fts_count = cursor.fetchone()[0]
        self.assertGreater(fts_count, 0, "FTS search should find matching documents")
        
        # Test FTS search on articles
        cursor.execute("SELECT COUNT(*) FROM articles_fts WHERE articles_fts MATCH ?", ("ماده",))
        article_fts_count = cursor.fetchone()[0]
        self.assertGreater(article_fts_count, 0, "FTS search should find matching articles")


def run_tests():
    """Run all tests with proper setup and teardown."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase2Database)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)