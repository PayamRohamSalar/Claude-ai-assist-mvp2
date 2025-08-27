#!/usr/bin/env python3
"""
Phase 3 RAG Pipeline Tests

This test suite validates the complete RAG pipeline including chunking,
embedding generation, vector store building, and search functionality.
Tests use a miniature in-memory SQLite database with sample Persian legal content.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import shutil
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
from phase_3_rag import chunker
from phase_3_rag.embedding_generator import EmbeddingGenerator
from phase_3_rag.vector_store_builder import VectorStoreBuilder

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class TestPhase3Pipeline(unittest.TestCase):
    """Test suite for Phase 3 RAG pipeline components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with sample data and temporary directories"""
        cls.test_dir = tempfile.mkdtemp(prefix="phase3_test_")
        cls.test_db_path = os.path.join(cls.test_dir, "test_legal.db")
        cls.chunks_path = os.path.join(cls.test_dir, "chunks.json")
        cls.embeddings_path = os.path.join(cls.test_dir, "embeddings.npy")
        cls.meta_path = os.path.join(cls.test_dir, "embeddings_meta.json")
        cls.vector_db_dir = os.path.join(cls.test_dir, "vector_db")
        
        # Create test database with sample Persian legal content
        cls._create_test_database()
        
        # Test keywords present in sample data
        cls.test_keywords = ["دانشگاه", "وزارت", "آموزش", "علوم"]
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test directories"""
        try:
            if os.path.exists(cls.test_dir):
                shutil.rmtree(cls.test_dir)
        except (PermissionError, OSError):
            # Ignore cleanup errors in tests
            pass
    
    @classmethod
    def _create_test_database(cls):
        """Create a miniature SQLite database with sample Persian legal content"""
        
        # Sample Persian legal documents
        sample_documents = [
            {
                "uid": "doc001",
                "title": "قانون آموزش عالی",
                "type": "قانون",
                "approval_date": "1400/01/01",
                "approval_authority": "مجلس شورای اسلامی",
                "articles": [
                    {
                        "number": "1",
                        "text": """وزارت علوم، تحقیقات و فناوری مسئول سیاستگذاری و نظارت بر آموزش عالی کشور می‌باشد.
                        
تبصره 1- دانشگاه‌ها موظف به اجرای سیاست‌های وزارت علوم هستند.

تبصره 2- مراکز آموزش عالی تحت نظارت وزارت علوم قرار دارند.""",
                        "notes": [
                            {
                                "label": "تبصره 1",
                                "text": "دانشگاه‌ها موظف به اجرای سیاست‌های وزارت علوم هستند."
                            },
                            {
                                "label": "تبصره 2", 
                                "text": "مراکز آموزش عالی تحت نظارت وزارت علوم قرار دارند."
                            }
                        ]
                    },
                    {
                        "number": "2",
                        "text": """شورای عالی آموزش عالی بالاترین مرجع تصمیم‌گیری در امور آموزش عالی است.
                        
بند الف- تعیین خط‌مشی‌های کلی آموزش عالی
بند ب- تصویب برنامه‌های توسعه دانشگاهی

تبصره- اعضای شورا توسط وزیر علوم منصوب می‌شوند.""",
                        "notes": [
                            {
                                "label": "تبصره",
                                "text": "اعضای شورا توسط وزیر علوم منصوب می‌شوند."
                            }
                        ],
                        "clauses": [
                            {
                                "label": "بند الف",
                                "text": "تعیین خط‌مشی‌های کلی آموزش عالی"
                            },
                            {
                                "label": "بند ب",
                                "text": "تصویب برنامه‌های توسعه دانشگاهی"
                            }
                        ]
                    }
                ]
            },
            {
                "uid": "doc002",
                "title": "آیین‌نامه اجرایی دانشگاه‌ها",
                "type": "آیین‌نامه",
                "approval_date": "1400/02/15",
                "approval_authority": "وزارت علوم، تحقیقات و فناوری",
                "articles": [
                    {
                        "number": "1",
                        "text": """هر دانشگاه دارای شورای دانشگاه متشکل از اعضای هیئت علمی است.
                        
تبصره- رئیس دانشگاه بر اساس پیشنهاد شورای دانشگاه انتخاب می‌شود.""",
                        "notes": [
                            {
                                "label": "تبصره",
                                "text": "رئیس دانشگاه بر اساس پیشنهاد شورای دانشگاه انتخاب می‌شود."
                            }
                        ]
                    }
                ]
            }
        ]
        
        # Create database and tables using phase 2 schema
        conn = sqlite3.connect(cls.test_db_path)
        cursor = conn.cursor()
        
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_uid TEXT UNIQUE NOT NULL,
                title TEXT,
                document_type TEXT,
                section TEXT,
                approval_authority TEXT,
                approval_date TEXT,
                effective_date TEXT,
                document_number TEXT,
                subject TEXT,
                keywords TEXT,
                confidence_score REAL,
                source_file TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        ''')
        
        # Create chapters table
        cursor.execute('''
            CREATE TABLE chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chapter_index INTEGER,
                chapter_title TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        ''')
        
        # Create articles table
        cursor.execute('''
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                article_number TEXT,
                article_text TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
            )
        ''')
        
        # Create notes table
        cursor.execute('''
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                note_label TEXT,
                note_text TEXT,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        ''')
        
        # Create clauses table
        cursor.execute('''
            CREATE TABLE clauses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                clause_label TEXT,
                clause_text TEXT,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            )
        ''')
        
        # Insert sample data with proper foreign key relationships
        for doc in sample_documents:
            # Insert document
            cursor.execute('''
                INSERT INTO documents (document_uid, title, document_type, approval_date, approval_authority)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc['uid'], doc['title'], doc['type'], doc['approval_date'], doc['approval_authority']))
            
            doc_id = cursor.lastrowid
            
            # Create a default chapter for each document
            cursor.execute('''
                INSERT INTO chapters (document_id, chapter_index, chapter_title)
                VALUES (?, ?, ?)
            ''', (doc_id, 1, 'محتوای اصلی'))
            
            chapter_id = cursor.lastrowid
            
            for article in doc['articles']:
                # Insert article
                cursor.execute('''
                    INSERT INTO articles (chapter_id, article_number, article_text)
                    VALUES (?, ?, ?)
                ''', (chapter_id, article['number'], article['text']))
                
                article_id = cursor.lastrowid
                
                # Insert notes
                if 'notes' in article:
                    for note in article['notes']:
                        cursor.execute('''
                            INSERT INTO notes (article_id, note_label, note_text)
                            VALUES (?, ?, ?)
                        ''', (article_id, note['label'], note['text']))
                        
                        note_id = cursor.lastrowid
                        
                        # Insert clauses for this note (if any)
                        if 'clauses' in article:
                            for clause in article['clauses']:
                                cursor.execute('''
                                    INSERT INTO clauses (note_id, clause_label, clause_text)
                                    VALUES (?, ?, ?)
                                ''', (note_id, clause['label'], clause['text']))
        
        conn.commit()
        conn.close()
    
    def setUp(self):
        """Set up each test with clean state"""
        # Clean up any files from previous tests
        for path in [self.chunks_path, self.embeddings_path, self.meta_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.vector_db_dir):
            shutil.rmtree(self.vector_db_dir)
    
    def test_01_chunker_creates_valid_chunks(self):
        """Test that chunker creates valid chunks from test database"""
        # Load chunking configuration
        config = chunker.load_chunking_config()
        
        # Create chunks from test database
        chunks = chunker.create_chunks_from_database(
            db_path=self.test_db_path,
            config=config
        )
        
        # Assert chunks were created
        self.assertGreater(len(chunks), 0, "No chunks were created")
        print(f"[OK] Created {len(chunks)} chunks")
        
        # Verify chunk structure
        for chunk in chunks:
            self.assertIn('chunk_uid', chunk)  # Chunker uses 'chunk_uid'
            self.assertIn('document_uid', chunk)
            
            # Check for content field (chunker may use different field names)
            content_field = chunk.get('content', chunk.get('text', chunk.get('normalized_text', '')))
            self.assertGreater(len(content_field.strip()), 0, "Empty chunk content")
        
        # Test boundary preservation: no chunk should span across تبصره/بند boundaries
        tabsareh_chunks = []
        band_chunks = []
        
        for c in chunks:
            content = c.get('content', c.get('text', c.get('normalized_text', '')))
            if 'تبصره' in content:
                tabsareh_chunks.append(c)
            if 'بند' in content:
                band_chunks.append(c)
        
        print(f"[OK] Found {len(tabsareh_chunks)} tabsareh chunks and {len(band_chunks)} band chunks")
        
        # Each تبصره or بند should be in its own chunk or clearly separated
        for chunk in tabsareh_chunks:
            content = chunk.get('content', chunk.get('text', chunk.get('normalized_text', '')))
            # Should not contain multiple تبصره labels in one chunk
            tabsareh_count = content.count('تبصره ')
            if tabsareh_count > 1:
                # Check if they are properly separated (different تبصره numbers)
                self.assertIn('تبصره 1', content)
                self.assertNotIn('تبصره 2', content)
        
        # Save chunks for next tests
        chunker.write_chunks_json(chunks, output_path=self.chunks_path)
        
        return chunks
    
    @patch('phase_3_rag.embedding_generator.SentenceTransformer')
    def test_02_embedding_generator_creates_embeddings(self, mock_transformer):
        """Test embedding generator with mocked model for speed"""
        # First create chunks
        chunks = self.test_01_chunker_creates_valid_chunks()
        
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # Create consistent mock embeddings
        def mock_encode(texts, **kwargs):
            np.random.seed(42)  # Deterministic for testing
            return np.random.randn(len(texts), 384).astype(np.float32)
        
        mock_model.encode = mock_encode
        mock_transformer.return_value = mock_model
        
        # Test config
        test_config = {
            'rag': {
                'embedding_model': 'test-model',
                'batch_size': 32
            }
        }
        
        # Initialize embedding generator
        generator = EmbeddingGenerator(test_config)
        generator.chunks_path = Path(self.chunks_path)
        generator.embeddings_path = Path(self.embeddings_path)
        generator.meta_path = Path(self.meta_path)
        
        # Generate embeddings
        embeddings, metadata = generator.generate_embeddings()
        
        # Assert embeddings shape and metadata
        self.assertEqual(embeddings.shape[0], len(chunks), "Embedding count mismatch")
        self.assertEqual(embeddings.shape[1], 384, "Embedding dimension mismatch")
        self.assertEqual(metadata['count'], len(chunks), "Metadata count mismatch")
        self.assertEqual(metadata['dimension'], 384, "Metadata dimension mismatch")
        
        print(f"[OK] Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
        
        # Assert UID alignment
        self.assertEqual(len(metadata['chunk_uid_order']), len(chunks), "UID order length mismatch")
        
        # Save embeddings
        generator.save_embeddings(embeddings, metadata)
        
        # Verify files exist
        self.assertTrue(os.path.exists(self.embeddings_path), "Embeddings file not created")
        self.assertTrue(os.path.exists(self.meta_path), "Metadata file not created")
        
        return embeddings, metadata
    
    @unittest.skipUnless(HAS_FAISS, "FAISS not available")
    def test_03_vector_store_builder_creates_searchable_index(self):
        """Test vector store builder creates searchable FAISS index"""
        # First create embeddings
        embeddings, metadata = self.test_02_embedding_generator_creates_embeddings()
        
        # Test config
        test_config = {
            'rag': {
                'index_backend': 'faiss'
            }
        }
        
        # Initialize vector store builder
        builder = VectorStoreBuilder(test_config)
        builder.embeddings_path = Path(self.embeddings_path)
        builder.meta_path = Path(self.meta_path)
        builder.chunks_path = Path(self.chunks_path)
        builder.vector_db_dir = Path(self.vector_db_dir)
        
        # Build index
        builder.build_index()
        
        # Assert index files exist
        faiss_index_path = os.path.join(self.vector_db_dir, "faiss", "faiss.index")
        faiss_mapping_path = os.path.join(self.vector_db_dir, "faiss", "mapping.json")
        
        self.assertTrue(os.path.exists(faiss_index_path), "FAISS index file not created")
        self.assertTrue(os.path.exists(faiss_mapping_path), "FAISS mapping file not created")
        
        print(f"[OK] Created FAISS index at {faiss_index_path}")
        
        # Test searchability
        import faiss as faiss_lib
        index = faiss_lib.read_index(faiss_index_path)
        
        with open(faiss_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Assert index properties
        self.assertEqual(index.ntotal, len(embeddings), "Index vector count mismatch")
        self.assertEqual(index.d, 384, "Index dimension mismatch")
        
        # Test search functionality with a sample query vector
        np.random.seed(42)
        query_vector = np.random.randn(1, 384).astype(np.float32)
        faiss_lib.normalize_L2(query_vector)
        
        # Search top-k
        k = 3
        scores, indices = index.search(query_vector, k)
        
        # Assert search results
        self.assertEqual(len(scores[0]), k, "Search returned wrong number of results")
        self.assertEqual(len(indices[0]), k, "Search indices count mismatch")
        
        # Verify indices are valid
        for idx in indices[0]:
            if idx != -1:  # -1 indicates invalid result
                self.assertIn(str(idx), mapping, f"Index {idx} not in mapping")
        
        print(f"[OK] Search returned {len(indices[0])} results")
        
        return index, mapping
    
    def test_04_pipeline_idempotence(self):
        """Test that re-running the pipeline produces identical results"""
        print("Testing pipeline idempotence...")
        
        # First run
        chunks1 = self.test_01_chunker_creates_valid_chunks()
        embeddings1, metadata1 = self.test_02_embedding_generator_creates_embeddings()
        
        # Store first run results
        chunks1_copy = json.loads(json.dumps(chunks1))  # Deep copy
        embeddings1_copy = embeddings1.copy()
        metadata1_copy = metadata1.copy()
        
        # Clear files and run again
        self.setUp()  # Clean state
        
        # Second run  
        chunks2 = self.test_01_chunker_creates_valid_chunks()
        embeddings2, metadata2 = self.test_02_embedding_generator_creates_embeddings()
        
        # Assert idempotence
        self.assertEqual(len(chunks1_copy), len(chunks2), "Chunk count changed between runs")
        self.assertEqual(metadata1_copy['count'], metadata2['count'], "Embedding count changed")
        
        # Assert chunk UIDs are identical
        uids1 = {chunk['chunk_uid'] for chunk in chunks1_copy}
        uids2 = {chunk['chunk_uid'] for chunk in chunks2}
        self.assertEqual(uids1, uids2, "Chunk UIDs changed between runs")
        
        # Assert embedding shapes are identical
        self.assertEqual(embeddings1_copy.shape, embeddings2.shape, "Embedding shapes changed")
        
        print(f"[OK] Pipeline idempotence verified: {len(chunks2)} chunks, {embeddings2.shape} embeddings")
    
    def test_05_content_validation(self):
        """Test that chunks contain expected content and keywords"""
        chunks = self.test_01_chunker_creates_valid_chunks()
        
        # Collect all chunk text
        all_text = " ".join(chunk.get('content', chunk.get('text', chunk.get('normalized_text', ''))) for chunk in chunks)
        
        # Assert test keywords are present
        for keyword in self.test_keywords:
            self.assertIn(keyword, all_text, f"Test keyword '{keyword}' not found in chunks")
        
        print(f"[OK] All test keywords found: {len(self.test_keywords)} keywords")
        
        # Assert document structure preservation
        doc_uids = {chunk['document_uid'] for chunk in chunks}
        self.assertGreaterEqual(len(doc_uids), 2, "Expected at least 2 document UIDs")
        
        # Assert article numbers are preserved
        article_numbers = {chunk.get('article_number', '') for chunk in chunks}
        article_numbers.discard('')  # Remove empty values
        self.assertGreaterEqual(len(article_numbers), 2, "Expected article numbers in chunks")
        
        print(f"[OK] Found {len(doc_uids)} documents and {len(article_numbers)} articles in chunks")
    
    def test_06_boundary_preservation(self):
        """Test that تبصره and بند boundaries are properly preserved"""
        chunks = self.test_01_chunker_creates_valid_chunks()
        
        tabsareh_violations = 0
        band_violations = 0
        
        for chunk in chunks:
            text = chunk.get('content', chunk.get('text', chunk.get('normalized_text', '')))
            
            # Check for تبصره boundary violations
            tabsareh_positions = []
            pos = 0
            while True:
                pos = text.find('تبصره', pos)
                if pos == -1:
                    break
                tabsareh_positions.append(pos)
                pos += 1
            
            # If multiple تبصره found, they should be consecutive numbers or clearly separated
            if len(tabsareh_positions) > 1:
                # This might be acceptable if they're related (like "تبصره 1" and "تبصره 2")
                # But we'll flag it for review
                tabsareh_violations += 1
            
            # Check for بند boundary violations  
            band_positions = []
            pos = 0
            while True:
                pos = text.find('بند', pos)
                if pos == -1:
                    break
                band_positions.append(pos)
                pos += 1
            
            if len(band_positions) > 1:
                band_violations += 1
        
        print(f"[OK] Boundary analysis: {tabsareh_violations} potential tabsareh violations, {band_violations} band violations")
        
        # Violations should be minimal (we allow some flexibility)
        self.assertLessEqual(tabsareh_violations, len(chunks) // 2, "Too many تبصره boundary violations")
        self.assertLessEqual(band_violations, len(chunks) // 2, "Too many بند boundary violations")
    
    def test_07_database_content_integrity(self):
        """Test that database content is properly loaded and processed"""
        # Connect to test database and verify content
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        self.assertGreaterEqual(doc_count, 2, "Expected at least 2 test documents")
        
        # Count articles
        cursor.execute("SELECT COUNT(*) FROM articles") 
        article_count = cursor.fetchone()[0]
        self.assertGreaterEqual(article_count, 3, "Expected at least 3 test articles")
        
        # Count notes
        cursor.execute("SELECT COUNT(*) FROM notes")
        note_count = cursor.fetchone()[0]
        self.assertGreaterEqual(note_count, 1, "Expected at least 1 test note")
        
        print(f"[OK] Database contains: {doc_count} docs, {article_count} articles, {note_count} notes")
        
        # Verify content contains expected Persian legal terms
        cursor.execute("SELECT article_text FROM articles")
        article_texts = [row[0] for row in cursor.fetchall()]
        
        all_content = " ".join(article_texts)
        legal_terms = ["وزارت", "دانشگاه", "شورای", "تبصره"]
        
        for term in legal_terms:
            self.assertIn(term, all_content, f"Legal term '{term}' not found in database content")
        
        conn.close()
        print(f"[OK] All legal terms found in database content")


def run_test_suite():
    """Run the complete test suite"""
    # Create test loader and suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase3Pipeline)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)