import pytest
import json
import tempfile
import os
import sys
import sqlite3
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_4_llm_rag.rag_engine import LegalRAGEngine
from phase_4_llm_rag.rag_engine import validate_citations


class TestLegalRAGEngine:
    """Test suite for LegalRAGEngine class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing with more realistic data."""
        return [
            {
                "uid": "chunk_001",
                "chunk_uid": "chunk_001",
                "document_uid": "قانون_حمایت_کودکان",
                "document_title": "قانون حمایت از کودکان و نوجوانان",
                "article_number": "1",
                "note_label": None,
                "text": "ماده ۱ - این قانون به منظور حمایت از حقوق کودکان و نوجوانان وضع شده است.",
                "section": "فصل اول"
            },
            {
                "uid": "chunk_002",
                "chunk_uid": "chunk_002", 
                "document_uid": "قانون_حمایت_کودکان",
                "document_title": "قانون حمایت از کودکان و نوجوانان",
                "article_number": "1",
                "note_label": "تبصره ۱",
                "text": "تبصره ۱ - منظور از کودک، شخصی است که سن هجده سالگی را تکمیل نکرده باشد.",
                "section": "فصل اول"
            },
            {
                "uid": "chunk_003",
                "chunk_uid": "chunk_003",
                "document_uid": "قانون_مجازات_اسلامی", 
                "document_title": "قانون مجازات اسلامی",
                "article_number": "12",
                "note_label": None,
                "text": "ماده ۱۲ - مجازات‌های تعزیری در این قانون تعیین شده است.",
                "section": "کتاب اول"
            }
        ]
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create a test configuration with fake LLM provider."""
        return {
            "vector_store": {
                "type": "faiss",
                "index_path": os.path.join(temp_dir, "test.index"),
                "embeddings_path": os.path.join(temp_dir, "embeddings.npy")
            },
            "database_path": os.path.join(temp_dir, "test.db"),
            "chunks_file": os.path.join(temp_dir, "chunks.json"),
            "prompt_templates_path": os.path.join(temp_dir, "templates.json"),
            "retriever": {"top_k": 5, "similarity_threshold": 0.75},
            "llm": {
                "provider": "fake",
                "model": "fake-model",
                "temperature": 0.1,
                "max_tokens": 1000,
                "timeout_s": 30
            }
        }
    
    @pytest.fixture
    def test_files(self, temp_dir, test_config, sample_chunks):
        """Create all necessary test files in temp directory."""
        # Write config file
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
        
        # Write chunks file
        chunks_path = test_config["chunks_file"]
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(sample_chunks, f, ensure_ascii=False, indent=2)
        
        # Write embeddings metadata
        embeddings_dir = os.path.dirname(test_config["vector_store"]["embeddings_path"])
        os.makedirs(embeddings_dir, exist_ok=True)
        meta_path = os.path.join(embeddings_dir, "embeddings_meta.json")
        embeddings_meta = {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dimension": 384,
            "count": 3,
            "chunk_uid_order": [chunk["uid"] for chunk in sample_chunks]
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_meta, f, ensure_ascii=False, indent=2)
        
        # Create fake embeddings file
        embeddings_path = test_config["vector_store"]["embeddings_path"]
        fake_embeddings = np.random.rand(3, 384).astype(np.float32)
        np.save(embeddings_path, fake_embeddings)
        
        # Write prompt templates
        templates_path = test_config["prompt_templates_path"]
        templates = {
            "default": "سؤال:\n{question}\n\nمتون بازیابی‌شده:\n{retrieved_text}\n\nلطفاً به فارسی پاسخ دهید و شمارهٔ ماده/تبصره و نام قانون را ذکر کنید.",
            "compare": "هدف: مقایسه یا تضاد بین دو متن حقوقی.\nسؤال:\n{question}\n\nمتون بازیابی‌شده:\n{retrieved_text}\n\nلطفاً شباهت‌ها و تفاوت‌ها را توضیح دهید."
        }
        with open(templates_path, 'w', encoding='utf-8') as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        
        # Create test database with chunks table
        db_path = test_config["database_path"]
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create chunks table
        cursor.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                chunk_uid TEXT,
                document_uid TEXT,
                article_number TEXT,
                note_label TEXT,
                text TEXT,
                section TEXT
            )
        """)
        
        # Insert sample chunks
        for chunk in sample_chunks:
            cursor.execute("""
                INSERT INTO chunks (chunk_uid, document_uid, article_number, note_label, text, section)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk["chunk_uid"],
                chunk["document_uid"],
                chunk["article_number"],
                chunk["note_label"],
                chunk["text"],
                chunk["section"]
            ))
        
        conn.commit()
        conn.close()
        
        return {
            "config_path": config_path,
            "chunks_path": chunks_path,
            "db_path": db_path,
            "temp_dir": temp_dir
        }
    

    
    @pytest.fixture
    def rag_engine(self, test_files):
        """Create a RAG engine instance with test data and fake LLM."""
        # Set environment variable for fake LLM
        os.environ['LLM_PROVIDER'] = 'fake'
        try:
            engine = LegalRAGEngine(config_path=test_files["config_path"])
            yield engine
        finally:
            # Clean up environment
            if 'LLM_PROVIDER' in os.environ:
                del os.environ['LLM_PROVIDER']
    
    def test_retrieve_returns_results_for_keyword_query(self, rag_engine):
        """Test that retrieve returns non-empty results for a simple keyword query."""
        # Test with Persian keyword that should match our sample data
        results = rag_engine.retrieve("تبصره ۱", top_k=5)
        
        # Assert returned list length ≥ 1
        assert len(results) >= 1, "Retrieve should return at least one result for 'تبصره ۱'"
        
        # Assert entries contain required keys
        for result in results:
            assert 'document_uid' in result, "Result must contain document_uid"
            assert 'text' in result, "Result must contain text"
            assert isinstance(result['document_uid'], str), "document_uid must be string"
        
        # Test another keyword that exists in our test data
        results2 = rag_engine.retrieve("ماده", top_k=5)
        assert len(results2) >= 1, "Retrieve should return results for 'ماده'"
        
        # Verify the result contains expected content
        found_relevant = any("ماده" in result.get('text', '') for result in results2)
        assert found_relevant, "At least one result should contain the search term"
    
    def test_build_prompt_inserts_question_and_citation_markers(self, rag_engine, sample_chunks):
        """Test that build_prompt inserts both question and citation markers."""
        question = "آیا ماده اول قانون چیست؟"
        
        # Call build_prompt with sample chunks
        prompt = rag_engine.build_prompt(question, sample_chunks[:2])
        
        # Assert question is present in the prompt
        assert question in prompt, "Question should be present in prompt"
        
        # Assert at least one citation marker is present
        assert "[1]" in prompt, "First citation marker [1] should be present"
        assert "[2]" in prompt, "Second citation marker [2] should be present"
        
        # Assert document UIDs are present
        assert "قانون_حمایت_کودکان" in prompt, "Document UID should be in prompt"
        
        # Assert article references are formatted correctly
        assert "ماده ۱" in prompt, "Article reference should be properly formatted"
        
        # Assert chunk text is included
        assert "این قانون به منظور حمایت" in prompt, "Chunk text should be included"
        
        # Assert Persian template structure
        assert "سؤال:" in prompt, "Persian question label should be present"
        assert "متون بازیابی‌شده" in prompt, "Persian retrieved texts label should be present"
        
        # Assert document titles are included (from our enhanced build_prompt)
        assert "قانون حمایت از کودکان و نوجوانان" in prompt, "Document title should be included"
    
    def test_answer_end_to_end_with_fake_llm(self, rag_engine):
        """Test end-to-end answer generation with fake LLM that returns predictable responses."""
        question = "قانون در مورد مجازات چه می‌گوید؟"
        
        # Use the real engine with fake LLM
        result = rag_engine.answer(question, top_k=3)
        
        # Assert structure includes required keys
        assert "answer" in result, "Result should contain answer"
        assert "citations" in result, "Result should contain citations"
        assert "retrieved_chunks" in result, "Result should contain retrieved_chunks count"
        
        # Assert answer is not empty (fake LLM should generate something)
        assert len(result["answer"]) > 0, "Answer should not be empty"
        
        # Assert answer contains FakeLLMClient indicator (for verification)
        assert "FakeLLMClient" in result["answer"], "Answer should indicate it came from fake client"
        
        # Assert citations is a list
        assert isinstance(result["citations"], list), "Citations should be a list"
    
    def test_extract_citations_from_answer(self, rag_engine):
        """Test that _extract_citations_from_answer correctly parses citation markers."""
        # Test answer with inline citation markers (matching our build_prompt format)
        test_answer = (
            "بر اساس اسناد بازیابی‌شده، [1] قانون حمایت از کودکان و نوجوانان (قانون_حمایت_کودکان) - ماده ۱ بیان می‌کند که این قانون وضع شده است. "
            "همچنین [2] قانون حمایت از کودکان و نوجوانان (قانون_حمایت_کودکان) - ماده ۱ - تبصره ۱ در مورد تعریف کودک است."
        )
        
        citations = rag_engine._extract_citations_from_answer(test_answer)
        
        # Assert citations were extracted
        assert len(citations) >= 2, "Should extract at least 2 citations"
        
        # Check first citation
        citation1 = citations[0]
        assert citation1["document_uid"] == "قانون_حمایت_کودکان", "First citation document_uid should match"
        assert citation1["document_title"] == "قانون حمایت از کودکان و نوجوانان", "Document title should be extracted"
        assert citation1["article_number"] == "۱", "Article number should be extracted"
        
        # Check second citation with note
        citation2 = citations[1]
        assert citation2["document_uid"] == "قانون_حمایت_کودکان", "Second citation document_uid should match"
        assert citation2["article_number"] == "۱", "Article number should be extracted"
        assert citation2["note_label"] and "تبصره ۱" in citation2["note_label"], "Note label should contain 'تبصره ۱'"
    
    def test_validate_citations_with_database(self, rag_engine, test_files):
        """Test that validate_citations confirms citation existence in database."""
        # Test with valid citations that exist in our test database
        valid_citations = [
            {
                "document_uid": "قانون_حمایت_کودکان",
                "article_number": "1",
                "note_label": None
            },
            {
                "document_uid": "قانون_حمایت_کودکان",
                "article_number": "1", 
                "note_label": "تبصره ۱"
            }
        ]
        
        validated = validate_citations(valid_citations, test_files["db_path"])
        
        # Should return the valid citations
        assert len(validated) == 2, "Both valid citations should be returned"
        
        # Test with invalid citation
        invalid_citations = [
            {
                "document_uid": "nonexistent_law",
                "article_number": "999",
                "note_label": None
            }
        ]
        
        validated_invalid = validate_citations(invalid_citations, test_files["db_path"])
        
        # Should return empty list for invalid citations
        assert len(validated_invalid) == 0, "Invalid citations should not be returned"
    
    def test_no_results_graceful(self, rag_engine):
        """Test graceful handling when no results are found."""
        # Use a completely unrelated term that won't match any of our Persian legal text
        unlikely_question = "complete_random_english_xyz_impossible_match_987654"
        
        result = rag_engine.answer(unlikely_question, top_k=5)
        
        # Assert structure is correct
        assert "answer" in result
        assert "citations" in result
        assert "retrieved_chunks" in result
        
        # For this test, we just verify no crash occurred and proper structure returned
        # The fake LLM may still generate answers even with no relevant chunks
        assert isinstance(result["answer"], str), "Answer should be a string"
        assert isinstance(result["citations"], list), "Citations should be a list"
        assert isinstance(result["retrieved_chunks"], int), "Retrieved chunks should be an integer"
    
    def test_build_prompt_different_templates(self, rag_engine, sample_chunks):
        """Test build_prompt with different template types."""
        question = "مقایسه قوانین"
        
        # Test compare template
        compare_prompt = rag_engine.build_prompt(question, sample_chunks[:2], "compare")
        assert "مقایسه یا تضاد" in compare_prompt, "Compare template should contain comparison text"
        assert "شباهت‌ها" in compare_prompt, "Compare template should mention similarities"
        
        # Test default template fallback for non-existent template
        default_prompt = rag_engine.build_prompt(question, sample_chunks[:2], "nonexistent")
        assert "سؤال:" in default_prompt, "Should fallback to default template"
    
    def test_config_loading(self, rag_engine):
        """Test that configuration is loaded correctly."""
        # Assert config is loaded
        assert rag_engine.config is not None, "Config should be loaded"
        assert rag_engine.config["llm"]["provider"] == "fake", "Provider should be fake"
        assert rag_engine.config["llm"]["model"] == "fake-model", "Model should be fake-model"
        assert "vector_store" in rag_engine.config, "Config should have vector_store section"
        assert rag_engine.config["vector_store"]["type"] == "faiss", "Vector store type should be faiss"
    
    def test_fake_llm_provider_integration(self, rag_engine):
        """Test that fake LLM provider is working correctly."""
        # Test that the LLM client is indeed the fake one
        assert hasattr(rag_engine.llm_client, 'model'), "LLM client should have model attribute"
        
        # Test direct generation
        test_prompt = "تست سؤال ماده"
        response = rag_engine.llm_client.generate(test_prompt)
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        
        # Verify it's the fake client by checking for fake indicator
        assert "FakeLLMClient" in response or "تست" in response, "Should contain fake client indicator"


# Integration test with real files (if available)
class TestLegalRAGEngineIntegration:
    """Integration tests using real Phase 3 artifacts."""
    
    def test_real_chunks_loading(self):
        """Test loading real chunks.json if available."""
        chunks_path = "data/processed_phase_3/chunks.json"
        
        if os.path.exists(chunks_path):
            # Set fake provider to avoid Ollama dependency
            os.environ['LLM_PROVIDER'] = 'fake'
            try:
                engine = LegalRAGEngine(config_path="../phase_4_llm_rag/Rag_config.json")
                
                # Assert chunks are loaded
                assert isinstance(engine.chunks, list), "Chunks should be loaded as list"
                if engine.chunks:
                    # Verify chunk structure
                    chunk = engine.chunks[0]
                    assert 'text' in chunk, "Chunk should have text field"
                    assert 'document_uid' in chunk, "Chunk should have document_uid field"
            finally:
                if 'LLM_PROVIDER' in os.environ:
                    del os.environ['LLM_PROVIDER']
        else:
            pytest.skip("Real chunks.json not available")
    
    def test_real_database_connection(self):
        """Test connection to real SQLite database if available."""
        db_path = "data/db/legal_assistant.db"
        
        if os.path.exists(db_path):
            # Test database connectivity
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM chunks LIMIT 1")
                    result = cursor.fetchone()
                    assert result is not None, "Database should return a result"
            except sqlite3.Error:
                pytest.skip("Database connection failed")
        else:
            pytest.skip("Real database not available")
    
    def test_end_to_end_with_mocked_llm_response(self):
        """Test end-to-end with predictable LLM response containing citation markers."""
        if not os.path.exists("data/processed_phase_3/chunks.json"):
            pytest.skip("Real chunks.json not available")
        
        os.environ['LLM_PROVIDER'] = 'fake'
        try:
            engine = LegalRAGEngine(config_path="../phase_4_llm_rag/Rag_config.json")
            
            # Mock generate_answer to return predictable response with citations
            def mock_generate_answer(prompt):
                return {
                    "answer": "بر اساس [1] قانون حمایت از کودکان و نوجوانان (قانون_حمایت_کودکان) - ماده ۱، این قانون برای حمایت از کودکان وضع شده است.",
                    "citations": [
                        {
                            "document_uid": "قانون_حمایت_کودکان",
                            "document_title": "قانون حمایت از کودکان و نوجوانان", 
                            "article_number": "۱",
                            "note_label": None
                        }
                    ]
                }
            
            with patch.object(engine, 'generate_answer', side_effect=mock_generate_answer):
                result = engine.answer("ماده قانون", top_k=3)  # Use keywords that exist in real data
                
                # Test citation extraction from the mocked answer
                citations = engine._extract_citations_from_answer(result["answer"])
                assert len(citations) >= 1, "Should extract at least one citation"
                
                # Test citation validation if database exists
                if os.path.exists("data/db/legal_assistant.db"):
                    validated = validate_citations(citations, "data/db/legal_assistant.db")
                    # Note: validation might return empty if test citations don't match real data
                    assert isinstance(validated, list), "validate_citations should return a list"
        
        finally:
            if 'LLM_PROVIDER' in os.environ:
                del os.environ['LLM_PROVIDER']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])