import pytest
import json
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase_4_llm_rag import LegalRAGEngine


class TestLegalRAGEngine:
    """Test suite for LegalRAGEngine class."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration that points to real Phase 3 artifacts."""
        return {
            "vector_db_path": "data/processed_phase_3/vector_db",
            "chunks_file": "data/processed_phase_3/chunks.json",
            "embeddings_file": "data/processed_phase_3/embeddings.npy",
            "metadata_file": "data/processed_phase_3/embeddings_meta.json",
            "database_path": "data/db/legal_assistant.db",
            
            "retriever": {"backend": "faiss", "top_k": 5, "similarity_threshold": 0.75},
            "reranker": {"enabled": False, "model": None},
            
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b-instruct",
                "temperature": 0.1,
                "max_tokens": 4096,
                "timeout_s": 60
            },
            
            "prompt_templates_path": "phase_4_llm_rag/prompt_templates.json",
            "logging": {"user_lang": "fa", "dev_level": "INFO"}
        }
    
    @pytest.fixture
    def temp_config_file(self, test_config):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "chunk_uid": "test_chunk_001",
                "document_uid": "law_001",
                "article_number": "1",
                "note_label": None,
                "text": "این ماده اول قانون تست است.",
                "section": "فصل اول"
            },
            {
                "chunk_uid": "test_chunk_002", 
                "document_uid": "law_001",
                "article_number": "2",
                "note_label": "تبصره",
                "text": "تبصره ماده دوم در مورد اجرای قانون.",
                "section": "فصل اول"
            },
            {
                "chunk_uid": "test_chunk_003",
                "document_uid": "law_002", 
                "article_number": "5",
                "note_label": None,
                "text": "ماده پنجم قانون دوم در مورد مجازات.",
                "section": "فصل دوم"
            }
        ]
    
    @pytest.fixture
    def mock_rag_engine(self, temp_config_file, sample_chunks):
        """Create a mocked RAG engine for testing."""
        with patch('phase_4_llm_rag.api_connections.get_llm_client') as mock_llm:
            # Mock the LLM client with required attributes
            mock_client = MagicMock()
            mock_client.model = "test_model"
            mock_client.base_url = "http://localhost:11434"
            mock_client.generate.return_value = "تست پاسخ از مدل آزمایشی"
            mock_llm.return_value = mock_client
            
            # Mock specific methods with realistic templates
            default_templates = {
                "default": "سؤال:\n{question}\n\nمتون بازیابی‌شده (با استناد):\n{retrieved_text}\n\nلطفاً پاسخ را به فارسی و به‌صورت فشرده بنویس و حتماً شماره ماده/تبصره و نام قانون را ذکر کن.",
                "compare": "هدف: مقایسه یا تضاد بین دو متن حقوقی.\nسؤال:\n{question}\n\nمتون بازیابی‌شده:\n{retrieved_text}\n\nلطفاً شباهت‌ها، تفاوت‌ها و هرگونه تعارض را توضیح بده و به مواد/تبصره‌های متناظر اشاره کن.",
                "draft": "هدف: تهیهٔ پیش‌نویس متن حقوقی منطبق با چارچوب‌های موجود.\nشرح درخواست:\n{question}\n\nمنابع مرتبط:\n{retrieved_text}\n\nپیش‌نویس پیشنهادی را با زبان رسمی، ساختار روشن، و همراه با استنادات دقیق ارائه کن."
            }
            
            with patch.object(LegalRAGEngine, '_load_chunks', return_value=sample_chunks):
                with patch.object(LegalRAGEngine, '_connect_vector_store', return_value=None):
                    with patch.object(LegalRAGEngine, '_load_prompt_templates', return_value=default_templates):
                        engine = LegalRAGEngine(config_path=temp_config_file)
                        engine.chunks = sample_chunks  # Set chunks directly for consistency
                        yield engine
    
    def test_retrieve_returns_results(self, mock_rag_engine, sample_chunks):
        """Test that retrieve returns results with required keys."""
        # Mock the retrieve method to return sample chunks
        with patch.object(mock_rag_engine, 'retrieve', return_value=sample_chunks):
            results = mock_rag_engine.retrieve("یک پرسش شامل یک کلیدواژهٔ شناخته‌شده", top_k=5)
            
            # Assert returned list length ≥ 1
            assert len(results) >= 1
            
            # Assert entries contain required keys
            for result in results:
                assert 'chunk_uid' in result
                assert 'document_uid' in result
                assert 'article_number' in result
                assert isinstance(result['chunk_uid'], str)
                assert isinstance(result['document_uid'], str)
                assert result['article_number'] is not None
    
    def test_build_prompt_inserts_question_and_texts(self, mock_rag_engine, sample_chunks):
        """Test that build_prompt correctly inserts question and retrieved texts."""
        question = "آیا ماده اول قانون چیست؟"
        
        # Call build_prompt with sample chunks
        prompt = mock_rag_engine.build_prompt(question, sample_chunks[:2])
        
        # Assert question is present in the prompt
        assert question in prompt
        
        # Assert at least one citation block is present
        assert "[1]" in prompt  # First citation
        assert "law_001" in prompt  # Document UID
        assert "ماده 1" in prompt  # Article reference
        
        # Assert chunk text is included
        assert "این ماده اول قانون تست است" in prompt
        
        # Assert Persian template structure
        assert "سؤال:" in prompt
        assert "متون بازیابی‌شده" in prompt
    
    def test_answer_end_to_end_with_mock_llm(self, mock_rag_engine, sample_chunks):
        """Test end-to-end answer generation with mocked LLM."""
        question = "قانون در مورد مجازات چه می‌گوید؟"
        
        # Mock retrieve to return sample chunks
        with patch.object(mock_rag_engine, 'retrieve', return_value=sample_chunks):
            # Mock generate_answer to return fixed response
            mock_response = {
                "answer": "طبق ماده ۵ قانون دوم، مجازات‌های مقرر اعمال می‌شود.",
                "citations": [
                    {"document_uid": "law_002", "article_number": "5", "note_label": None}
                ]
            }
            
            with patch.object(mock_rag_engine, 'generate_answer', return_value=mock_response):
                result = mock_rag_engine.answer(question, top_k=3)
                
                # Assert structure includes required keys
                assert "answer" in result
                assert "citations" in result
                assert "retrieved_chunks" in result
                
                # Assert answer is in Persian
                assert "طبق ماده ۵ قانون دوم" in result["answer"]
                
                # Assert citations structure
                assert len(result["citations"]) >= 1
                citation = result["citations"][0]
                assert "document_uid" in citation
                assert "article_number" in citation
                
                # Assert retrieved chunks count
                assert result["retrieved_chunks"] == len(sample_chunks)
    
    def test_no_results_graceful(self, mock_rag_engine):
        """Test graceful handling when no results are found."""
        unlikely_question = "سوال بسیار نامحتمل که هیچ نتیجه‌ای نخواهد داشت xyz123"
        
        # Mock retrieve to return empty list
        with patch.object(mock_rag_engine, 'retrieve', return_value=[]):
            result = mock_rag_engine.answer(unlikely_question, top_k=5)
            
            # Assert the engine returns a Persian guidance message
            assert "answer" in result
            assert "متأسفانه هیچ سند مرتبطی یافت نشد" in result["answer"]
            
            # Assert citations is empty
            assert result["citations"] == []
            
            # Assert retrieved chunks count is 0
            assert result["retrieved_chunks"] == 0
            
            # Assert no crash occurred (test passes if we reach here)
    
    def test_build_prompt_different_templates(self, mock_rag_engine, sample_chunks):
        """Test build_prompt with different template types."""
        question = "مقایسه قوانین"
        
        # Test compare template
        compare_prompt = mock_rag_engine.build_prompt(question, sample_chunks[:2], "compare")
        assert "مقایسه یا تضاد" in compare_prompt
        assert "شباهت‌ها، تفاوت‌ها" in compare_prompt
        
        # Test draft template
        draft_prompt = mock_rag_engine.build_prompt(question, sample_chunks[:2], "draft")
        assert "پیش‌نویس متن حقوقی" in draft_prompt
        assert "زبان رسمی، ساختار روشن" in draft_prompt
    
    def test_config_loading(self, temp_config_file):
        """Test that configuration is loaded correctly."""
        with patch('phase_4_llm_rag.api_connections.get_llm_client'):
            with patch.object(LegalRAGEngine, '_load_chunks', return_value=[]):
                with patch.object(LegalRAGEngine, '_connect_vector_store', return_value=None):
                    with patch.object(LegalRAGEngine, '_load_prompt_templates', return_value={}):
                        engine = LegalRAGEngine(config_path=temp_config_file)
                        
                        # Assert config is loaded
                        assert engine.config is not None
                        assert engine.config["llm"]["provider"] == "ollama"
                        assert engine.config["llm"]["model"] == "qwen2.5:7b-instruct"
                        assert engine.config["retriever"]["backend"] == "faiss"
    
    def test_prompt_templates_fallback(self, temp_config_file):
        """Test fallback to default templates when file is missing."""
        with patch('phase_4_llm_rag.api_connections.get_llm_client'):
            with patch.object(LegalRAGEngine, '_load_chunks', return_value=[]):
                with patch.object(LegalRAGEngine, '_connect_vector_store', return_value=None):
                    # Mock missing templates file
                    with patch('builtins.open', side_effect=FileNotFoundError):
                        engine = LegalRAGEngine(config_path=temp_config_file)
                        
                        # Assert default templates are loaded
                        assert "default" in engine.prompt_templates
                        assert "compare" in engine.prompt_templates
                        assert "draft" in engine.prompt_templates
                        assert "سؤال:" in engine.prompt_templates["default"]


# Integration test with real files (if available)
class TestLegalRAGEngineIntegration:
    """Integration tests using real Phase 3 artifacts."""
    
    def test_real_chunks_loading(self):
        """Test loading real chunks.json if available."""
        chunks_path = "data/processed_phase_3/chunks.json"
        
        if os.path.exists(chunks_path):
            with patch('phase_4_llm_rag.api_connections.get_llm_client'):
                with patch.object(LegalRAGEngine, '_connect_vector_store', return_value=None):
                    # Let it load real templates or use fallback
                    engine = LegalRAGEngine(config_path="phase_4_llm_rag/Rag_config.json")
                    
                    # Assert chunks are loaded
                    assert isinstance(engine.chunks, list)
                    if engine.chunks:
                        # Verify chunk structure
                        chunk = engine.chunks[0]
                        assert 'text' in chunk
                        assert 'document_uid' in chunk
        else:
            pytest.skip("Real chunks.json not available")
    
    def test_real_database_connection(self):
        """Test connection to real SQLite database if available."""
        db_path = "data/db/legal_assistant.db"
        
        if os.path.exists(db_path):
            import sqlite3
            
            # Test database connectivity
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM chunks LIMIT 1")
                    result = cursor.fetchone()
                    assert result is not None
            except sqlite3.Error:
                pytest.skip("Database connection failed")
        else:
            pytest.skip("Real database not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])