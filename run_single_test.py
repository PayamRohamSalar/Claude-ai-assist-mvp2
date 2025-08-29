#!/usr/bin/env python3
"""
Run a single test to verify fixes.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def run_single_test():
    """Run just one test method to check if fixes work."""
    try:
        # Import test requirements
        import pytest
        from unittest.mock import patch, MagicMock
        import json
        import tempfile
        
        # Import our module
        from phase_4_llm_rag import LegalRAGEngine
        
        print("✓ All imports successful")
        
        # Create a simple test config
        config = {
            "vector_db_path": "data/processed_phase_3/vector_db",
            "chunks_file": "data/processed_phase_3/chunks.json",
            "database_path": "data/db/legal_assistant.db",
            "prompt_templates_path": "phase_4_llm_rag/prompt_templates.json",
            "llm": {"provider": "ollama", "model": "test_model"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            temp_config = f.name
        
        sample_chunks = [
            {
                "chunk_uid": "test_chunk_001",
                "document_uid": "law_001", 
                "article_number": "1",
                "note_label": None,
                "text": "این ماده اول قانون تست است."
            }
        ]
        
        templates = {
            "default": "سؤال:\n{question}\n\nمتون بازیابی‌شده:\n{retrieved_text}\n\nپاسخ:"
        }
        
        try:
            # Test engine creation with mocks
            with patch('phase_4_llm_rag.api_connections.get_llm_client') as mock_llm:
                mock_client = MagicMock()
                mock_client.model = "test_model"
                mock_client.base_url = "http://localhost:11434"
                mock_client.generate.return_value = "تست پاسخ"
                mock_llm.return_value = mock_client
                
                with patch.object(LegalRAGEngine, '_load_chunks', return_value=sample_chunks):
                    with patch.object(LegalRAGEngine, '_connect_vector_store', return_value=None):
                        with patch.object(LegalRAGEngine, '_load_prompt_templates', return_value=templates):
                            
                            print("Creating RAG engine...")
                            engine = LegalRAGEngine(config_path=temp_config)
                            print(f"✓ Engine created with model: {engine.llm_client.model}")
                            
                            # Test build_prompt
                            print("Testing build_prompt...")
                            question = "تست سؤال"
                            prompt = engine.build_prompt(question, sample_chunks[:1])
                            assert question in prompt
                            assert "این ماده اول قانون تست است" in prompt
                            print("✓ build_prompt works correctly")
                            
                            # Test generate_answer
                            print("Testing generate_answer...")
                            result = engine.generate_answer("تست prompt")
                            assert "answer" in result
                            assert "citations" in result
                            print("✓ generate_answer works correctly")
                            
                            print("🎉 Single test passed!")
                            return True
                            
        finally:
            # Clean up temp file
            if os.path.exists(temp_config):
                os.unlink(temp_config)
        
    except Exception as e:
        import traceback
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if run_single_test():
        print("\n✅ Ready to run full test suite!")
        print("Run: python -m pytest tests/test_rag_engine.py -v")
    else:
        print("\n❌ Fix the issues above before running full tests")
        sys.exit(1)