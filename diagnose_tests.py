#!/usr/bin/env python3
"""
Diagnostic script to identify test errors step by step.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_step(step_name, func):
    """Test a step and report results."""
    print(f"\n{'='*50}")
    print(f"Testing: {step_name}")
    print('='*50)
    try:
        result = func()
        print(f"✓ {step_name} - SUCCESS")
        if result:
            print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"✗ {step_name} - FAILED")
        print(f"Error: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        return False

def test_basic_imports():
    """Test basic imports."""
    from phase_4_llm_rag import LegalRAGEngine
    from phase_4_llm_rag.api_connections import make_llm_client
    return "Imports successful"

def test_mock_imports():
    """Test pytest and mock imports."""
    import pytest
    from unittest.mock import patch, MagicMock
    return "Mock imports successful"

def test_temp_config():
    """Test creating temporary config."""
    import json
    import tempfile
    
    config = {
        "vector_db_path": "data/processed_phase_3/vector_db",
        "chunks_file": "data/processed_phase_3/chunks.json",
        "database_path": "data/db/legal_assistant.db",
        "llm": {"provider": "ollama", "model": "test"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    # Clean up
    os.unlink(temp_path)
    return "Temp config creation successful"

def test_mock_engine_creation():
    """Test creating mocked engine."""
    import json
    import tempfile
    from unittest.mock import patch, MagicMock
    from phase_4_llm_rag import LegalRAGEngine
    
    # Create temp config
    config = {
        "vector_db_path": "data/processed_phase_3/vector_db",
        "chunks_file": "data/processed_phase_3/chunks.json",
        "database_path": "data/db/legal_assistant.db",
        "prompt_templates_path": "phase_4_llm_rag/prompt_templates.json",
        "llm": {"provider": "ollama", "model": "test"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    sample_chunks = [{"chunk_uid": "test", "document_uid": "doc", "article_number": "1", "text": "test"}]
    
    try:
        with patch('phase_4_llm_rag.api_connections.get_llm_client') as mock_llm:
            # Mock LLM client
            mock_client = MagicMock()
            mock_client.model = "test_model"
            mock_client.base_url = "http://localhost:11434"
            mock_client.generate.return_value = "test response"
            mock_llm.return_value = mock_client
            
            with patch.object(LegalRAGEngine, '_load_chunks', return_value=sample_chunks):
                with patch.object(LegalRAGEngine, '_connect_vector_store', return_value=None):
                    engine = LegalRAGEngine(config_path=temp_path)
                    return f"Engine created successfully with model: {engine.llm_client.model}"
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_retrieve_method():
    """Test retrieve method."""
    import json
    import tempfile
    from unittest.mock import patch, MagicMock
    from phase_4_llm_rag import LegalRAGEngine
    
    config = {
        "vector_db_path": "data/processed_phase_3/vector_db",
        "chunks_file": "data/processed_phase_3/chunks.json",
        "database_path": "data/db/legal_assistant.db",
        "prompt_templates_path": "phase_4_llm_rag/prompt_templates.json",
        "llm": {"provider": "ollama", "model": "test"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    sample_chunks = [
        {"chunk_uid": "test1", "document_uid": "doc1", "article_number": "1", "text": "test 1"},
        {"chunk_uid": "test2", "document_uid": "doc2", "article_number": "2", "text": "test 2"}
    ]
    
    try:
        with patch('phase_4_llm_rag.api_connections.get_llm_client') as mock_llm:
            mock_client = MagicMock()
            mock_client.model = "test_model"
            mock_client.base_url = "http://localhost:11434"
            mock_client.generate.return_value = "test response"
            mock_llm.return_value = mock_client
            
            with patch.object(LegalRAGEngine, '_load_chunks', return_value=sample_chunks):
                with patch.object(LegalRAGEngine, '_connect_vector_store', return_value=None):
                    engine = LegalRAGEngine(config_path=temp_path)
                    
                    # Mock retrieve to return sample chunks
                    with patch.object(engine, 'retrieve', return_value=sample_chunks):
                        results = engine.retrieve("test query")
                        return f"Retrieved {len(results)} chunks"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_single_pytest():
    """Test running a single pytest test."""
    try:
        # Try to run a simple test
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_rag_engine.py::TestLegalRAGEngine::test_retrieve_returns_results', 
            '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        return f"Subprocess failed: {e}"

def main():
    """Run all diagnostic tests."""
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Mock Imports", test_mock_imports), 
        ("Temp Config Creation", test_temp_config),
        ("Mock Engine Creation", test_mock_engine_creation),
        ("Retrieve Method", test_retrieve_method),
        ("Single PyTest", test_single_pytest)
    ]
    
    print("🔍 DIAGNOSTIC TEST SUITE")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if test_step(test_name, test_func):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        print("❌ Issues found - see details above")
        sys.exit(1)
    else:
        print("✅ All diagnostics passed!")

if __name__ == "__main__":
    main()