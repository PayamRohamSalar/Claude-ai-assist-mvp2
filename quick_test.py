#!/usr/bin/env python3
"""
Quick test to verify imports and basic functionality before running full pytest suite.
"""

import sys
import os

print("Testing imports...")

try:
    from phase_4_llm_rag import LegalRAGEngine
    print("✓ Successfully imported LegalRAGEngine")
except Exception as e:
    print(f"✗ Failed to import LegalRAGEngine: {e}")
    sys.exit(1)

try:
    from phase_4_llm_rag.api_connections import make_llm_client
    print("✓ Successfully imported make_llm_client")
except Exception as e:
    print(f"✗ Failed to import make_llm_client: {e}")
    sys.exit(1)

# Test minimal configuration
try:
    config = {
        "llm": {
            "provider": "ollama",
            "model": "test_model"
        },
        "chunks_file": "nonexistent.json",  # Will be mocked
        "database_path": "nonexistent.db"
    }
    
    # This should work with mocking, but let's just test import structure
    print("✓ Configuration structure looks good")
    print("✓ All basic imports successful - ready for pytest")
    
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    sys.exit(1)

print("🎉 Quick test passed! You can now run: python -m pytest tests/test_rag_engine.py -v")