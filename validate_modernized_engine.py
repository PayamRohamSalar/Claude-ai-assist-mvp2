#!/usr/bin/env python3
"""
Simple validation script for the modernized LegalRAGEngine
"""

import sys
import os
import json
from pathlib import Path

def validate_config_compatibility():
    """Validate that the modernized engine can load the new config schema."""
    print("🔍 Validating config compatibility...")
    
    # Check if the new config file exists
    config_path = "phase_4_llm_rag/Rag_config.json"
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Load and validate the config structure
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check required keys
        required_keys = ["vector_store", "database_path", "chunks_file", "llm"]
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing required config keys: {missing_keys}")
            return False
        
        # Check vector_store structure
        vector_store = config.get("vector_store", {})
        if "type" not in vector_store:
            print("❌ vector_store.type is missing")
            return False
        
        if "index_path" not in vector_store:
            print("❌ vector_store.index_path is missing")
            return False
        
        # Check llm structure
        llm = config.get("llm", {})
        if "model" not in llm:
            print("❌ llm.model is missing")
            return False
        
        print("✅ Config structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def validate_engine_initialization():
    """Validate that the modernized engine can be initialized."""
    print("🔍 Validating engine initialization...")
    
    try:
        # Import the modernized engine
        sys.path.insert(0, '.')
        from rag_engine import LegalRAGEngine
        
        # Try to initialize with the new config
        engine = LegalRAGEngine('phase_4_llm_rag/Rag_config.json')
        
        # Check basic stats
        stats = engine.get_stats()
        print(f"✅ Engine initialized successfully")
        print(f"   - Chunks loaded: {stats.get('chunks_loaded', 0)}")
        print(f"   - Vector backend: {stats.get('vector_backend', 'None')}")
        print(f"   - Database connected: {stats.get('database_connected', False)}")
        print(f"   - LLM client available: {stats.get('llm_client_available', False)}")
        
        engine.close()
        return True
        
    except Exception as e:
        print(f"❌ Error initializing engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_file_paths():
    """Validate that required files exist."""
    print("🔍 Validating required files...")
    
    required_files = [
        "phase_4_llm_rag/Rag_config.json",
        "data/processed_phase_3/chunks.json",
        "data/db/legal_assistant.db",
        "phase_4_llm_rag/prompt_templates.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all validation tests."""
    print("🧪 Validating modernized LegalRAGEngine...")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: File paths
    if not validate_file_paths():
        all_passed = False
    
    print()
    
    # Test 2: Config compatibility
    if not validate_config_compatibility():
        all_passed = False
    
    print()
    
    # Test 3: Engine initialization
    if not validate_engine_initialization():
        all_passed = False
    
    print()
    print("=" * 50)
    
    if all_passed:
        print("🎉 All validation tests passed!")
        print("✅ The modernized LegalRAGEngine is working correctly with the new config schema.")
        return 0
    else:
        print("❌ Some validation tests failed.")
        print("🔧 Please check the errors above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
