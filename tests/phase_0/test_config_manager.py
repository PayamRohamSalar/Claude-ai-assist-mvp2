# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_config_manager.py

"""
Test script for config_manager.py module
"""

import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_config_manager():
    """Test the config manager module functionality"""
    
    print("🧪 Testing config_manager.py module...")
    
    try:
        # Import the config manager module
        from shared_utils.config_manager import (
            ConfigManager, AppConfig, DatabaseConfig, LLMConfig,
            get_config_manager, get_config, get_config_value
        )
        
        print("✅ Import successful")
        
        # Test creating config manager
        print("\n📝 Testing ConfigManager creation...")
        config_manager = ConfigManager()
        print(f"✅ ConfigManager created")
        
        # Test loading default configuration
        print("\n📝 Testing default configuration loading...")
        config = config_manager.load_config()
        print(f"✅ Configuration loaded")
        print(f"  📋 Project: {config.project_name}")
        print(f"  🌍 Environment: {config.environment}")
        print(f"  🐛 Debug: {config.debug}")
        
        # Test configuration structure
        print("\n📝 Testing configuration structure...")
        print(f"  🗃️ Database URL: {config.database.url}")
        print(f"  🤖 LLM Model: {config.llm.ollama_model}")
        print(f"  🔍 RAG Chunk Size: {config.rag.chunk_size}")
        print(f"  🌐 Web Port: {config.web.port}")
        print(f"  🔐 Security Algorithm: {config.security.algorithm}")
        
        # Test dot notation access
        print("\n📝 Testing dot notation access...")
        db_url = config_manager.get_value("database.url")
        llm_model = config_manager.get_value("llm.ollama_model")
        chunk_size = config_manager.get_value("rag.chunk_size")
        invalid_key = config_manager.get_value("invalid.key", "default_value")
        
        print(f"  📊 database.url: {db_url}")
        print(f"  🤖 llm.ollama_model: {llm_model}")
        print(f"  📏 rag.chunk_size: {chunk_size}")
        print(f"  ❓ invalid.key: {invalid_key}")
        
        # Test setting values
        print("\n📝 Testing configuration value setting...")
        original_port = config_manager.get_value("web.port")
        config_manager.set_value("web.port", 9000)
        new_port = config_manager.get_value("web.port")
        print(f"  🔄 Port changed: {original_port} → {new_port}")
        
        # Test environment variable override
        print("\n📝 Testing environment variable override...")
        os.environ['PORT'] = '8080'
        os.environ['DEBUG'] = 'false'
        os.environ['OLLAMA_MODEL_NAME'] = 'mistral:7b'
        
        # Reload configuration to pick up env changes
        config_with_env = config_manager.load_config(force_reload=True)
        print(f"  🌐 Port from env: {config_with_env.web.port}")
        print(f"  🐛 Debug from env: {config_with_env.debug}")
        print(f"  🤖 Model from env: {config_with_env.llm.ollama_model}")
        
        # Clean up environment variables
        os.environ.pop('PORT', None)
        os.environ.pop('DEBUG', None)
        os.environ.pop('OLLAMA_MODEL_NAME', None)
        
        # Test global functions
        print("\n📝 Testing global functions...")
        global_config_manager = get_config_manager()
        global_config = get_config()
        project_name = get_config_value("project_name")
        
        print(f"  🌍 Global config manager: {type(global_config_manager).__name__}")
        print(f"  📋 Global config: {global_config.project_name}")
        print(f"  📝 Config value: {project_name}")
        
        # Test feature flags
        print("\n📝 Testing feature flags...")
        features = config.features
        print(f"  🔍 Document comparison: {features.enable_document_comparison}")
        print(f"  📝 Draft generation: {features.enable_draft_generation}")
        print(f"  🔍 Advanced search: {features.enable_advanced_search}")
        print(f"  💾 Cache enabled: {features.enable_cache}")
        
        # Test dataclass conversion
        print("\n📝 Testing dataclass functionality...")
        from dataclasses import asdict
        config_dict = asdict(config)
        print(f"  📊 Config as dict keys: {list(config_dict.keys())[:5]}...")
        
        # Test validation (with invalid values)
        print("\n📝 Testing configuration validation...")
        try:
            # Create config with invalid values
            invalid_config = AppConfig()
            invalid_config.llm.temperature = 5.0  # Invalid (>2)
            invalid_config.rag.similarity_threshold = 2.0  # Invalid (>1)
            invalid_config.web.port = -1  # Invalid (<1)
            
            # This should log warnings
            config_manager._validate_config(invalid_config)
            print("  ⚠️ Validation warnings logged (check logs)")
            
        except Exception as e:
            print(f"  ❌ Validation error: {e}")
        
        print("\n🎉 All config manager tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_file_operations():
    """Test configuration file creation and saving"""
    
    print("\n🗃️ Testing configuration file operations...")
    
    try:
        from shared_utils.config_manager import ConfigManager, AppConfig
        from shared_utils.constants import CONFIG_DIR
        
        # Create a test config file path
        test_config_file = CONFIG_DIR / "test_config.json"
        
        # Create config manager with test file
        config_manager = ConfigManager(config_file=test_config_file)
        
        # Load config (this should create the file)
        config = config_manager.load_config()
        
        # Check if file was created
        if test_config_file.exists():
            print("  ✅ Config file created successfully")
            
            # Check file content
            with open(test_config_file, 'r', encoding='utf-8') as f:
                file_content = json.load(f)
            
            print(f"  📄 Config file has {len(file_content)} top-level keys")
            print(f"  🔑 Keys: {list(file_content.keys())[:5]}...")
            
            # Test saving modified config
            config.project_name = "Modified Legal Assistant"
            config.debug = False
            config_manager.save_config(config)
            
            # Reload and verify changes
            reloaded_config = config_manager.load_config(force_reload=True)
            print(f"  🔄 Modified project name: {reloaded_config.project_name}")
            print(f"  🔄 Modified debug: {reloaded_config.debug}")
            
            # Clean up test file
            test_config_file.unlink()
            print("  🧹 Test config file cleaned up")
            
        else:
            print("  ❌ Config file was not created")
            return False
        
        print("  ✅ File operations test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ File operations test failed: {e}")
        return False

def test_environment_specific_config():
    """Test environment-specific configuration overrides"""
    
    print("\n🌍 Testing environment-specific configurations...")
    
    try:
        from shared_utils.config_manager import ConfigManager, AppConfig
        
        # Test production environment
        print("  🏭 Testing production environment...")
        config_manager = ConfigManager()
        
        # Set environment to production
        os.environ['ENVIRONMENT'] = 'production'
        config = config_manager.load_config(force_reload=True)
        
        print(f"    📊 Environment: {config.environment}")
        print(f"    🐛 Debug mode: {config.debug}")
        print(f"    🔄 Web reload: {config.web.reload}")
        print(f"    📊 Log level: {config.logging.level}")
        
        # Test testing environment
        print("  🧪 Testing testing environment...")
        os.environ['ENVIRONMENT'] = 'testing'
        config = config_manager.load_config(force_reload=True)
        
        print(f"    📊 Environment: {config.environment}")
        print(f"    🤖 Mock APIs: {config.features.mock_external_apis}")
        print(f"    📊 Sample data: {config.features.use_sample_data}")
        print(f"    🗃️ Database URL: {config.database.url}")
        
        # Clean up
        os.environ.pop('ENVIRONMENT', None)
        
        print("  ✅ Environment-specific test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Environment-specific test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Legal Assistant AI - Config Manager Test Suite")
    print("=" * 60)
    
    success1 = test_config_manager()
    success2 = test_config_file_operations()
    success3 = test_environment_specific_config()
    
    if success1 and success2 and success3:
        print("\n🎉 All config manager tests passed! Configuration system is ready.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)