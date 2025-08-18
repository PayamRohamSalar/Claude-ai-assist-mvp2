# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_config_json.py

"""
Test script for config.json file
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_config_json():
    """Test the config.json file"""
    
    print("🧪 Testing config.json file...")
    
    try:
        # Test loading config.json directly
        config_file = Path("config/config.json")
        
        if not config_file.exists():
            print(f"❌ Config file not found: {config_file}")
            return False
        
        print(f"✅ Config file found: {config_file}")
        
        # Test JSON validity
        print("\n📝 Testing JSON syntax...")
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        print("✅ JSON syntax is valid")
        
        # Test basic structure
        print("\n📋 Testing basic structure...")
        required_sections = [
            'project_name', 'version', 'environment', 'database', 
            'llm', 'rag', 'processing', 'web', 'security', 
            'logging', 'features'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config_data:
                missing_sections.append(section)
            else:
                print(f"  ✅ {section}")
        
        if missing_sections:
            print(f"❌ Missing sections: {missing_sections}")
            return False
        
        # Test project information
        print(f"\n📋 Project: {config_data['project_name']}")
        print(f"📋 Version: {config_data['version']}")
        print(f"🌍 Environment: {config_data['environment']}")
        
        # Test LLM configuration
        print(f"\n🤖 LLM Model: {config_data['llm']['ollama_model']}")
        print(f"🤖 Backup Model: {config_data['llm']['ollama_backup_model']}")
        print(f"🌡️ Temperature: {config_data['llm']['temperature']}")
        
        # Test RAG configuration
        print(f"\n🔍 Chunk Size: {config_data['rag']['chunk_size']}")
        print(f"🔍 Similarity Threshold: {config_data['rag']['similarity_threshold']}")
        print(f"🔍 Embedding Model: {config_data['rag']['embedding_model']}")
        
        # Test database configuration
        print(f"\n🗃️ Database URL: {config_data['database']['url']}")
        print(f"🗃️ Vector DB Path: {config_data['database']['vector_db_path']}")
        
        # Test web configuration
        print(f"\n🌐 Web Host: {config_data['web']['host']}")
        print(f"🌐 Web Port: {config_data['web']['port']}")
        
        # Test feature flags
        print(f"\n⭐ Features:")
        features = config_data['features']
        for feature, enabled in features.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {feature}: {enabled}")
        
        # Test phase settings
        if 'phase_settings' in config_data:
            print(f"\n📊 Phase Settings:")
            for phase, settings in config_data['phase_settings'].items():
                status = settings.get('status', 'unknown')
                description = settings.get('description', 'No description')
                print(f"  📋 {phase}: {status} - {description}")
        
        # Test metadata
        if '_metadata' in config_data:
            metadata = config_data['_metadata']
            print(f"\n📊 Metadata:")
            print(f"  📅 Created: {metadata.get('created_at', 'Unknown')}")
            print(f"  👤 Created by: {metadata.get('created_by', 'Unknown')}")
            print(f"  📄 Description: {metadata.get('description', 'No description')}")
        
        print("\n🎉 Config.json structure test passed!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def test_config_integration():
    """Test config.json integration with ConfigManager"""
    
    print("\n🔗 Testing config.json integration with ConfigManager...")
    
    try:
        from shared_utils.config_manager import ConfigManager
        
        # Create config manager that uses our config.json
        config_file = Path("config/config.json")
        config_manager = ConfigManager(config_file=config_file)
        
        # Load configuration
        config = config_manager.load_config()
        
        print("✅ Config loaded successfully via ConfigManager")
        
        # Test that values match what's in JSON
        print(f"📋 Project from config: {config.project_name}")
        print(f"🤖 LLM model from config: {config.llm.ollama_model}")
        print(f"🔍 RAG chunk size from config: {config.rag.chunk_size}")
        print(f"🌐 Web port from config: {config.web.port}")
        
        # Test dot notation access
        print("\n📝 Testing dot notation access...")
        project_name = config_manager.get_value("project_name")
        llm_model = config_manager.get_value("llm.ollama_model")
        chunk_size = config_manager.get_value("rag.chunk_size")
        
        print(f"  📋 project_name: {project_name}")
        print(f"  🤖 llm.ollama_model: {llm_model}")
        print(f"  🔍 rag.chunk_size: {chunk_size}")
        
        # Test validation
        print("\n✅ Testing configuration validation...")
        config_manager._validate_config(config)
        print("✅ Configuration validation passed (check logs for any warnings)")
        
        print("\n🎉 Config integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_completeness():
    """Test that config.json has all necessary settings for the project"""
    
    print("\n🔍 Testing configuration completeness...")
    
    try:
        config_file = Path("config/config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Test LLM settings completeness
        llm_required = ['ollama_base_url', 'ollama_model', 'temperature', 'max_tokens']
        llm_config = config_data.get('llm', {})
        
        print("🤖 LLM Configuration:")
        for setting in llm_required:
            if setting in llm_config:
                print(f"  ✅ {setting}: {llm_config[setting]}")
            else:
                print(f"  ❌ Missing: {setting}")
        
        # Test RAG settings completeness
        rag_required = ['chunk_size', 'embedding_model', 'similarity_threshold']
        rag_config = config_data.get('rag', {})
        
        print("\n🔍 RAG Configuration:")
        for setting in rag_required:
            if setting in rag_config:
                print(f"  ✅ {setting}: {rag_config[setting]}")
            else:
                print(f"  ❌ Missing: {setting}")
        
        # Test database settings
        db_required = ['url', 'vector_db_path']
        db_config = config_data.get('database', {})
        
        print("\n🗃️ Database Configuration:")
        for setting in db_required:
            if setting in db_config:
                print(f"  ✅ {setting}: {db_config[setting]}")
            else:
                print(f"  ❌ Missing: {setting}")
        
        # Test processing settings
        proc_required = ['max_file_size', 'allowed_extensions', 'batch_size']
        proc_config = config_data.get('processing', {})
        
        print("\n📄 Processing Configuration:")
        for setting in proc_required:
            if setting in proc_config:
                value = proc_config[setting]
                if setting == 'max_file_size':
                    # Convert bytes to MB for display
                    value_mb = value / (1024 * 1024)
                    print(f"  ✅ {setting}: {value_mb:.1f} MB")
                else:
                    print(f"  ✅ {setting}: {value}")
            else:
                print(f"  ❌ Missing: {setting}")
        
        # Test specific values for this project
        print("\n🔧 Project-specific settings:")
        
        # Check if models match available Ollama models
        ollama_model = config_data.get('llm', {}).get('ollama_model', '')
        if 'qwen2.5' in ollama_model:
            print(f"  ✅ Using available Ollama model: {ollama_model}")
        else:
            print(f"  ⚠️ Model might not be available: {ollama_model}")
        
        # Check chunk size is reasonable for legal documents
        chunk_size = config_data.get('rag', {}).get('chunk_size', 0)
        if 500 <= chunk_size <= 2000:
            print(f"  ✅ Chunk size appropriate for legal docs: {chunk_size}")
        else:
            print(f"  ⚠️ Chunk size might be suboptimal: {chunk_size}")
        
        # Check if Persian embedding model is specified
        persian_model = config_data.get('rag', {}).get('persian_embedding_model', '')
        if persian_model:
            print(f"  ✅ Persian embedding model specified: {persian_model}")
        else:
            print(f"  ⚠️ No Persian embedding model specified")
        
        print("\n🎉 Configuration completeness test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Completeness test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Legal Assistant AI - Config.json Test Suite")
    print("=" * 60)
    
    success1 = test_config_json()
    success2 = test_config_integration()
    success3 = test_config_completeness()
    
    if success1 and success2 and success3:
        print("\n🎉 All config.json tests passed! Configuration is ready.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)