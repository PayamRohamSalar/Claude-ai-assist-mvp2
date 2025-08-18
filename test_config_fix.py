# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_config_fix.py

"""
Quick test to verify configuration and Python version fixes
"""

import sys
from pathlib import Path

def test_configuration_fix():
    """Test that configuration loads without errors"""
    
    print("🔧 Testing Configuration Fix...")
    print("=" * 40)
    
    try:
        # Test 1: Config loading
        print("⚙️ Testing config.json loading...")
        from shared_utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"  ✅ Config loaded: {config.project_name}")
        print(f"  ✅ Database URL: {config.database.url}")
        print(f"  ✅ LLM Model: {config.llm.ollama_model}")
        
        # Test 2: No error messages in logs
        print("\n📝 Testing for error messages...")
        # If we got here without exceptions, the config loading worked
        print("  ✅ No config loading errors")
        
        # Test 3: Dot notation access
        print("\n🔗 Testing dot notation access...")
        project_name = config_manager.get_value("project_name")
        db_url = config_manager.get_value("database.url")
        
        if project_name and db_url:
            print("  ✅ Dot notation working")
        else:
            print("  ❌ Dot notation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_python_version_fix():
    """Test that Python version check now passes"""
    
    print("\n🐍 Testing Python Version Fix...")
    print("=" * 40)
    
    try:
        from phase_0_setup.environment_setup import EnvironmentSetup
        
        setup = EnvironmentSetup()
        
        print(f"📋 Required Python version: {setup.required_python_version}")
        print(f"📋 Current Python version: {sys.version_info[:2]}")
        
        python_check = setup._check_python_version()
        
        if python_check:
            print("  ✅ Python version check passed")
            return True
        else:
            print("  ❌ Python version check still failing")
            return False
            
    except Exception as e:
        print(f"❌ Python version test failed: {e}")
        return False

def test_setup_scripts_functionality():
    """Test that setup scripts functionality works"""
    
    print("\n🛠️ Testing Setup Scripts Functionality...")
    print("=" * 40)
    
    try:
        from phase_0_setup.environment_setup import EnvironmentSetup
        
        setup = EnvironmentSetup()
        
        # Test individual methods
        print("🔍 Testing individual setup methods...")
        
        python_ok = setup._check_python_version()
        print(f"  Python check: {'✅' if python_ok else '❌'}")
        
        config_ok = setup._setup_configuration() 
        print(f"  Config setup: {'✅' if config_ok else '❌'}")
        
        dirs_ok = setup._setup_directories()
        print(f"  Directory setup: {'✅' if dirs_ok else '❌'}")
        
        validation_ok = setup._final_validation()
        print(f"  Final validation: {'✅' if validation_ok else '❌'}")
        
        # All should pass now
        if python_ok and config_ok and dirs_ok and validation_ok:
            print("\n🎉 All setup script methods working!")
            return True
        else:
            print(f"\n⚠️ Some methods still have issues")
            return False
            
    except Exception as e:
        print(f"❌ Setup scripts functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Configuration and Python Version Fixes")
    print("=" * 60)
    
    success1 = test_configuration_fix()
    success2 = test_python_version_fix() 
    success3 = test_setup_scripts_functionality()
    
    overall_success = success1 and success2 and success3
    
    print("\n" + "=" * 60)
    print("📊 Fix Test Results:")
    print("=" * 60)
    
    print(f"{'✅' if success1 else '❌'} Configuration Fix")
    print(f"{'✅' if success2 else '❌'} Python Version Fix")
    print(f"{'✅' if success3 else '❌'} Setup Scripts Functionality")
    
    if overall_success:
        print("\n🎉 All fixes successful!")
        print("🚀 Ready to run full Phase 0 test again")
        print("📝 Run: python test_phase0_complete.py")
    else:
        print("\n❌ Some fixes still need work")
    
    sys.exit(0 if overall_success else 1)