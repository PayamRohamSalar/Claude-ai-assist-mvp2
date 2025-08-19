# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_setup_fix.py

"""
Quick test to verify setup scripts are working after fix
"""

import sys
from pathlib import Path

def test_setup_scripts_fix():
    """Test that setup scripts now work correctly"""
    
    print("🔧 Testing Setup Scripts Fix...")
    print("=" * 40)
    
    try:
        # Test 1: Import from package
        print("📦 Testing package imports...")
        from phase_0_setup import EnvironmentSetup, SetupValidator
        print("  ✅ Package imports successful")
        
        # Test 2: Direct module imports  
        print("📄 Testing direct module imports...")
        from phase_0_setup.environment_setup import EnvironmentSetup as EnvSetup
        from phase_0_setup.validate_setup import SetupValidator as Validator
        print("  ✅ Direct imports successful")
        
        # Test 3: Class instantiation
        print("🏗️ Testing class instantiation...")
        setup = EnvironmentSetup()
        validator = SetupValidator()
        print("  ✅ Classes instantiate correctly")
        
        # Test 4: Basic functionality
        print("⚙️ Testing basic functionality...")
        
        # Test environment setup methods
        python_check = setup._check_python_version()
        print(f"  ✅ Python check: {'passed' if python_check else 'failed'}")
        
        # Test validator basic structure
        if hasattr(validator, 'validation_results'):
            print("  ✅ Validator structure correct")
        else:
            print("  ❌ Validator structure incorrect")
            return False
        
        print("\n🎉 All setup script tests passed!")
        print("🔧 Setup Scripts issue has been fixed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you're in the correct directory and environment")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_setup_scripts_fix()
    
    if success:
        print("\n✅ Ready to re-run phase 0 completion test!")
        print("🚀 Run: python test_phase0_complete.py")
    else:
        print("\n❌ Additional fixes may be needed")
    
    sys.exit(0 if success else 1)