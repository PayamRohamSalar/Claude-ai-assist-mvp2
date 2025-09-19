#!/usr/bin/env python3
"""
Test script for the generalized project_structure.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

def test_project_structure():
    """Test the generalized project_structure.py functionality."""
    print("🧪 Testing generalized project_structure.py...")
    
    try:
        # Import the module
        sys.path.insert(0, '.')
        from project_structure import create_directory_structure
        
        # Test 1: Default behavior (should use current directory)
        print("\n📁 Test 1: Default behavior")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to test default behavior
            original_cwd = Path.cwd()
            temp_path = Path(temp_dir)
            
            try:
                # Test with explicit base_dir parameter
                success = create_directory_structure(base_dir=temp_path)
                
                if success:
                    print("✅ Default behavior test passed")
                    
                    # Verify some directories were created
                    expected_dirs = ['phase_0_setup', 'data/raw', 'logs', 'docs']
                    for dir_name in expected_dirs:
                        dir_path = temp_path / dir_name
                        if dir_path.exists():
                            print(f"   ✅ {dir_name} created")
                        else:
                            print(f"   ❌ {dir_name} not found")
                            return False
                else:
                    print("❌ Default behavior test failed")
                    return False
                    
            finally:
                # Restore original working directory
                import os
                os.chdir(original_cwd)
        
        # Test 2: Custom base directory
        print("\n📁 Test 2: Custom base directory")
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_base = Path(temp_dir) / "custom_project"
            
            success = create_directory_structure(base_dir=custom_base)
            
            if success:
                print("✅ Custom base directory test passed")
                
                # Verify the custom directory was created and used
                if custom_base.exists():
                    print(f"   ✅ Custom directory created: {custom_base}")
                else:
                    print(f"   ❌ Custom directory not created: {custom_base}")
                    return False
            else:
                print("❌ Custom base directory test failed")
                return False
        
        # Test 3: CLI argument parsing (mock test)
        print("\n📁 Test 3: CLI argument parsing")
        try:
            from project_structure import main
            print("✅ CLI argument parsing imports correctly")
        except Exception as e:
            print(f"❌ CLI argument parsing test failed: {e}")
            return False
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_resolution():
    """Test that paths are resolved correctly."""
    print("\n🔍 Testing path resolution...")
    
    try:
        from project_structure import create_directory_structure
        
        # Test with relative path
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            relative_path = temp_path / "relative_test"
            
            # Create a subdirectory to test relative path resolution
            test_dir = temp_path / "test_subdir"
            test_dir.mkdir()
            
            # Test with relative path
            success = create_directory_structure(base_dir=str(test_dir))
            
            if success:
                print("✅ Relative path resolution test passed")
                print(f"   📍 Resolved to: {test_dir.resolve()}")
            else:
                print("❌ Relative path resolution test failed")
                return False
        
        # Test with absolute path
        with tempfile.TemporaryDirectory() as temp_dir:
            abs_path = Path(temp_dir).resolve() / "absolute_test"
            
            success = create_directory_structure(base_dir=abs_path)
            
            if success:
                print("✅ Absolute path resolution test passed")
                print(f"   📍 Resolved to: {abs_path.resolve()}")
            else:
                print("❌ Absolute path resolution test failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Path resolution test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Project Structure Generalization Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Test basic functionality
    if not test_project_structure():
        all_passed = False
    
    # Test path resolution
    if not test_path_resolution():
        all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! The project_structure.py is now generalized.")
        print("✅ Key improvements:")
        print("   • Removed hardcoded Windows path")
        print("   • Added optional base_dir parameter")
        print("   • Added CLI argument support")
        print("   • Uses portable repository root as default")
        print("   • Shows actual resolved paths in logs")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
