#!/usr/bin/env python3
"""
Manual test for ask_cli.py - check basic functionality without running full CLI
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all imports work."""
    print("🔍 Testing imports...")
    
    try:
        # Test importing the CLI module
        import phase_4_llm_rag.ask_cli as cli
        print("✅ CLI module imported successfully")
        
        # Test individual functions
        filters = cli.parse_filters(["document_type=law", "section=test"])
        assert filters == {"document_type": "law", "section": "test"}
        print("✅ parse_filters works correctly")
        
        # Test format_sources with mock data
        mock_citations = [
            {
                "document_title": "قانون آزمایشی", 
                "article_number": "1",
                "note_label": "تبصره"
            }
        ]
        sources = cli.format_sources(mock_citations)
        assert "منابع:" in sources
        assert "قانون آزمایشی" in sources
        print("✅ format_sources works correctly")
        
        # Test argument parsing function exists
        assert hasattr(cli, 'parse_args')
        assert callable(cli.parse_args)
        print("✅ parse_args function exists")
        
        # Test main function exists
        assert hasattr(cli, 'main')
        assert callable(cli.main)
        print("✅ main function exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_help():
    """Test that help can be displayed."""
    print("\n🔍 Testing help display...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 
            'phase_4_llm_rag/ask_cli.py', 
            '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Help display works")
            if "متن پرسش به زبان فارسی" in result.stdout:
                print("✅ Persian help text present")
            return True
        else:
            print(f"❌ Help failed with code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Help test failed: {e}")
        return False

def test_validation():
    """Test argument validation."""
    print("\n🔍 Testing argument validation...")
    
    try:
        import subprocess
        
        # Test missing required argument
        result = subprocess.run([
            sys.executable,
            'phase_4_llm_rag/ask_cli.py'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("✅ Missing question argument properly rejected")
            return True
        else:
            print("❌ Should have failed with missing question")
            return False
            
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def main():
    """Run manual tests."""
    print("🧪 MANUAL CLI TEST")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Help Display", test_help),
        ("Argument Validation", test_validation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 CLI appears to be working correctly!")
        print("\n💡 To test with real data:")
        print("python phase_4_llm_rag/ask_cli.py --question 'سوال شما' --show-sources")
    else:
        print("❌ Some issues found. Check output above.")
    
    return failed

if __name__ == "__main__":
    sys.exit(main())