#!/usr/bin/env python3
"""
Test script for ask_cli.py

This script tests the CLI functionality step by step to identify any issues.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_command(cmd: list, description: str, expect_success: bool = True) -> tuple:
    """Run a command and return result."""
    print(f"\n{'='*60}")
    print(f"🔍 Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=str(project_root),
            timeout=60  # 1 minute timeout
        )
        
        success = (result.returncode == 0) if expect_success else (result.returncode != 0)
        
        if success:
            print(f"✅ {description} - SUCCESS")
        else:
            print(f"❌ {description} - FAILED")
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.stdout:
            print(f"\nSTDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr}")
        
        return success, result
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (>60s)")
        return False, None
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False, None


def test_import():
    """Test if the CLI script can be imported without errors."""
    print("\n🔧 Testing imports...")
    try:
        # Test basic import
        from phase_4_llm_rag import ask_cli
        print("✅ Basic import successful")
        
        # Test argument parsing
        import argparse
        parser = ask_cli.parse_args.__wrapped__ if hasattr(ask_cli.parse_args, '__wrapped__') else None
        if parser is None:
            # Call parse_args with test args
            test_args = ['--question', 'test']
            original_argv = sys.argv[:]
            sys.argv = ['test'] + test_args
            try:
                args = ask_cli.parse_args()
                print("✅ Argument parsing works")
            finally:
                sys.argv = original_argv
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_basic():
    """Test basic CLI functionality."""
    tests = [
        {
            'name': 'Help Display',
            'cmd': [sys.executable, 'phase_4_llm_rag/ask_cli.py', '--help'],
            'expect_success': True,
            'description': 'Show help message'
        },
        {
            'name': 'Missing Question Error',
            'cmd': [sys.executable, 'phase_4_llm_rag/ask_cli.py'],
            'expect_success': False,
            'description': 'Error when no question provided'
        },
        {
            'name': 'Invalid Template Error', 
            'cmd': [sys.executable, 'phase_4_llm_rag/ask_cli.py', '--question', 'test', '--template', 'invalid'],
            'expect_success': False,
            'description': 'Error with invalid template'
        }
    ]
    
    results = []
    for test in tests:
        success, result = run_command(
            test['cmd'], 
            test['description'], 
            test['expect_success']
        )
        results.append(success)
    
    return all(results)


def test_cli_with_mocks():
    """Test CLI with mocked RAG engine to avoid needing real artifacts."""
    
    # Create a test script that patches the RAG engine
    test_script = """
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import patch, MagicMock
from phase_4_llm_rag.ask_cli import main
import sys

# Mock the RAG engine
mock_result = {
    "answer": "این یک پاسخ آزمایشی است که نشان می‌دهد سیستم RAG به درستی کار می‌کند.",
    "citations": [
        {
            "document_title": "قانون آزمایشی",
            "document_uid": "test_law_001", 
            "article_number": "۱",
            "note_label": "تبصره"
        }
    ],
    "retrieved_chunks": 3
}

with patch('phase_4_llm_rag.ask_cli.LegalRAGEngine') as mock_engine_class:
    mock_engine = MagicMock()
    mock_engine.answer.return_value = mock_result
    mock_engine.config = {"retriever": {"top_k": 5}}
    mock_engine_class.return_value = mock_engine
    
    # Mock the validation function to avoid file checks
    with patch('phase_4_llm_rag.ask_cli.validate_config_and_artifacts'):
        # Set up test arguments
        sys.argv = [
            'ask_cli.py',
            '--question', 'شرایط مرخصی اعضای هیئت علمی چیست؟',
            '--show-sources'
        ]
        
        exit_code = main()
        print(f"Exit code: {exit_code}")
"""
    
    # Write test script to a temporary file
    test_file = project_root / 'temp_cli_test.py'
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        # Run the test
        success, result = run_command(
            [sys.executable, str(test_file)],
            "Mocked CLI execution with sources",
            expect_success=True
        )
        
        return success
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_filter_parsing():
    """Test the filter parsing function."""
    print("\n🔧 Testing filter parsing...")
    try:
        from phase_4_llm_rag.ask_cli import parse_filters
        
        # Test cases
        test_cases = [
            {
                'input': ['document_type=law', 'section=بخش ۱'],
                'expected': {'document_type': 'law', 'section': 'بخش ۱'}
            },
            {
                'input': ['invalid_filter', 'key=value'],
                'expected': {'key': 'value'}
            },
            {
                'input': ['key="quoted value"'],
                'expected': {'key': 'quoted value'}
            }
        ]
        
        all_passed = True
        for i, test in enumerate(test_cases, 1):
            result = parse_filters(test['input'])
            if result == test['expected']:
                print(f"✅ Filter test {i} passed: {test['input']} -> {result}")
            else:
                print(f"❌ Filter test {i} failed: expected {test['expected']}, got {result}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Filter parsing test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 TESTING ask_cli.py")
    print("="*70)
    
    tests = [
        ("Import Test", test_import),
        ("Basic CLI Test", test_cli_basic), 
        ("Filter Parsing Test", test_filter_parsing),
        ("Mocked CLI Test", test_cli_with_mocks)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'🔬'*3} {test_name} {'🔬'*3}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"📊 TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("🎉 All tests passed! ask_cli.py is ready to use.")
        print("\n📖 Usage examples:")
        print("python phase_4_llm_rag/ask_cli.py --question 'سوال شما' --show-sources")
        print("python phase_4_llm_rag/ask_cli.py --question 'سوال شما' --template compare --json")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())