#!/usr/bin/env python3
"""
Script to run API tests for the Smart Legal Assistant.
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the API tests using pytest."""
    print("🚀 Running Smart Legal Assistant API Tests")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("webapp/tests/test_api_qa.py").exists():
        print("❌ Error: webapp/tests/test_api_qa.py not found!")
        print("   Please run this script from the project root directory.")
        return False

    # Run the tests
    try:
        cmd = [
            sys.executable, "-m", "pytest",
            "webapp/tests/test_api_qa.py",
            "-v",
            "--tb=short",
            "--color=yes"
        ]

        print(f"📋 Running command: {' '.join(cmd)}")
        print("-" * 50)

        result = subprocess.run(cmd, cwd=Path.cwd())

        if result.returncode == 0:
            print("\n✅ All API tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with exit code: {result.returncode}")
            return False

    except FileNotFoundError:
        print("❌ Error: pytest not found!")
        print("   Please install testing dependencies:")
        print("   pip install pytest pytest-asyncio pytest-mock")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_specific_test(test_name):
    """Run a specific test."""
    print(f"🎯 Running specific test: {test_name}")
    print("-" * 30)

    try:
        cmd = [
            sys.executable, "-m", "pytest",
            f"webapp/tests/test_api_qa.py::{test_name}",
            "-v",
            "--tb=short",
            "--color=yes"
        ]

        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode == 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        success = run_tests()

    sys.exit(0 if success else 1)
