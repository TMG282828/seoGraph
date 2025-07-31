#!/usr/bin/env python3
"""
Simple test runner to bypass pytest configuration issues.
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import and run test functions
from tests.test_agents.test_content_generation.test_structure import (
    test_test_directory_structure,
    test_project_structure,
    test_module_structure_without_import,
    test_python_path_setup
)

def run_tests():
    """Run all structure tests."""
    tests = [
        ("Directory Structure", test_test_directory_structure),
        ("Project Structure", test_project_structure), 
        ("Module Structure", test_module_structure_without_import),
        ("Python Path Setup", test_python_path_setup)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running {test_name}...")
            test_func()
            print(f"✓ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)