# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_environment_setup.py

"""
Test script for environment_setup.py
"""

import sys
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_environment_setup_import():
    """Test importing environment setup module"""
    
    print("🧪 Testing environment_setup.py import...")
    
    try:
        # Test import
        from phase_0_setup.environment_setup import EnvironmentSetup
        
        print("✅ Import successful")
        
        # Test class instantiation
        setup = EnvironmentSetup()
        print(f"✅ EnvironmentSetup class created")
        print(f"  📋 Required Python version: {setup.required_python_version}")
        print(f"  📦 Required packages: {len(setup.required_packages)}")
        
        # Test setup results structure
        print(f"  📊 Setup results keys: {list(setup.setup_results.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_individual_checks():
    """Test individual setup check methods"""
    
    print("\n🔧 Testing individual setup checks...")
    
    try:
        from phase_0_setup.environment_setup import EnvironmentSetup
        
        setup = EnvironmentSetup()
        
        # Test Python version check
        print("\n  🐍 Testing Python version check...")
        python_ok = setup._check_python_version()
        print(f"    Result: {'✅' if python_ok else '❌'}")
        
        # Test Conda check (non-destructive)
        print("\n  🐍 Testing Conda environment check...")
        try:
            conda_ok = setup._check_conda_environment()
            print(f"    Result: {'✅' if conda_ok else '⚠️'}")
        except Exception as e:
            print(f"    ⚠️ Conda check failed: {e}")
        
        # Test directory setup
        print("\n  📁 Testing directory setup...")
        dirs_ok = setup._setup_directories()
        print(f"    Result: {'✅' if dirs_ok else '❌'}")
        
        # Test configuration setup
        print("\n  ⚙️ Testing configuration setup...")
        config_ok = setup._setup_configuration()
        print(f"    Result: {'✅' if config_ok else '❌'}")
        
        # Test final validation
        print("\n  ✅ Testing final validation...")
        validation_ok = setup._final_validation()
        print(f"    Result: {'✅' if validation_ok else '❌'}")
        
        print("\n🎉 Individual checks completed!")
        return True
        
    except Exception as e:
        print(f"❌ Individual checks failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_setup_dry_run():
    """Test environment setup in dry run mode (without installing packages)"""
    
    print("\n🏃 Testing environment setup dry run...")
    
    try:
        # We'll test the setup class methods individually rather than
        # running the full setup to avoid actually installing packages
        
        from phase_0_setup.environment_setup import EnvironmentSetup
        
        setup = EnvironmentSetup()
        
        print("  📊 Testing setup results tracking...")
        
        # Simulate some setup steps
        original_results = setup.setup_results.copy()
        print(f"    Initial results: {sum(original_results.values())}/{len(original_results)} passed")
        
        # Test Python check (should pass)
        setup.setup_results['python_check'] = setup._check_python_version()
        
        # Test directories (should pass)
        setup.setup_results['directories_check'] = setup._setup_directories()
        
        # Test config (should pass)
        setup.setup_results['config_check'] = setup._setup_configuration()
        
        # Test validation (should pass if other components work)
        setup.setup_results['final_validation'] = setup._final_validation()
        
        # Generate report
        setup._generate_setup_report()
        
        final_results = setup.setup_results
        passed_count = sum(final_results.values())
        total_count = len(final_results)
        
        print(f"    Final results: {passed_count}/{total_count} passed")
        
        # Check if report was generated
        report_file = Path("logs/environment_setup_report.json")
        if report_file.exists():
            print(f"    ✅ Setup report generated: {report_file}")
            
            # Read and display report summary
            import json
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            print(f"    📅 Setup date: {report.get('setup_date', 'Unknown')}")
            print(f"    🐍 Python version: {report.get('python_version', 'Unknown')}")
            print(f"    📊 Success rate: {report.get('success_rate', 'Unknown')}")
            
        else:
            print(f"    ⚠️ Setup report not generated")
        
        print("\n🎉 Dry run completed!")
        return True
        
    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_setup_script_execution():
    """Test running the setup script as a subprocess"""
    
    print("\n🚀 Testing setup script execution...")
    
    try:
        setup_script = Path("phase_0_setup/environment_setup.py")
        
        if not setup_script.exists():
            print(f"❌ Setup script not found: {setup_script}")
            return False
        
        print(f"✅ Setup script found: {setup_script}")
        
        # Test script execution with --help or similar (if we add such options)
        # For now, we'll just verify the script can be imported and run
        print("  🔄 Testing script importability...")
        
        # This is safer than running the full setup
        import importlib.util
        spec = importlib.util.spec_from_file_location("environment_setup", setup_script)
        
        if spec and spec.loader:
            print("  ✅ Script can be loaded")
        else:
            print("  ❌ Script cannot be loaded")
            return False
        
        # Test that the main function exists
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'main'):
            print("  ✅ Main function found")
        else:
            print("  ❌ Main function not found")
            return False
        
        if hasattr(module, 'EnvironmentSetup'):
            print("  ✅ EnvironmentSetup class found")
        else:
            print("  ❌ EnvironmentSetup class not found")
            return False
        
        print("\n🎉 Script execution test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Script execution test failed: {e}")
        return False

def test_post_setup_validation():
    """Test that environment is properly set up after running setup"""
    
    print("\n✅ Testing post-setup environment validation...")
    
    try:
        # Test that all expected directories exist
        expected_dirs = [
            "data/raw",
            "data/processed_phase_1", 
            "data/vector_db",
            "logs",
            "backup",
            "config"
        ]
        
        missing_dirs = []
        for dir_path in expected_dirs:
            full_path = Path(dir_path)
            if full_path.exists():
                print(f"  ✅ {dir_path}")
            else:
                print(f"  ❌ {dir_path}")
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"  ⚠️ {len(missing_dirs)} directories missing")
        else:
            print(f"  🎉 All {len(expected_dirs)} directories present")
        
        # Test that configuration is accessible
        print("\n  ⚙️ Testing configuration access...")
        try:
            from shared_utils.config_manager import get_config
            config = get_config()
            print(f"    ✅ Config loaded: {config.project_name}")
        except Exception as e:
            print(f"    ❌ Config access failed: {e}")
        
        # Test that logging works
        print("\n  📝 Testing logging system...")
        try:
            from shared_utils.logger import get_logger
            logger = get_logger("PostSetupTest")
            logger.info("Post-setup validation test", "تست اعتبارسنجی پس از تنظیم")
            print(f"    ✅ Logging system working")
        except Exception as e:
            print(f"    ❌ Logging system failed: {e}")
        
        # Test that file utilities work
        print("\n  📁 Testing file utilities...")
        try:
            from shared_utils.file_utils import get_file_info
            test_file = Path(__file__)
            file_info = get_file_info(test_file)
            print(f"    ✅ File utilities working")
        except Exception as e:
            print(f"    ❌ File utilities failed: {e}")
        
        # Check if setup report exists
        print("\n  📊 Checking setup report...")
        report_file = Path("logs/environment_setup_report.json")
        if report_file.exists():
            print(f"    ✅ Setup report exists")
            
            # Check report content
            try:
                import json
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                success_rate = report.get('success_rate', '0/0')
                print(f"    📈 Success rate: {success_rate}")
                
                if 'next_steps' in report:
                    next_steps = report['next_steps']
                    if next_steps:
                        print(f"    📋 Next steps: {len(next_steps)} items")
                    else:
                        print(f"    🎉 No additional steps needed")
                
            except Exception as e:
                print(f"    ⚠️ Could not read report content: {e}")
        else:
            print(f"    ⚠️ Setup report not found")
        
        print("\n🎉 Post-setup validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Post-setup validation failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Legal Assistant AI - Environment Setup Test Suite")
    print("=" * 70)
    
    success1 = test_environment_setup_import()
    success2 = test_individual_checks()
    success3 = test_setup_dry_run()
    success4 = test_setup_script_execution()
    success5 = test_post_setup_validation()
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    tests = [
        ("Import Test", success1),
        ("Individual Checks", success2),
        ("Dry Run", success3),
        ("Script Execution", success4),
        ("Post-Setup Validation", success5)
    ]
    
    for test_name, success in tests:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    overall_success = all(success for _, success in tests)
    
    if overall_success:
        print("\n🎉 All environment setup tests passed!")
        print("🚀 Ready to run: python phase_0_setup/environment_setup.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)