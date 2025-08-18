# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_phase0_complete.py

"""
Legal Assistant AI - Phase 0 Complete Integration Test
Final comprehensive test for Phase 0 completion
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def test_phase0_completion():
    """Complete integration test for Phase 0"""
    
    print("=" * 80)
    print("🧪 LEGAL ASSISTANT AI - PHASE 0 COMPLETION TEST")
    print("🧪 تست تکمیل فاز صفر - دستیار حقوقی هوشمند")
    print("=" * 80)
    
    results = {
        'project_structure': False,
        'shared_utils': False,
        'configuration': False,
        'setup_scripts': False,
        'documentation': False,
        'integration': False
    }
    
    # Test 1: Project Structure
    print("\n📁 Testing Project Structure...")
    print("-" * 40)
    
    required_files = [
        "requirements.txt",
        ".env.template", 
        ".gitignore",
        "README.md",
        "config/config.json",
        "shared_utils/__init__.py",
        "shared_utils/constants.py",
        "shared_utils/logger.py", 
        "shared_utils/config_manager.py",
        "shared_utils/file_utils.py",
        "phase_0_setup/environment_setup.py",
        "phase_0_setup/validate_setup.py"
    ]
    
    required_dirs = [
        "data/raw",
        "data/processed_phase_1",
        "data/vector_db",
        "logs",
        "config",
        "shared_utils",
        "phase_0_setup",
        "tests"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/")
            missing_dirs.append(dir_path)
    
    if not missing_files and not missing_dirs:
        results['project_structure'] = True
        print("  🎉 Project structure complete!")
    else:
        print(f"  ❌ Missing {len(missing_files)} files, {len(missing_dirs)} directories")
    
    # Test 2: Shared Utils Integration
    print("\n🔧 Testing Shared Utils Integration...")
    print("-" * 40)
    
    try:
        from shared_utils import (
            get_logger, get_config, read_document,
            PROJECT_NAME, Messages, DocumentType
        )
        print("  ✅ Package imports successful")
        
        # Test logger
        logger = get_logger("Phase0Test")
        logger.info("Phase 0 completion test", "تست تکمیل فاز صفر")
        print("  ✅ Logger functionality")
        
        # Test config
        config = get_config()
        print(f"  ✅ Config loaded: {config.project_name}")
        
        # Test constants
        print(f"  ✅ Constants: {len(list(DocumentType))} document types")
        
        # Test file utils
        file_info_test = Path(__file__)
        from shared_utils.file_utils import get_file_info
        file_info = get_file_info(file_info_test)
        print(f"  ✅ File utils: {file_info.name}")
        
        results['shared_utils'] = True
        print("  🎉 Shared utils integration complete!")
        
    except Exception as e:
        print(f"  ❌ Shared utils error: {str(e)}")
    
    # Test 3: Configuration System
    print("\n⚙️ Testing Configuration System...")
    print("-" * 40)
    
    try:
        from shared_utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Test critical config values
        critical_configs = [
            ("project_name", config.project_name),
            ("database.url", config.database.url),
            ("llm.ollama_model", config.llm.ollama_model),
            ("rag.chunk_size", config.rag.chunk_size),
            ("web.port", config.web.port)
        ]
        
        config_ok = True
        for key, value in critical_configs:
            if value:
                print(f"  ✅ {key}: {value}")
            else:
                print(f"  ❌ {key}: not set")
                config_ok = False
        
        # Test dot notation
        test_value = config_manager.get_value("project_name")
        if test_value:
            print(f"  ✅ Dot notation access: {test_value}")
        else:
            print(f"  ❌ Dot notation access failed")
            config_ok = False
        
        results['configuration'] = config_ok
        if config_ok:
            print("  🎉 Configuration system complete!")
        
    except Exception as e:
        print(f"  ❌ Configuration error: {str(e)}")
    
    # Test 4: Setup Scripts
    print("\n🛠️ Testing Setup Scripts...")
    print("-" * 40)
    
    try:
        # Test environment setup import
        from phase_0_setup.environment_setup import EnvironmentSetup
        setup = EnvironmentSetup()
        print("  ✅ Environment setup script importable")
        
        # Test validate setup import
        from phase_0_setup.validate_setup import SetupValidator
        validator = SetupValidator()
        print("  ✅ Validation script importable")
        
        # Test setup functionality (non-destructive)
        python_check = setup._check_python_version()
        config_check = setup._setup_configuration()
        
        if python_check and config_check:
            print("  ✅ Setup scripts functional")
            results['setup_scripts'] = True
            print("  🎉 Setup scripts complete!")
        else:
            print("  ⚠️ Setup scripts have issues")
        
    except Exception as e:
        print(f"  ❌ Setup scripts error: {str(e)}")
    
    # Test 5: Documentation
    print("\n📄 Testing Documentation...")
    print("-" * 40)
    
    try:
        readme_file = Path("README.md")
        if readme_file.exists():
            readme_content = readme_file.read_text(encoding='utf-8')
            
            # Check key sections
            required_sections = [
                "# 🤖 دستیار حقوقی هوشمند",
                "## 🚀 راه‌اندازی سریع", 
                "## 📊 وضعیت فازهای توسعه",
                "### ✅ فاز صفر: آماده‌سازی محیط (تکمیل شده)"
            ]
            
            sections_found = 0
            for section in required_sections:
                if section in readme_content:
                    sections_found += 1
                    print(f"  ✅ Section found: {section.split()[1] if len(section.split()) > 1 else 'Header'}")
                else:
                    print(f"  ❌ Section missing: {section}")
            
            if sections_found == len(required_sections):
                results['documentation'] = True
                print("  🎉 Documentation complete!")
            else:
                print(f"  ⚠️ Documentation incomplete: {sections_found}/{len(required_sections)}")
        else:
            print("  ❌ README.md not found")
    
    except Exception as e:
        print(f"  ❌ Documentation error: {str(e)}")
    
    # Test 6: Integration Test
    print("\n🔗 Testing Complete Integration...")
    print("-" * 40)
    
    try:
        # Simulate complete workflow
        from shared_utils import get_config, get_logger
        
        # Test 1: Get configuration
        config = get_config()
        print(f"  ✅ Config: {config.project_name} v{config.version}")
        
        # Test 2: Setup logging
        logger = get_logger("IntegrationTest")
        logger.info("Integration test started", "تست یکپارچگی شروع شد")
        print("  ✅ Logging: Messages logged")
        
        # Test 3: File operations
        from shared_utils.file_utils import get_file_manager
        file_manager = get_file_manager()
        
        # Test directory access
        data_dir_info = file_manager.get_directory_info("data")
        if data_dir_info.get('exists'):
            print(f"  ✅ File ops: {data_dir_info['file_count']} files found")
        else:
            print("  ❌ File ops: data directory issue")
        
        # Test 4: Ready for Phase 1 check
        raw_data_dir = Path("data/raw")
        processed_dir = Path("data/processed_phase_1")
        
        if raw_data_dir.exists() and processed_dir.exists():
            print("  ✅ Phase 1 dirs: Ready for data processing")
            results['integration'] = True
            print("  🎉 Integration complete!")
        else:
            print("  ❌ Phase 1 dirs: Not ready")
    
    except Exception as e:
        print(f"  ❌ Integration error: {str(e)}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📊 PHASE 0 COMPLETION SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"📈 Tests Passed: {passed_tests}/{total_tests}")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    if passed_tests == total_tests:
        print("\n🎉 PHASE 0 COMPLETED SUCCESSFULLY!")
        print("🎉 فاز صفر با موفقیت تکمیل شد!")
        print("\n🚀 Ready for Phase 1: Data Processing")
        print("🚀 آماده برای فاز یک: پردازش داده‌ها")
        
        # Generate completion report
        completion_report = {
            "phase": "Phase 0",
            "status": "completed", 
            "completion_date": datetime.now().isoformat(),
            "test_results": results,
            "success_rate": f"{passed_tests}/{total_tests}",
            "next_phase": "Phase 1: Data Processing",
            "next_steps": [
                "Place raw legal documents in data/raw/",
                "Start Phase 1 with document_parser.py",
                "Review Phase 1 requirements"
            ]
        }
        
        # Save completion report
        report_file = Path("logs/phase0_completion_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(completion_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 Completion report saved: {report_file}")
        return True
        
    else:
        print(f"\n❌ PHASE 0 INCOMPLETE")
        print(f"❌ فاز صفر ناتمام")
        print(f"\n🔧 Fix {total_tests - passed_tests} failing tests before proceeding")
        return False

if __name__ == "__main__":
    success = test_phase0_completion()
    sys.exit(0 if success else 1)