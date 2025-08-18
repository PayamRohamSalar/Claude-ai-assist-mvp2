# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\phase_0_setup\environment_setup.py

"""
Legal Assistant AI - Environment Setup Script
Automatically configures the development environment for the Legal Assistant AI project
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import platform
from datetime import datetime

# Add parent directory to path to import shared utilities
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils.logger import get_logger, log_system_startup
from shared_utils.config_manager import get_config_manager
from shared_utils.constants import PROJECT_NAME, BASE_DIR
from shared_utils.file_utils import create_directory


class EnvironmentSetup:
    """
    Environment setup and validation for Legal Assistant AI
    """
    
    def __init__(self):
        self.logger = get_logger("EnvironmentSetup")
        self.setup_results = {
            'python_check': False,
            'conda_check': False,
            'dependencies_check': False,
            'directories_check': False,
            'config_check': False,
            'ollama_check': False,
            'final_validation': False
        }
        self.required_python_version = (3, 10)
        self.required_packages = [
            'pandas', 'numpy', 'fastapi', 'python-dotenv',
            'PyPDF2', 'python-docx', 'chardet'
        ]
        
    def run_full_setup(self) -> bool:
        """
        Run complete environment setup process
        
        Returns:
            bool: True if all setup steps successful
        """
        
        self.logger.info(
            "Starting environment setup for Legal Assistant AI",
            "شروع تنظیم محیط برای دستیار حقوقی هوشمند"
        )
        
        try:
            # Step 1: Check Python version
            self._print_step("🐍 بررسی نسخه Python")
            self.setup_results['python_check'] = self._check_python_version()
            
            # Step 2: Check Conda environment
            self._print_step("🐍 بررسی محیط Conda")
            self.setup_results['conda_check'] = self._check_conda_environment()
            
            # Step 3: Install/Check dependencies
            self._print_step("📦 بررسی و نصب dependencies")
            self.setup_results['dependencies_check'] = self._setup_dependencies()
            
            # Step 4: Create/Check directories
            self._print_step("📁 ایجاد و بررسی دایرکتوری‌ها")
            self.setup_results['directories_check'] = self._setup_directories()
            
            # Step 5: Setup configuration
            self._print_step("⚙️ تنظیم و بررسی configuration")
            self.setup_results['config_check'] = self._setup_configuration()
            
            # Step 6: Check Ollama setup
            self._print_step("🤖 بررسی تنظیمات Ollama")
            self.setup_results['ollama_check'] = self._check_ollama_setup()
            
            # Step 7: Final validation
            self._print_step("✅ اعتبارسنجی نهایی محیط")
            self.setup_results['final_validation'] = self._final_validation()
            
            # Generate setup report
            self._generate_setup_report()
            
            # Check overall success
            success = all(self.setup_results.values())
            
            if success:
                self._print_success_message()
            else:
                self._print_failure_message()
            
            return success
            
        except Exception as e:
            self.logger.error(
                f"Environment setup failed: {str(e)}",
                f"خطا در تنظیم محیط: {str(e)}"
            )
            return False
    
    def _check_python_version(self) -> bool:
        """Check if Python version meets requirements"""
        
        try:
            current_version = sys.version_info[:2]
            required_version = self.required_python_version
            
            self.logger.info(
                f"Python version: {sys.version}",
                f"نسخه Python: {current_version[0]}.{current_version[1]}"
            )
            
            if current_version >= required_version:
                print(f"  ✅ Python {current_version[0]}.{current_version[1]} (مناسب)")
                return True
            else:
                print(f"  ❌ Python {current_version[0]}.{current_version[1]} (نیاز به {required_version[0]}.{required_version[1]}+)")
                self.logger.warning(
                    f"Python version {current_version} is below required {required_version}",
                    f"نسخه Python {current_version} کمتر از نسخه مورد نیاز {required_version}"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking Python version: {str(e)}")
            return False
    
    def _check_conda_environment(self) -> bool:
        """Check Conda environment setup"""
        
        try:
            # Check if conda is available
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            
            if result.returncode != 0:
                print("  ❌ Conda not found")
                self.logger.warning("Conda not available")
                return False
            
            conda_version = result.stdout.strip()
            print(f"  ✅ {conda_version}")
            
            # Check current environment
            current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
            print(f"  📍 محیط فعلی: {current_env}")
            
            # Check if we're in the right environment
            if current_env in ['claude-ai', 'legal_assistant']:
                print(f"  ✅ محیط مناسب فعال است: {current_env}")
                return True
            else:
                print(f"  ⚠️ محیط claude-ai یا legal_assistant فعال نیست")
                print(f"  💡 برای فعال‌سازی: conda activate claude-ai")
                return True  # Not critical, just a warning
                
        except FileNotFoundError:
            print("  ❌ Conda not found in PATH")
            self.logger.warning("Conda not found in system PATH")
            return False
        except Exception as e:
            self.logger.error(f"Error checking Conda: {str(e)}")
            return False
    
    def _setup_dependencies(self) -> bool:
        """Install and verify required dependencies"""
        
        try:
            # Check pip availability
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                 capture_output=True, text=True)
            
            if result.returncode != 0:
                print("  ❌ pip not available")
                return False
            
            print(f"  ✅ pip available")
            
            # Check each required package
            missing_packages = []
            installed_packages = []
            
            for package in self.required_packages:
                try:
                    __import__(package.replace('-', '_'))
                    installed_packages.append(package)
                    print(f"  ✅ {package}")
                except ImportError:
                    missing_packages.append(package)
                    print(f"  ❌ {package} (missing)")
            
            # Install missing packages
            if missing_packages:
                print(f"\n  📦 نصب {len(missing_packages)} پکیج مفقود...")
                
                for package in missing_packages:
                    print(f"    🔄 نصب {package}...")
                    result = subprocess.run([
                        sys.executable, '-m', 'pip', 'install', package
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"    ✅ {package} نصب شد")
                    else:
                        print(f"    ❌ خطا در نصب {package}")
                        self.logger.error(f"Failed to install {package}: {result.stderr}")
            
            # Final verification
            all_installed = True
            for package in self.required_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    all_installed = False
                    break
            
            if all_installed:
                print(f"  🎉 تمام {len(self.required_packages)} پکیج آماده است")
                return True
            else:
                print(f"  ❌ برخی پکیج‌ها هنوز مفقود هستند")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up dependencies: {str(e)}")
            return False
    
    def _setup_directories(self) -> bool:
        """Create and verify required directories"""
        
        try:
            required_dirs = [
                "data/raw",
                "data/processed_phase_1", 
                "data/vector_db",
                "logs",
                "backup",
                "config"
            ]
            
            success_count = 0
            
            for dir_path in required_dirs:
                full_path = BASE_DIR / dir_path
                
                if create_directory(full_path):
                    print(f"  ✅ {dir_path}")
                    success_count += 1
                else:
                    print(f"  ❌ {dir_path}")
                    self.logger.error(f"Failed to create directory: {dir_path}")
            
            if success_count == len(required_dirs):
                print(f"  🎉 تمام {len(required_dirs)} دایرکتوری آماده است")
                return True
            else:
                print(f"  ⚠️ {success_count}/{len(required_dirs)} دایرکتوری آماده")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up directories: {str(e)}")
            return False
    
    def _setup_configuration(self) -> bool:
        """Setup and validate configuration"""
        
        try:
            config_manager = get_config_manager()
            config = config_manager.load_config()
            
            print(f"  ✅ پیکربندی بارگذاری شد")
            print(f"  📋 پروژه: {config.project_name}")
            print(f"  🌍 محیط: {config.environment}")
            print(f"  🤖 مدل LLM: {config.llm.ollama_model}")
            
            # Check critical settings
            critical_checks = [
                (config.database.url, "Database URL"),
                (config.llm.ollama_base_url, "Ollama URL"),
                (config.rag.embedding_model, "Embedding Model"),
                (config.web.port, "Web Port")
            ]
            
            all_critical_ok = True
            for value, name in critical_checks:
                if value:
                    print(f"  ✅ {name}")
                else:
                    print(f"  ❌ {name} مشخص نشده")
                    all_critical_ok = False
            
            # Create .env file if it doesn't exist
            env_file = BASE_DIR / ".env"
            if not env_file.exists():
                print(f"  🔧 ایجاد فایل .env")
                env_template = BASE_DIR / ".env.template"
                if env_template.exists():
                    shutil.copy(env_template, env_file)
                    print(f"  ✅ فایل .env از template ایجاد شد")
                else:
                    # Create basic .env
                    with open(env_file, 'w', encoding='utf-8') as f:
                        f.write(f"PROJECT_NAME={config.project_name}\n")
                        f.write(f"ENVIRONMENT={config.environment}\n")
                        f.write(f"DEBUG={str(config.debug).lower()}\n")
                    print(f"  ✅ فایل .env پایه ایجاد شد")
            else:
                print(f"  ✅ فایل .env موجود است")
            
            return all_critical_ok
            
        except Exception as e:
            self.logger.error(f"Error setting up configuration: {str(e)}")
            return False
    
    def _check_ollama_setup(self) -> bool:
        """Check Ollama installation and models"""
        
        try:
            # Check if Ollama is available
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
            
            if result.returncode != 0:
                print("  ❌ Ollama not found")
                print("  💡 نصب Ollama: https://ollama.ai/download")
                return False
            
            ollama_version = result.stdout.strip()
            print(f"  ✅ {ollama_version}")
            
            # Check if Ollama service is running
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            
            if result.returncode != 0:
                print("  ❌ سرویس Ollama در حال اجرا نیست")
                print("  💡 شروع سرویس: ollama serve")
                return False
            
            # Parse available models
            models_output = result.stdout.strip()
            if not models_output or "NAME" not in models_output:
                print("  ⚠️ هیچ مدلی نصب نشده")
                return False
            
            # Extract model names
            lines = models_output.split('\n')[1:]  # Skip header
            available_models = []
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    available_models.append(model_name)
                    print(f"  📖 {model_name}")
            
            # Check if required models are available
            config = get_config_manager().get_config()
            required_model = config.llm.ollama_model
            backup_model = config.llm.ollama_backup_model
            
            model_check = False
            if any(required_model in model for model in available_models):
                print(f"  ✅ مدل اصلی موجود: {required_model}")
                model_check = True
            elif any(backup_model in model for model in available_models):
                print(f"  ✅ مدل پشتیبان موجود: {backup_model}")
                model_check = True
            else:
                print(f"  ⚠️ مدل‌های مورد نیاز موجود نیست")
                print(f"  💡 نصب مدل: ollama pull {required_model}")
            
            return model_check
            
        except FileNotFoundError:
            print("  ❌ Ollama not found in PATH")
            return False
        except Exception as e:
            self.logger.error(f"Error checking Ollama: {str(e)}")
            return False
    
    def _final_validation(self) -> bool:
        """Perform final environment validation"""
        
        try:
            validation_checks = []
            
            # Check Python imports
            try:
                from shared_utils import get_logger, get_config, read_document
                validation_checks.append(("Shared utilities import", True))
                print("  ✅ Shared utilities")
            except ImportError as e:
                validation_checks.append(("Shared utilities import", False))
                print(f"  ❌ Shared utilities: {e}")
            
            # Check configuration loading
            try:
                config = get_config_manager().get_config()
                validation_checks.append(("Configuration loading", True))
                print("  ✅ Configuration loading")
            except Exception as e:
                validation_checks.append(("Configuration loading", False))
                print(f"  ❌ Configuration loading: {e}")
            
            # Check logging system
            try:
                logger = get_logger("ValidationTest")
                logger.info("Test log message", "پیام تست")
                validation_checks.append(("Logging system", True))
                print("  ✅ Logging system")
            except Exception as e:
                validation_checks.append(("Logging system", False))
                print(f"  ❌ Logging system: {e}")
            
            # Check file utilities
            try:
                from shared_utils.file_utils import get_file_info
                test_file = Path(__file__)
                file_info = get_file_info(test_file)
                validation_checks.append(("File utilities", True))
                print("  ✅ File utilities")
            except Exception as e:
                validation_checks.append(("File utilities", False))
                print(f"  ❌ File utilities: {e}")
            
            # Check essential directories
            essential_dirs = ["data/raw", "config", "logs"]
            dir_check = True
            
            for dir_path in essential_dirs:
                full_path = BASE_DIR / dir_path
                if not full_path.exists():
                    dir_check = False
                    print(f"  ❌ Missing directory: {dir_path}")
            
            if dir_check:
                validation_checks.append(("Essential directories", True))
                print("  ✅ Essential directories")
            else:
                validation_checks.append(("Essential directories", False))
            
            # Overall validation result
            all_passed = all(result for _, result in validation_checks)
            
            print(f"\n  📊 نتیجه اعتبارسنجی:")
            passed_count = sum(1 for _, result in validation_checks if result)
            total_count = len(validation_checks)
            print(f"    {passed_count}/{total_count} تست موفق")
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Error in final validation: {str(e)}")
            return False
    
    def _generate_setup_report(self):
        """Generate detailed setup report"""
        
        try:
            report = {
                "setup_date": datetime.now().isoformat(),
                "project_name": PROJECT_NAME,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.platform(),
                "setup_results": self.setup_results,
                "success_rate": f"{sum(self.setup_results.values())}/{len(self.setup_results)}",
                "next_steps": []
            }
            
            # Add next steps based on results
            if not self.setup_results['python_check']:
                report["next_steps"].append("Upgrade Python to 3.11+")
            
            if not self.setup_results['conda_check']:
                report["next_steps"].append("Install or configure Conda")
            
            if not self.setup_results['ollama_check']:
                report["next_steps"].append("Install Ollama and required models")
            
            if all(self.setup_results.values()):
                report["next_steps"].append("محیط آماده است - شروع فاز یک")
                report["next_phase"] = "Phase 1: Data Processing"
            
            # Save report
            report_file = BASE_DIR / "logs" / "environment_setup_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(
                f"Setup report saved: {report_file}",
                f"گزارش تنظیم محیط ذخیره شد: {report_file}"
            )
            
        except Exception as e:
            self.logger.error(f"Error generating setup report: {str(e)}")
    
    def _print_step(self, message: str):
        """Print setup step with formatting"""
        print(f"\n{message}")
        print("=" * len(message))
    
    def _print_success_message(self):
        """Print success message"""
        print("\n" + "🎉" * 50)
        print("🎉 محیط توسعه با موفقیت تنظیم شد!")
        print("🎉 Environment Setup Completed Successfully!")
        print("🎉" * 50)
        print("\n📋 وضعیت نهایی:")
        
        for step, success in self.setup_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {step}")
        
        print("\n🚀 مراحل بعدی:")
        print("  1. بررسی فایل‌های خام در data/raw/")
        print("  2. اجرای فاز یک: python phase_1_data_processing/document_parser.py")
        print("  3. مطالعه مستندات در docs/")
        
        self.logger.info(
            "Environment setup completed successfully",
            "تنظیم محیط با موفقیت تکمیل شد"
        )
    
    def _print_failure_message(self):
        """Print failure message with guidance"""
        print("\n" + "❌" * 50)
        print("❌ تنظیم محیط با مشکل مواجه شد!")
        print("❌ Environment Setup Failed!")
        print("❌" * 50)
        print("\n📋 وضعیت:")
        
        for step, success in self.setup_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {step}")
        
        print("\n🔧 اقدامات پیشنهادی:")
        
        if not self.setup_results['python_check']:
            print("  • ارتقاء Python به نسخه 3.11 یا بالاتر")
        
        if not self.setup_results['conda_check']:
            print("  • نصب یا پیکربندی Conda")
            print("  • فعال‌سازی محیط: conda activate claude-ai")
        
        if not self.setup_results['dependencies_check']:
            print("  • نصب پکیج‌های مفقود: pip install -r requirements.txt")
        
        if not self.setup_results['ollama_check']:
            print("  • نصب Ollama: https://ollama.ai/download")
            print("  • نصب مدل: ollama pull qwen2.5:7b-instruct")
        
        print("\n💡 برای کمک بیشتر:")
        print("  • بررسی فایل logs/environment_setup_report.json")
        print("  • مطالعه مستندات در docs/setup_guide.md")
        
        self.logger.error(
            "Environment setup failed - see report for details",
            "تنظیم محیط ناموفق - گزارش تفصیلی را بررسی کنید"
        )


def main():
    """Main execution function"""
    
    print("=" * 70)
    print(f"🛠️  {PROJECT_NAME} - Environment Setup")
    print("🛠️  تنظیم خودکار محیط توسعه")
    print("=" * 70)
    
    # Start system logging
    log_system_startup()
    
    # Create and run setup
    setup = EnvironmentSetup()
    success = setup.run_full_setup()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()