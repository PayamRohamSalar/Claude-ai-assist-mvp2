# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\phase_0_setup\validate_setup.py

"""
Legal Assistant AI - Setup Validation Script
Validates that the development environment is properly configured for Phase 1
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import importlib.util

# Add parent directory to path to import shared utilities
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils.logger import get_logger
from shared_utils.config_manager import get_config_manager
from shared_utils.constants import PROJECT_NAME, BASE_DIR
from shared_utils.file_utils import get_file_manager


class SetupValidator:
    """
    Comprehensive validation of the Legal Assistant AI development environment
    """
    
    def __init__(self):
        self.logger = get_logger("SetupValidator")
        self.validation_results = {}
        self.critical_errors = []
        self.warnings = []
        self.recommendations = []
        
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run complete environment validation
        
        Returns:
            Dict containing validation results and recommendations
        """
        
        self.logger.info(
            "Starting comprehensive environment validation",
            "شروع اعتبارسنجی جامع محیط توسعه"
        )
        
        try:
            validation_steps = [
                ("🐍 Python Environment", self._validate_python_environment),
                ("📦 Dependencies", self._validate_dependencies),
                ("📁 Directory Structure", self._validate_directory_structure),
                ("⚙️ Configuration", self._validate_configuration),
                ("📝 Logging System", self._validate_logging_system),
                ("🗃️ Database Setup", self._validate_database_setup),
                ("🤖 LLM Integration", self._validate_llm_integration),
                ("📊 File Processing", self._validate_file_processing),
                ("🔧 Shared Utils", self._validate_shared_utilities),
                ("🚀 Phase 1 Readiness", self._validate_phase1_readiness)
            ]
            
            print("=" * 70)
            print(f"🔍 {PROJECT_NAME} - Environment Validation")
            print("🔍 اعتبارسنجی محیط توسعه")
            print("=" * 70)
            
            for step_name, validator_func in validation_steps:
                print(f"\n{step_name}")
                print("-" * len(step_name))
                
                try:
                    result = validator_func()
                    self.validation_results[step_name] = result
                    
                    if result.get('status') == 'pass':
                        print("✅ موفق")
                    elif result.get('status') == 'warning':
                        print("⚠️ هشدار")
                        self.warnings.extend(result.get('warnings', []))
                    else:
                        print("❌ ناموفق")
                        self.critical_errors.extend(result.get('errors', []))
                    
                    # Display details
                    for detail in result.get('details', []):
                        print(f"  {detail}")
                        
                except Exception as e:
                    print(f"❌ خطا در اعتبارسنجی: {str(e)}")
                    self.critical_errors.append(f"{step_name}: {str(e)}")
                    self.validation_results[step_name] = {
                        'status': 'error',
                        'errors': [str(e)]
                    }
            
            # Generate final report
            final_report = self._generate_validation_report()
            
            # Display summary
            self._display_validation_summary()
            
            return final_report
            
        except Exception as e:
            self.logger.error(
                f"Validation process failed: {str(e)}",
                f"فرآیند اعتبارسنجی ناموفق: {str(e)}"
            )
            return {'status': 'failed', 'error': str(e)}
    
    def _validate_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment setup"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            # Check Python version
            version = sys.version_info
            version_str = f"{version.major}.{version.minor}.{version.micro}"
            
            if version >= (3, 10):
                details.append(f"✅ Python {version_str}")
            elif version >= (3, 9):
                details.append(f"⚠️ Python {version_str} (پیشنهاد: 3.10+)")
                warnings.append("Python version should be 3.10+ for optimal performance")
            else:
                details.append(f"❌ Python {version_str} (نیاز: 3.11+)")
                errors.append("Python version is too old")
            
            # Check virtual environment
            venv = os.environ.get('CONDA_DEFAULT_ENV')
            if venv and venv != 'base':
                details.append(f"✅ Virtual environment: {venv}")
            else:
                details.append("⚠️ No virtual environment detected")
                warnings.append("Consider using conda environment for isolation")
            
            # Check pip
            try:
                import pip
                details.append(f"✅ pip available")
            except ImportError:
                details.append("❌ pip not available")
                errors.append("pip is required for package management")
            
            status = 'error' if errors else ('warning' if warnings else 'pass')
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"Python environment check failed: {str(e)}"],
                'details': []
            }
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate required dependencies"""
        
        details = []
        errors = []
        warnings = []
        
        required_packages = [
            ('pandas', 'Data manipulation'),
            ('numpy', 'Numerical computing'),
            ('fastapi', 'Web framework'),
            ('python-dotenv', 'Environment variables'),
            ('PyPDF2', 'PDF processing'),
            ('python-docx', 'Word document processing'),
            ('chardet', 'Character encoding detection'),
            ('json', 'JSON handling (builtin)'),
            ('pathlib', 'Path handling (builtin)')
        ]
        
        optional_packages = [
            ('transformers', 'HuggingFace transformers'),
            ('sentence-transformers', 'Sentence embeddings'),
            ('chromadb', 'Vector database'),
            ('openai', 'OpenAI API'),
            ('anthropic', 'Anthropic API')
        ]
        
        # Check required packages
        missing_required = []
        for package, description in required_packages:
            try:
                if package in ['json', 'pathlib']:
                    # Built-in modules
                    __import__(package)
                else:
                    __import__(package.replace('-', '_'))
                details.append(f"✅ {package}: {description}")
            except ImportError:
                details.append(f"❌ {package}: {description}")
                missing_required.append(package)
        
        if missing_required:
            errors.append(f"Missing required packages: {', '.join(missing_required)}")
        
        # Check optional packages
        missing_optional = []
        for package, description in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                details.append(f"✅ {package}: {description}")
            except ImportError:
                details.append(f"⚠️ {package}: {description} (optional)")
                missing_optional.append(package)
        
        if missing_optional:
            warnings.append(f"Optional packages not installed: {', '.join(missing_optional)}")
        
        status = 'error' if errors else ('warning' if warnings else 'pass')
        
        return {
            'status': status,
            'details': details,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_directory_structure(self) -> Dict[str, Any]:
        """Validate project directory structure"""
        
        details = []
        errors = []
        warnings = []
        
        required_directories = [
            ('data', 'Data storage'),
            ('data/raw', 'Raw documents'),
            ('data/processed_phase_1', 'Phase 1 output'),
            ('data/vector_db', 'Vector database'),
            ('config', 'Configuration files'),
            ('logs', 'Log files'),
            ('shared_utils', 'Shared utilities'),
            ('phase_0_setup', 'Phase 0 setup scripts'),
            ('tests', 'Test files')
        ]
        
        optional_directories = [
            ('backup', 'Backup storage'),
            ('docs', 'Documentation'),
            ('phase_1_data_processing', 'Phase 1 modules')
        ]
        
        # Check required directories
        missing_required = []
        for dir_path, description in required_directories:
            full_path = BASE_DIR / dir_path
            if full_path.exists() and full_path.is_dir():
                details.append(f"✅ {dir_path}: {description}")
            else:
                details.append(f"❌ {dir_path}: {description}")
                missing_required.append(dir_path)
        
        if missing_required:
            errors.append(f"Missing required directories: {', '.join(missing_required)}")
        
        # Check optional directories
        for dir_path, description in optional_directories:
            full_path = BASE_DIR / dir_path
            if full_path.exists() and full_path.is_dir():
                details.append(f"✅ {dir_path}: {description}")
            else:
                details.append(f"⚠️ {dir_path}: {description} (optional)")
        
        # Check directory permissions
        test_dirs = ['data', 'logs', 'config']
        for dir_name in test_dirs:
            dir_path = BASE_DIR / dir_name
            if dir_path.exists():
                if os.access(dir_path, os.R_OK | os.W_OK):
                    details.append(f"✅ {dir_name}: read/write permissions")
                else:
                    details.append(f"❌ {dir_name}: insufficient permissions")
                    errors.append(f"Directory {dir_name} lacks read/write permissions")
        
        status = 'error' if errors else 'pass'
        
        return {
            'status': status,
            'details': details,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration setup"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            # Check config.json exists
            config_file = BASE_DIR / "config" / "config.json"
            if config_file.exists():
                details.append("✅ config.json exists")
                
                # Validate JSON syntax
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    details.append("✅ config.json has valid syntax")
                    
                    # Check essential sections
                    required_sections = ['project_name', 'llm', 'rag', 'database']
                    for section in required_sections:
                        if section in config_data:
                            details.append(f"✅ {section} section present")
                        else:
                            details.append(f"❌ {section} section missing")
                            errors.append(f"Config section {section} is missing")
                    
                except json.JSONDecodeError as e:
                    details.append(f"❌ config.json syntax error: {str(e)}")
                    errors.append("Configuration file has invalid JSON syntax")
                    
            else:
                details.append("❌ config.json not found")
                errors.append("Configuration file is missing")
            
            # Check ConfigManager functionality
            try:
                config_manager = get_config_manager()
                config = config_manager.get_config()
                details.append("✅ ConfigManager working")
                details.append(f"✅ Project: {config.project_name}")
                
                # Test dot notation access
                test_value = config_manager.get_value("project_name")
                if test_value:
                    details.append("✅ Dot notation access working")
                else:
                    warnings.append("Dot notation access may have issues")
                    
            except Exception as e:
                details.append(f"❌ ConfigManager error: {str(e)}")
                errors.append("Configuration manager is not working properly")
            
            # Check .env file
            env_file = BASE_DIR / ".env"
            if env_file.exists():
                details.append("✅ .env file exists")
            else:
                details.append("⚠️ .env file not found")
                warnings.append(".env file recommended for environment variables")
            
            status = 'error' if errors else ('warning' if warnings else 'pass')
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"Configuration validation failed: {str(e)}"],
                'details': []
            }
    
    def _validate_logging_system(self) -> Dict[str, Any]:
        """Validate logging system"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            # Test logger creation
            logger = get_logger("ValidationTest")
            details.append("✅ Logger creation successful")
            
            # Test logging functionality
            test_message = "Validation test message"
            persian_message = "پیام تست اعتبارسنجی"
            
            logger.info(test_message, persian_message)
            details.append("✅ Logging functionality working")
            
            # Check log files
            logs_dir = BASE_DIR / "logs"
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.jsonl"))
                if log_files:
                    details.append(f"✅ {len(log_files)} log files found")
                    for log_file in log_files[:3]:  # Show first 3
                        details.append(f"  📄 {log_file.name}")
                else:
                    details.append("⚠️ No log files found")
                    warnings.append("No existing log files found")
            else:
                details.append("❌ Logs directory not found")
                errors.append("Logs directory is missing")
            
            # Test Persian logging
            try:
                from shared_utils.logger import log_info
                log_info("Test info message", "پیام تست اطلاعات")
                details.append("✅ Persian logging working")
            except Exception as e:
                details.append(f"⚠️ Persian logging issue: {str(e)}")
                warnings.append("Persian logging may have issues")
            
            status = 'error' if errors else ('warning' if warnings else 'pass')
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"Logging system validation failed: {str(e)}"],
                'details': []
            }
    
    def _validate_database_setup(self) -> Dict[str, Any]:
        """Validate database configuration"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            config = get_config_manager().get_config()
            
            # Check database URL
            db_url = config.database.url
            if db_url:
                details.append(f"✅ Database URL: {db_url}")
                
                # Check if SQLite file can be created
                if db_url.startswith('sqlite:'):
                    db_path = db_url.replace('sqlite:///', '')
                    db_file = Path(db_path)
                    
                    try:
                        # Test database file creation
                        db_file.parent.mkdir(parents=True, exist_ok=True)
                        details.append("✅ Database directory accessible")
                    except Exception as e:
                        details.append(f"❌ Database directory error: {str(e)}")
                        errors.append("Cannot access database directory")
                        
            else:
                details.append("❌ Database URL not configured")
                errors.append("Database URL is not set")
            
            # Check vector database path
            vector_db_path = config.database.vector_db_path
            if vector_db_path:
                details.append(f"✅ Vector DB path: {vector_db_path}")
                
                vector_dir = Path(vector_db_path)
                try:
                    vector_dir.mkdir(parents=True, exist_ok=True)
                    details.append("✅ Vector DB directory accessible")
                except Exception as e:
                    details.append(f"❌ Vector DB directory error: {str(e)}")
                    errors.append("Cannot access vector database directory")
            else:
                details.append("❌ Vector DB path not configured")
                errors.append("Vector database path is not set")
            
            # Check SQLite availability
            try:
                import sqlite3
                details.append("✅ SQLite module available")
            except ImportError:
                details.append("❌ SQLite module not available")
                errors.append("SQLite is not available")
            
            status = 'error' if errors else 'pass'
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"Database validation failed: {str(e)}"],
                'details': []
            }
    
    def _validate_llm_integration(self) -> Dict[str, Any]:
        """Validate LLM integration setup"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            config = get_config_manager().get_config()
            
            # Check Ollama configuration
            ollama_url = config.llm.ollama_base_url
            ollama_model = config.llm.ollama_model
            
            details.append(f"✅ Ollama URL: {ollama_url}")
            details.append(f"✅ Ollama Model: {ollama_model}")
            
            # Test Ollama availability
            try:
                result = subprocess.run(['ollama', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    details.append("✅ Ollama CLI available")
                    
                    # Check models
                    result = subprocess.run(['ollama', 'list'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        models_output = result.stdout
                        if ollama_model.split(':')[0] in models_output:
                            details.append(f"✅ Model {ollama_model} available")
                        else:
                            details.append(f"⚠️ Model {ollama_model} not found")
                            warnings.append(f"Recommended model {ollama_model} is not installed")
                    else:
                        details.append("⚠️ Cannot list Ollama models")
                        warnings.append("Ollama service may not be running")
                else:
                    details.append("❌ Ollama CLI not working")
                    errors.append("Ollama CLI is not responding")
                    
            except (FileNotFoundError, subprocess.TimeoutExpired):
                details.append("⚠️ Ollama not found or not responding")
                warnings.append("Ollama is not installed or not running")
            
            # Check API keys (without revealing them)
            openai_key = config.llm.openai_api_key
            anthropic_key = config.llm.anthropic_api_key
            
            if openai_key and openai_key != "":
                details.append("✅ OpenAI API key configured")
            else:
                details.append("⚠️ OpenAI API key not set")
                warnings.append("OpenAI API key is not configured")
            
            if anthropic_key and anthropic_key != "":
                details.append("✅ Anthropic API key configured")
            else:
                details.append("⚠️ Anthropic API key not set")
                warnings.append("Anthropic API key is not configured")
            
            # Check LLM parameters
            temperature = config.llm.temperature
            if 0 <= temperature <= 2:
                details.append(f"✅ Temperature: {temperature}")
            else:
                details.append(f"⚠️ Temperature: {temperature} (should be 0-2)")
                warnings.append("LLM temperature is outside recommended range")
            
            status = 'error' if errors else ('warning' if warnings else 'pass')
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"LLM integration validation failed: {str(e)}"],
                'details': []
            }
    
    def _validate_file_processing(self) -> Dict[str, Any]:
        """Validate file processing capabilities"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            # Test file utilities import
            from shared_utils.file_utils import DocumentReader, FileManager, get_file_info
            details.append("✅ File utilities import successful")
            
            # Test document reader
            reader = DocumentReader()
            details.append("✅ DocumentReader instantiation")
            
            # Check supported formats
            supported_formats = reader.supported_formats
            details.append(f"✅ Supported formats: {len(supported_formats)}")
            for fmt in supported_formats.keys():
                details.append(f"  📄 {fmt}")
            
            # Test file manager
            file_manager = FileManager()
            details.append("✅ FileManager instantiation")
            
            # Test file info functionality
            test_file = Path(__file__)
            file_info = get_file_info(test_file)
            if file_info and file_info.name:
                details.append("✅ File info functionality working")
            else:
                details.append("❌ File info functionality failed")
                errors.append("File info functionality is not working")
            
            # Check critical file processing dependencies
            critical_packages = [
                ('PyPDF2', 'PDF processing'),
                ('python-docx', 'Word document processing'),
                ('chardet', 'Encoding detection')
            ]
            
            for package, description in critical_packages:
                try:
                    __import__(package.replace('-', '_'))
                    details.append(f"✅ {package}: {description}")
                except ImportError:
                    details.append(f"❌ {package}: {description}")
                    errors.append(f"Missing critical package: {package}")
            
            # Test basic file operations
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write("Test content")
                    temp_file = Path(f.name)
                
                # Test reading
                result = reader.read_document(temp_file)
                if result['success']:
                    details.append("✅ File reading test successful")
                else:
                    details.append("❌ File reading test failed")
                    errors.append("File reading functionality is not working")
                
                # Clean up
                temp_file.unlink()
                
            except Exception as e:
                details.append(f"⚠️ File operation test failed: {str(e)}")
                warnings.append("File operations may have issues")
            
            status = 'error' if errors else ('warning' if warnings else 'pass')
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"File processing validation failed: {str(e)}"],
                'details': []
            }
    
    def _validate_shared_utilities(self) -> Dict[str, Any]:
        """Validate shared utilities package"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            # Test package imports
            test_imports = [
                ('shared_utils.constants', 'Constants module'),
                ('shared_utils.logger', 'Logger module'),
                ('shared_utils.config_manager', 'Config manager module'),
                ('shared_utils.file_utils', 'File utils module')
            ]
            
            for module_name, description in test_imports:
                try:
                    importlib.import_module(module_name)
                    details.append(f"✅ {module_name}: {description}")
                except ImportError as e:
                    details.append(f"❌ {module_name}: {description} - {str(e)}")
                    errors.append(f"Cannot import {module_name}")
            
            # Test package-level imports
            try:
                from shared_utils import (
                    get_logger, get_config, read_document,
                    PROJECT_NAME, Messages
                )
                details.append("✅ Package-level imports working")
                
                # Test functionality
                logger = get_logger("PackageTest")
                config = get_config()
                
                if logger and config:
                    details.append("✅ Package functionality working")
                else:
                    details.append("❌ Package functionality failed")
                    errors.append("Package functionality is not working properly")
                    
            except ImportError as e:
                details.append(f"❌ Package imports failed: {str(e)}")
                errors.append("Package-level imports are not working")
            
            # Check constants
            try:
                from shared_utils.constants import (
                    DocumentType, ApprovalAuthority, PERSIAN_DIGITS
                )
                details.append("✅ Constants accessible")
                
                # Test enum functionality
                doc_types = list(DocumentType)
                if doc_types:
                    details.append(f"✅ Document types: {len(doc_types)} defined")
                else:
                    warnings.append("Document types enum may be empty")
                    
            except Exception as e:
                details.append(f"⚠️ Constants access issue: {str(e)}")
                warnings.append("Constants module may have issues")
            
            # Test utility functions
            try:
                from shared_utils.constants import (
                    persian_to_english_digits, validate_file_extension
                )
                
                # Test digit conversion
                test_text = "۱۲۳"
                converted = persian_to_english_digits(test_text)
                if converted == "123":
                    details.append("✅ Persian digit conversion working")
                else:
                    details.append("⚠️ Persian digit conversion issue")
                    warnings.append("Persian digit conversion may not work correctly")
                
                # Test file validation
                valid_pdf = validate_file_extension("test.pdf")
                invalid_xyz = validate_file_extension("test.xyz")
                
                if valid_pdf and not invalid_xyz:
                    details.append("✅ File validation working")
                else:
                    details.append("⚠️ File validation issue")
                    warnings.append("File validation may not work correctly")
                    
            except Exception as e:
                details.append(f"⚠️ Utility functions issue: {str(e)}")
                warnings.append("Utility functions may have issues")
            
            status = 'error' if errors else ('warning' if warnings else 'pass')
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"Shared utilities validation failed: {str(e)}"],
                'details': []
            }
    
    def _validate_phase1_readiness(self) -> Dict[str, Any]:
        """Validate readiness for Phase 1"""
        
        details = []
        errors = []
        warnings = []
        
        try:
            # Check raw data directory
            raw_data_dir = BASE_DIR / "data" / "raw"
            if raw_data_dir.exists():
                details.append("✅ Raw data directory exists")
                
                # Check for expected files (from phase settings)
                expected_files = [
                    "Part1_Policies.docx",
                    "Part2_Laws.docx", 
                    "Part3_Regulations.docx",
                    "Part4_Supreme_Council.docx",
                    "Part5_Science_Council.docx",
                    "Part6_Ministry.docx",
                    "Part7_Judiciary.docx"
                ]
                
                found_files = []
                missing_files = []
                
                for expected_file in expected_files:
                    file_path = raw_data_dir / expected_file
                    if file_path.exists():
                        found_files.append(expected_file)
                        details.append(f"✅ {expected_file}")
                    else:
                        missing_files.append(expected_file)
                        details.append(f"⚠️ {expected_file} (missing)")
                
                if found_files:
                    details.append(f"✅ {len(found_files)} data files found")
                else:
                    warnings.append("No input data files found")
                
                if missing_files:
                    warnings.append(f"Missing {len(missing_files)} expected files")
                    
            else:
                details.append("❌ Raw data directory missing")
                errors.append("Raw data directory does not exist")
            
            # Check output directory
            output_dir = BASE_DIR / "data" / "processed_phase_1"
            if output_dir.exists():
                details.append("✅ Phase 1 output directory exists")
            else:
                details.append("❌ Phase 1 output directory missing")
                errors.append("Phase 1 output directory does not exist")
            
            # Check if Phase 1 modules directory exists
            phase1_dir = BASE_DIR / "phase_1_data_processing"
            if phase1_dir.exists():
                details.append("✅ Phase 1 modules directory exists")
            else:
                details.append("⚠️ Phase 1 modules directory not yet created")
                warnings.append("Phase 1 modules will be created in next phase")
            
            # Check configuration for Phase 1
            try:
                config = get_config_manager().get_config()
                
                # Check processing settings
                max_file_size = config.processing.max_file_size
                batch_size = config.processing.batch_size
                
                details.append(f"✅ Max file size: {max_file_size / (1024*1024):.1f} MB")
                details.append(f"✅ Batch size: {batch_size}")
                
                # Check RAG settings for text processing
                chunk_size = config.rag.chunk_size
                if 100 <= chunk_size <= 2000:
                    details.append(f"✅ Chunk size: {chunk_size} (appropriate)")
                else:
                    details.append(f"⚠️ Chunk size: {chunk_size} (may need adjustment)")
                    warnings.append("Chunk size may not be optimal for legal documents")
                
            except Exception as e:
                details.append(f"⚠️ Configuration check failed: {str(e)}")
                warnings.append("Could not verify Phase 1 configuration settings")
            
            # Overall readiness assessment
            if not errors and len(warnings) <= 2:
                details.append("🚀 Ready for Phase 1")
                self.recommendations.append("محیط آماده شروع فاز یک است")
                self.recommendations.append("فایل‌های خام را در data/raw قرار دهید")
                self.recommendations.append("سپس اجرا کنید: python phase_1_data_processing/document_parser.py")
            elif not errors:
                details.append("⚠️ Mostly ready for Phase 1 (با هشدارها)")
                self.recommendations.append("محیط تقریباً آماده است اما نیاز به بررسی هشدارها دارد")
            else:
                details.append("❌ Not ready for Phase 1")
                self.recommendations.append("ابتدا خطاهای موجود را برطرف کنید")
            
            status = 'error' if errors else ('warning' if warnings else 'pass')
            
            return {
                'status': status,
                'details': details,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'errors': [f"Phase 1 readiness validation failed: {str(e)}"],
                'details': []
            }
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        try:
            # Calculate statistics
            total_checks = len(self.validation_results)
            passed_checks = sum(1 for result in self.validation_results.values() 
                              if result.get('status') == 'pass')
            warning_checks = sum(1 for result in self.validation_results.values() 
                               if result.get('status') == 'warning')
            error_checks = sum(1 for result in self.validation_results.values() 
                             if result.get('status') == 'error')
            
            # Overall status
            if error_checks > 0:
                overall_status = 'failed'
            elif warning_checks > 0:
                overall_status = 'warning'
            else:
                overall_status = 'passed'
            
            # Create report
            report = {
                'validation_date': datetime.now().isoformat(),
                'project_name': PROJECT_NAME,
                'overall_status': overall_status,
                'statistics': {
                    'total_checks': total_checks,
                    'passed': passed_checks,
                    'warnings': warning_checks,
                    'errors': error_checks,
                    'success_rate': f"{passed_checks}/{total_checks}"
                },
                'validation_results': self.validation_results,
                'critical_errors': self.critical_errors,
                'warnings': self.warnings,
                'recommendations': self.recommendations,
                'environment_info': {
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'platform': sys.platform,
                    'working_directory': str(BASE_DIR)
                }
            }
            
            # Save report
            report_file = BASE_DIR / "logs" / "setup_validation_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(
                f"Validation report saved: {report_file}",
                f"گزارش اعتبارسنجی ذخیره شد: {report_file}"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {str(e)}")
            return {
                'status': 'error',
                'error': f"Report generation failed: {str(e)}"
            }
    
    def _display_validation_summary(self):
        """Display validation summary"""
        
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY / خلاصه اعتبارسنجی")
        print("=" * 70)
        
        # Statistics
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results.values() 
                          if result.get('status') == 'pass')
        warning_checks = sum(1 for result in self.validation_results.values() 
                           if result.get('status') == 'warning')
        error_checks = sum(1 for result in self.validation_results.values() 
                         if result.get('status') == 'error')
        
        print(f"📈 Total Checks: {total_checks}")
        print(f"✅ Passed: {passed_checks}")
        print(f"⚠️ Warnings: {warning_checks}")
        print(f"❌ Errors: {error_checks}")
        print(f"📊 Success Rate: {passed_checks}/{total_checks} ({(passed_checks/total_checks)*100:.1f}%)")
        
        # Overall status
        if error_checks > 0:
            print("\n🔴 OVERALL STATUS: FAILED")
            print("🔴 وضعیت کلی: ناموفق")
            print("\n❌ Critical Errors:")
            for error in self.critical_errors[:5]:  # Show first 5
                print(f"  • {error}")
        elif warning_checks > 0:
            print("\n🟡 OVERALL STATUS: PASSED WITH WARNINGS")
            print("🟡 وضعیت کلی: موفق با هشدار")
        else:
            print("\n🟢 OVERALL STATUS: PASSED")
            print("🟢 وضعیت کلی: موفق")
        
        # Warnings (if any)
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:5]:  # Show first 5
                print(f"  • {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        # Recommendations
        if self.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in self.recommendations:
                print(f"  • {rec}")
        
        # Next steps
        print(f"\n🚀 Next Steps:")
        if error_checks > 0:
            print("  1. برطرف کردن خطاهای critical")
            print("  2. اجرای مجدد validation")
            print("  3. مطالعه گزارش تفصیلی در logs/")
        elif warning_checks > 0:
            print("  1. بررسی هشدارها (اختیاری)")
            print("  2. شروع فاز یک")
            print("  3. قرار دادن فایل‌های خام در data/raw/")
        else:
            print("  1. شروع فاز یک ✅")
            print("  2. قرار دادن فایل‌های خام در data/raw/")
            print("  3. اجرا: python phase_1_data_processing/document_parser.py")
        
        print("\n📄 Detailed report: logs/setup_validation_report.json")
        print("=" * 70)


def main():
    """Main execution function"""
    
    validator = SetupValidator()
    report = validator.run_full_validation()
    
    # Exit with appropriate code
    if report.get('overall_status') == 'failed':
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()