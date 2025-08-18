# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\shared_utils\config_manager.py

"""
Legal Assistant AI - Configuration Management System
Manages configuration from multiple sources with validation and type safety
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging

from .constants import CONFIG_DIR, ENV_FILE, BASE_DIR, PROJECT_NAME
from .logger import get_logger


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///legal_assistant.db"
    vector_db_path: str = "./data/vector_db"
    backup_path: str = "./backup/database"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class LLMConfig:
    """Language Model configuration settings"""
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_backup_model: str = "mistral:7b"
    ollama_timeout: int = 300
    ollama_context_length: int = 4096
    
    # API settings
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Generation parameters
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 4096


@dataclass
class RAGConfig:
    """RAG (Retrieval-Augmented Generation) configuration"""
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    persian_embedding_model: str = "HooshvareLab/bert-fa-base-uncased"
    vector_dimension: int = 384
    
    # Search settings
    similarity_threshold: float = 0.75
    max_search_results: int = 10
    search_timeout: int = 30


@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    max_concurrent_jobs: int = 4
    batch_size: int = 50
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = None
    processing_timeout: int = 600  # 10 minutes
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.json']


@dataclass
class WebConfig:
    """Web application configuration"""
    host: str = "localhost"
    port: int = 8000
    reload: bool = True
    workers: int = 1
    request_timeout: int = 30
    max_connections: int = 100
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8080"]


@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    secret_key: str = "your_secret_key_here_change_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    password_salt: str = "your_salt_here"
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_max_size: str = "10MB"
    backup_count: int = 5
    enable_json_logging: bool = True
    enable_console_logging: bool = True


@dataclass
class FeatureFlags:
    """Feature flags for enabling/disabling functionality"""
    enable_document_comparison: bool = True
    enable_draft_generation: bool = True
    enable_advanced_search: bool = True
    enable_export: bool = True
    enable_cache: bool = True
    enable_profiling: bool = False
    mock_external_apis: bool = False
    use_sample_data: bool = False


@dataclass
class AppConfig:
    """Main application configuration combining all sub-configurations"""
    project_name: str = PROJECT_NAME
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Sub-configurations
    database: DatabaseConfig = None
    llm: LLMConfig = None
    rag: RAGConfig = None
    processing: ProcessingConfig = None
    web: WebConfig = None
    security: SecurityConfig = None
    logging: LoggingConfig = None
    features: FeatureFlags = None
    
    def __post_init__(self):
        # Initialize sub-configurations if not provided
        if self.database is None:
            self.database = DatabaseConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.rag is None:
            self.rag = RAGConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.web is None:
            self.web = WebConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.features is None:
            self.features = FeatureFlags()


class ConfigManager:
    """
    Central configuration manager for the Legal Assistant AI system
    Handles loading, validation, and merging of configuration from multiple sources
    """
    
    def __init__(self, config_file: Optional[Path] = None, env_file: Optional[Path] = None):
        self.config_file = config_file or (CONFIG_DIR / "config.json")
        self.env_file = env_file or ENV_FILE
        self.logger = get_logger("ConfigManager")
        self._config: Optional[AppConfig] = None
        self._config_cache: Dict[str, Any] = {}
        
    def load_config(self, force_reload: bool = False) -> AppConfig:
        """
        Load configuration from all sources
        
        Args:
            force_reload: Force reload even if config is cached
            
        Returns:
            Complete application configuration
        """
        
        if self._config is not None and not force_reload:
            return self._config
            
        self.logger.info(
            "Loading configuration",
            "بارگذاری تنظیمات",
            config_file=str(self.config_file),
            env_file=str(self.env_file)
        )
        
        try:
            # Start with default configuration
            config_dict = asdict(AppConfig())
            
            # Load from environment variables
            env_config = self._load_from_env()
            config_dict = self._deep_merge(config_dict, env_config)
            
            # Load from JSON config file
            if self.config_file.exists():
                json_config = self._load_from_json()
                config_dict = self._deep_merge(config_dict, json_config)
            else:
                self.logger.warning(
                    f"Config file not found: {self.config_file}",
                    f"فایل تنظیمات یافت نشد: {self.config_file}"
                )
                # Create default config file
                self._create_default_config_file()
            
            # Convert back to AppConfig object
            self._config = self._dict_to_config(config_dict)
            
            # Validate configuration
            self._validate_config(self._config)
            
            # Apply environment-specific overrides
            self._apply_environment_overrides(self._config)
            
            self.logger.info(
                "Configuration loaded successfully",
                "تنظیمات با موفقیت بارگذاری شد",
                environment=self._config.environment,
                debug=self._config.debug
            )
            
            return self._config
            
        except Exception as e:
            self.logger.error(
                f"Failed to load configuration: {str(e)}",
                f"خطا در بارگذاری تنظیمات: {str(e)}"
            )
            # Return default configuration as fallback
            self._config = AppConfig()
            return self._config
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        
        # Load .env file if it exists
        if self.env_file.exists():
            load_dotenv(self.env_file)
        
        env_config = {}
        
        # Project settings
        if os.getenv('PROJECT_NAME'):
            env_config['project_name'] = os.getenv('PROJECT_NAME')
        if os.getenv('ENVIRONMENT'):
            env_config['environment'] = os.getenv('ENVIRONMENT')
        if os.getenv('DEBUG'):
            env_config['debug'] = os.getenv('DEBUG').lower() == 'true'
        
        # Database settings
        db_config = {}
        if os.getenv('DATABASE_URL'):
            db_config['url'] = os.getenv('DATABASE_URL')
        if os.getenv('VECTOR_DB_PATH'):
            db_config['vector_db_path'] = os.getenv('VECTOR_DB_PATH')
        if os.getenv('BACKUP_DB_PATH'):
            db_config['backup_path'] = os.getenv('BACKUP_DB_PATH')
        if db_config:
            env_config['database'] = db_config
        
        # LLM settings
        llm_config = {}
        if os.getenv('OLLAMA_BASE_URL'):
            llm_config['ollama_base_url'] = os.getenv('OLLAMA_BASE_URL')
        if os.getenv('OLLAMA_MODEL_NAME'):
            llm_config['ollama_model'] = os.getenv('OLLAMA_MODEL_NAME')
        if os.getenv('OLLAMA_BACKUP_MODEL'):
            llm_config['ollama_backup_model'] = os.getenv('OLLAMA_BACKUP_MODEL')
        if os.getenv('OLLAMA_TIMEOUT'):
            llm_config['ollama_timeout'] = int(os.getenv('OLLAMA_TIMEOUT'))
        if os.getenv('OPENAI_API_KEY'):
            llm_config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('ANTHROPIC_API_KEY'):
            llm_config['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
        if llm_config:
            env_config['llm'] = llm_config
        
        # RAG settings
        rag_config = {}
        if os.getenv('MAX_CHUNK_SIZE'):
            rag_config['chunk_size'] = int(os.getenv('MAX_CHUNK_SIZE'))
        if os.getenv('CHUNK_OVERLAP'):
            rag_config['chunk_overlap'] = int(os.getenv('CHUNK_OVERLAP'))
        if os.getenv('EMBEDDING_MODEL'):
            rag_config['embedding_model'] = os.getenv('EMBEDDING_MODEL')
        if os.getenv('SIMILARITY_THRESHOLD'):
            rag_config['similarity_threshold'] = float(os.getenv('SIMILARITY_THRESHOLD'))
        if os.getenv('MAX_SEARCH_RESULTS'):
            rag_config['max_search_results'] = int(os.getenv('MAX_SEARCH_RESULTS'))
        if rag_config:
            env_config['rag'] = rag_config
        
        # Web settings
        web_config = {}
        if os.getenv('HOST'):
            web_config['host'] = os.getenv('HOST')
        if os.getenv('PORT'):
            web_config['port'] = int(os.getenv('PORT'))
        if os.getenv('RELOAD'):
            web_config['reload'] = os.getenv('RELOAD').lower() == 'true'
        if web_config:
            env_config['web'] = web_config
        
        # Security settings
        security_config = {}
        if os.getenv('SECRET_KEY'):
            security_config['secret_key'] = os.getenv('SECRET_KEY')
        if os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'):
            security_config['access_token_expire_minutes'] = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'))
        if security_config:
            env_config['security'] = security_config
        
        return env_config
    
    def _load_from_json(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Invalid JSON in config file: {str(e)}",
                f"فرمت JSON نامعتبر در فایل تنظیمات: {str(e)}"
            )
            return {}
        except Exception as e:
            self.logger.error(
                f"Error reading config file: {str(e)}",
                f"خطا در خواندن فایل تنظیمات: {str(e)}"
            )
            return {}
    
    def _create_default_config_file(self):
        """Create a default configuration file"""
        
        default_config = asdict(AppConfig())
        
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
                
            self.logger.info(
                f"Created default config file: {self.config_file}",
                f"فایل تنظیمات پیش‌فرض ایجاد شد: {self.config_file}"
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to create default config file: {str(e)}",
                f"خطا در ایجاد فایل تنظیمات پیش‌فرض: {str(e)}"
            )
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object"""
        
        # Extract sub-configurations
        database_dict = config_dict.pop('database', {})
        llm_dict = config_dict.pop('llm', {})
        rag_dict = config_dict.pop('rag', {})
        processing_dict = config_dict.pop('processing', {})
        web_dict = config_dict.pop('web', {})
        security_dict = config_dict.pop('security', {})
        logging_dict = config_dict.pop('logging', {})
        features_dict = config_dict.pop('features', {})
        
        # Create sub-configuration objects
        database_config = DatabaseConfig(**database_dict)
        llm_config = LLMConfig(**llm_dict)
        rag_config = RAGConfig(**rag_dict)
        processing_config = ProcessingConfig(**processing_dict)
        web_config = WebConfig(**web_dict)
        security_config = SecurityConfig(**security_dict)
        logging_config = LoggingConfig(**logging_dict)
        features_config = FeatureFlags(**features_dict)
        
        # Create main configuration
        return AppConfig(
            database=database_config,
            llm=llm_config,
            rag=rag_config,
            processing=processing_config,
            web=web_config,
            security=security_config,
            logging=logging_config,
            features=features_config,
            **config_dict
        )
    
    def _validate_config(self, config: AppConfig):
        """Validate configuration values"""
        
        validation_errors = []
        
        # Validate database configuration
        if not config.database.url:
            validation_errors.append("Database URL is required")
        
        # Validate LLM configuration
        if config.llm.temperature < 0 or config.llm.temperature > 2:
            validation_errors.append("LLM temperature must be between 0 and 2")
        
        if config.llm.top_p < 0 or config.llm.top_p > 1:
            validation_errors.append("LLM top_p must be between 0 and 1")
        
        # Validate RAG configuration
        if config.rag.chunk_size < config.rag.min_chunk_size:
            validation_errors.append("Chunk size cannot be less than minimum chunk size")
        
        if config.rag.chunk_size > config.rag.max_chunk_size:
            validation_errors.append("Chunk size cannot be greater than maximum chunk size")
        
        if config.rag.similarity_threshold < 0 or config.rag.similarity_threshold > 1:
            validation_errors.append("Similarity threshold must be between 0 and 1")
        
        # Validate web configuration
        if config.web.port < 1 or config.web.port > 65535:
            validation_errors.append("Web port must be between 1 and 65535")
        
        # Validate security configuration
        if config.environment == "production" and config.security.secret_key == "your_secret_key_here_change_in_production":
            validation_errors.append("Secret key must be changed in production environment")
        
        if validation_errors:
            error_msg = "Configuration validation failed: " + "; ".join(validation_errors)
            persian_msg = "اعتبارسنجی تنظیمات ناموفق: " + "; ".join(validation_errors)
            self.logger.warning(error_msg, persian_msg)
    
    def _apply_environment_overrides(self, config: AppConfig):
        """Apply environment-specific configuration overrides"""
        
        if config.environment == "production":
            # Production overrides
            config.debug = False
            config.web.reload = False
            config.logging.level = "WARNING"
            config.features.enable_profiling = False
            config.features.mock_external_apis = False
            config.features.use_sample_data = False
            
        elif config.environment == "testing":
            # Testing overrides
            config.debug = True
            config.features.mock_external_apis = True
            config.features.use_sample_data = True
            config.database.url = "sqlite:///:memory:"
    
    def get_config(self) -> AppConfig:
        """Get current configuration (loads if not already loaded)"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def save_config(self, config: AppConfig = None):
        """Save current configuration to JSON file"""
        
        config = config or self._config
        if config is None:
            raise ValueError("No configuration to save")
        
        try:
            config_dict = asdict(config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
            self.logger.info(
                f"Configuration saved to {self.config_file}",
                f"تنظیمات در {self.config_file} ذخیره شد"
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to save configuration: {str(e)}",
                f"خطا در ذخیره تنظیمات: {str(e)}"
            )
            raise
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation (e.g., 'database.url')
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        
        config = self.get_config()
        
        try:
            value = config
            for key in key_path.split('.'):
                if hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return default
            return value
        except:
            return default
    
    def set_value(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        
        config = self.get_config()
        
        keys = key_path.split('.')
        target = config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if hasattr(target, key):
                target = getattr(target, key)
            else:
                raise ValueError(f"Invalid configuration path: {key_path}")
        
        # Set the final value
        final_key = keys[-1]
        if hasattr(target, final_key):
            setattr(target, final_key, value)
        else:
            raise ValueError(f"Invalid configuration key: {final_key}")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> AppConfig:
    """Get current application configuration"""
    return get_config_manager().get_config()

def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation"""
    return get_config_manager().get_value(key_path, default)