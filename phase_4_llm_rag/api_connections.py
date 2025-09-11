import requests
import json
import logging
import os
import re
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Configure logging
logger = logging.getLogger(__name__)


def _pick(env_key: str, cfg: dict, cfg_key: str, default=None):
    """Helper for configuration precedence: env > config > default.
    
    Args:
        env_key: Environment variable key
        cfg: Configuration dictionary
        cfg_key: Key in configuration dictionary
        default: Default value if neither env nor config have the key
        
    Returns:
        Value with precedence: environment > config > default
    """
    v = os.getenv(env_key)
    if v not in (None, ""):
        return v
    if cfg and cfg_key in cfg and cfg[cfg_key] not in (None, ""):
        return cfg[cfg_key]
    return default


def _pick_with_source(env_key: str, cfg: dict, cfg_key: str, default=None):
    """Pick a value with precedence and return its source as well.
    
    Precedence: ENV > config > default
    
    Returns:
        (value, source) where source in {"ENV", "config", "default"}
    """
    v = os.getenv(env_key)
    if v not in (None, ""):
        return v, "ENV"
    if cfg and cfg_key in cfg and cfg[cfg_key] not in (None, ""):
        return cfg[cfg_key], "config"
    return default, "default"


def _detect_runtime_environment():
    """Detect the runtime environment (Windows, WSL, Docker) and suggest appropriate base_url."""
    import platform
    import os
    
    # Check for Docker environment
    if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
        return "docker", "http://host.docker.internal:11434"
    
    # Check for WSL environment 
    if 'microsoft' in platform.uname().release.lower() or os.getenv('WSL_DISTRO_NAME'):
        return "wsl", "http://localhost:11434"
        
    # Check for Windows environment
    if platform.system() == "Windows":
        return "windows", "http://localhost:11434"
    
    # Default for Linux/macOS
    return "unix", "http://localhost:11434"


def _adaptive_base_url(configured_base_url: str) -> str:
    """Adapt base_url based on runtime environment if using default localhost."""
    # Only adapt if using default localhost URLs
    if configured_base_url not in ["http://localhost:11434", "http://127.0.0.1:11434"]:
        return configured_base_url
    
    env_type, suggested_url = _detect_runtime_environment()
    
    # For Docker, use host.docker.internal to reach host Ollama
    if env_type == "docker":
        logger.info(f"🐳 Docker environment detected - adapting base_url to: {suggested_url}")
        return suggested_url
    
    # For other environments, keep localhost but ensure it's the right format
    return configured_base_url


def _normalize_model_tag(model: str) -> str:
    """Normalize model tags by stripping whitespace.
    
    Args:
        model: Model name/tag
        
    Returns:
        Normalized model name
    """
    return model.strip() if model else ""


class FakeLLMClient:
    """Fake LLM client for testing purposes."""
    
    def __init__(self, model: str = "fake-model", **kwargs) -> None:
        """Initialize fake LLM client.
        
        Args:
            model: Model name (ignored for fake client)
            **kwargs: Additional arguments (ignored)
        """
        self.model = model
        logger.info(f"Initialized FakeLLMClient with model: {model}")
    
    def ping(self) -> bool:
        """Always returns True for fake client."""
        return True
    
    def list_models(self) -> List[str]:
        """Returns a fake list of models."""
        return ["fake-model", "test-model"]
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 4096, **kwargs) -> str:
        """Generate a fake response based on simple rules.
        
        Args:
            prompt: Input prompt
            temperature: Temperature (ignored)
            max_tokens: Max tokens (ignored)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            A rule-based fake response
        """
        prompt_lower = prompt.lower()
        
        # Rule-based responses for common patterns
        if 'سؤال' in prompt or 'question' in prompt_lower:
            if 'ماده' in prompt or 'article' in prompt_lower:
                return (
                    "بر اساس اسناد بازیابی‌شده، ماده ۱۲ قانون مربوطه بیان می‌کند که... "
                    "[این پاسخ توسط FakeLLMClient تولید شده است.] "
                    "برای اطلاعات دقیق‌تر، لطفاً به متن کامل قانون مراجعه کنید."
                )
            elif 'تبصره' in prompt or 'note' in prompt_lower:
                return (
                    "تبصره ۱ در این خصوص مقرر می‌دارد که... "
                    "[این پاسخ توسط FakeLLMClient تولید شده است.] "
                    "جهت مطالعه کامل، به متن اصلی مراجعه شود."
                )
            else:
                return (
                    "بر اساس بررسی اسناد موجود، پاسخ سؤال شما به شرح زیر است: "
                    "[این یک پاسخ آزمایشی از FakeLLMClient است.] "
                    "لطفاً برای دریافت پاسخ دقیق از مدل واقعی استفاده کنید."
                )
        else:
            # Echo the prompt with a prefix
            return f"[FakeLLMClient Echo] {prompt[:200]}{'...' if len(prompt) > 200 else ''}"


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 60) -> None:
        """Initialize Ollama client.
        
        Args:
            model: Name of the model to use (e.g., "llama2", "mistral")
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Disable proxy for local Ollama connections to avoid corporate proxy interference
        if 'localhost' in base_url or '127.0.0.1' in base_url:
            self.session.trust_env = False
            # Also set NO_PROXY environment variable for this session
            import os
            os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')
            os.environ.setdefault('no_proxy', 'localhost,127.0.0.1')
        
        self.last_error = None  # populated by ping() on failure
        
        logger.info(f"Initialized OllamaClient with model: {model}, base_url: {base_url}")
    
    def ping(self) -> bool:
        """Check if Ollama server is accessible.
        
        Returns:
            True if server responds to /api/tags, False otherwise
        """
        try:
            ping_timeout = min(self.timeout, 10)  # Use shorter timeout for ping
            logger.debug(f"Pinging Ollama server at {self.base_url} (timeout: {ping_timeout}s)")
            
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=ping_timeout
            )
            
            if response.status_code == 200:
                logger.debug(f"Ping successful: HTTP {response.status_code}")
                self.last_error = None
                return True
            else:
                logger.warning(f"Ping failed: HTTP {response.status_code} - {response.text[:100]}")
                self.last_error = requests.exceptions.RequestException(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return False
                
        except requests.exceptions.ConnectTimeout as e:
            logger.warning(f"Ping failed: Connection timeout after {ping_timeout}s to {self.base_url}")
            self.last_error = e
            return False
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Ping failed: Connection error to {self.base_url} - {e}")
            self.last_error = e
            return False
        except requests.exceptions.Timeout as e:
            logger.warning(f"Ping failed: Request timeout after {ping_timeout}s")
            self.last_error = e
            return False
        except Exception as e:
            logger.warning(f"Ping failed: Unexpected error - {type(e).__name__}: {e}")
            self.last_error = e
            return False
    
    def list_models(self) -> List[str]:
        """List available models on the Ollama server.
        
        Returns:
            List of available model names/tags, empty list on error
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                models = []
                if "models" in data:
                    for model_info in data["models"]:
                        if "name" in model_info:
                            models.append(_normalize_model_tag(model_info["name"]))
                logger.debug(f"Found {len(models)} models on Ollama server")
                return models
            else:
                logger.warning(f"Failed to list models: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 4096, **kwargs) -> str:
        """Generate text using Ollama API.
        
        Args:
            prompt: Input prompt for text generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Generated text response
            
        Raises:
            requests.exceptions.RequestException: On HTTP errors or timeouts
            ValueError: On invalid API response format
        """
        endpoint = f"{self.base_url}/api/generate"
        # Use effective defaults from client if caller used function defaults
        effective_temperature = (
            getattr(self, "default_temperature", temperature)
            if temperature == 0.1 else temperature
        )
        effective_max_tokens = (
            getattr(self, "default_max_tokens", max_tokens)
            if max_tokens == 4096 else max_tokens
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": effective_temperature,
                "num_predict": effective_max_tokens
            }
        }
        
        try:
            logger.debug(f"Sending request to Ollama API: {endpoint}")
            logger.debug(f"Payload: model={self.model}, temperature={effective_temperature}, max_tokens={effective_max_tokens}")
            
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"Ollama API returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise requests.exceptions.HTTPError(f"خطا در اتصال به سرور Ollama: {response.status_code}")
            
            # Parse JSON response
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise ValueError("پاسخ نامعتبر از سرور Ollama")
            
            # Extract generated text
            if "response" in result:
                generated_text = result["response"]
                logger.debug(f"Generated {len(generated_text)} characters")
                return generated_text
            elif "message" in result:
                # Handle error messages from Ollama
                error_msg = result.get("message", "Unknown error")
                logger.error(f"Ollama API error: {error_msg}")
                raise ValueError(f"خطا از سرور Ollama: {error_msg}")
            else:
                logger.error(f"Unexpected response format: {result}")
                raise ValueError("فرمت پاسخ نامعتبر از سرور Ollama")
                
        except requests.exceptions.Timeout:
            error_msg = f"Request to Ollama API timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise requests.exceptions.Timeout("اتصال به سرور Ollama قطع شد (timeout)")
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to Ollama API at {self.base_url}: {e}"
            logger.error(error_msg)
            raise requests.exceptions.ConnectionError("امکان اتصال به سرور Ollama وجود ندارد")
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Request to Ollama API failed: {e}"
            logger.error(error_msg)
            raise requests.exceptions.RequestException(f"خطا در ارسال درخواست به Ollama: {str(e)}")


def make_llm_client(config: Dict[str, Any]):
    """Factory function to create LLM client with environment variable precedence.
    
    Configuration precedence: environment variables > config file > defaults
    Environment variables:
        - LLM_PROVIDER: LLM provider (ollama, fake)
        - OLLAMA_BASE_URL: Ollama server URL
        - OLLAMA_TIMEOUT: Request timeout in seconds
        - OLLAMA_MODEL_NAME: Primary model name
        - OLLAMA_BACKUP_MODEL: Backup model name
    
    Args:
        config: Configuration dictionary containing LLM settings
        
    Returns:
        LLM client instance with model validation
        
    Raises:
        NotImplementedError: If provider is not supported
        KeyError: If no primary model is configured (for non-fake providers)
        ValueError: If requested model is not available and no valid backup
        ConnectionError: If Ollama server is not accessible
    """
    cfg = config.get("llm", {})
    
    # Resolve provider with precedence
    provider = _pick("LLM_PROVIDER", cfg, "provider", "ollama").lower()
    
    # Handle fake provider for testing
    if provider == "fake":
        logger.info("Creating fake LLM client for testing")
        model = _pick("OLLAMA_MODEL_NAME", cfg, "model", "fake-model")
        return FakeLLMClient(model=model)
    
    # Handle Ollama provider
    elif provider == "ollama":
        # Resolve Ollama settings with precedence (env > cfg > defaults)
        raw_base_url, base_url_source = _pick_with_source("OLLAMA_BASE_URL", cfg, "base_url", "http://localhost:11434")
        
        # Apply adaptive base_url based on runtime environment
        base_url = _adaptive_base_url(raw_base_url)
        timeout_str, timeout_source = _pick_with_source("OLLAMA_TIMEOUT", cfg, "timeout_s", 60)
        timeout = int(timeout_str)
        model, model_source = _pick_with_source("OLLAMA_MODEL_NAME", cfg, "model", None)
        backup = _pick("OLLAMA_BACKUP_MODEL", cfg, "backup_model", None)
        temperature = cfg.get("temperature", 0.1)
        max_tokens = cfg.get("max_tokens", 4096)
        
        # Log effective configuration values (source focuses on base_url per requirement)
        logger.info(
            f"Using LLM base_url={base_url}, model={model}, timeout={timeout}s (source: {base_url_source})"
        )
        
        if model is None:
            raise KeyError("No primary LLM model configured (env OLLAMA_MODEL_NAME or llm.model).")
        
        # Normalize model names
        model = _normalize_model_tag(model)
        backup = _normalize_model_tag(backup) if backup else None
        
        logger.info(f"🔧 Creating Ollama client for model: {model}")
        
        # Create provisional client
        client = OllamaClient(model=model, base_url=base_url, timeout=timeout)
        # Store effective generation defaults on client for later use
        try:
            client.default_temperature = float(temperature)
        except Exception:
            client.default_temperature = 0.1
        try:
            client.default_max_tokens = int(max_tokens)
        except Exception:
            client.default_max_tokens = 4096
        
        # Test connectivity BEFORE listing models
        logger.info(f"🔍 Testing connectivity to Ollama server at {base_url}")
        if not client.ping():
            err = getattr(client, "last_error", None)
            err_type = type(err).__name__ if err else "UnknownError"
            logger.error(
                f"Ollama ping failed: base_url={base_url}, timeout={timeout}s, error_type={err_type}, error={err}"
            )
            # Raise Persian-friendly message for end users
            raise ConnectionError("امکان اتصال به سرور Ollama وجود ندارد")
        
        logger.info(f"✅ Ollama server ping successful")
        
        # Validate model availability
        try:
            logger.info(f"📋 Listing available models...")
            available_models = client.list_models()
            available = set(available_models)
            
            logger.info(
                f"📋 Found {len(available_models)} models: {', '.join(available_models) if len(available_models) <= 20 else ', '.join(available_models[:20]) + '...'}"
            )
            logger.info(f"Requested model: {model}")
            
            if model not in available:
                logger.warning(f"⚠️ Requested model '{model}' not found in available models")
                if backup and backup in available:
                    logger.warning(
                        f"🔄 Primary model '{model}' not found. Switching to backup model '{backup}'."
                    )
                    client = OllamaClient(model=backup, base_url=base_url, timeout=timeout)
                    # Preserve generation defaults
                    client.default_temperature = float(temperature)
                    client.default_max_tokens = int(max_tokens)
                    logger.info(f"✅ Successfully switched to backup model: {backup}")
                else:
                    logger.error(f"❌ Model '{model}' not available")
                    logger.info(f"💡 Suggestion: Run 'ollama pull {model}' or check 'ollama list' for exact tags")
                    
                    error_msg = f"مدل '{model}' در سرور Ollama یافت نشد. برای نصب مدل، دستور زیر را اجرا کنید: ollama pull {model}"
                    raise ValueError(error_msg)
            else:
                logger.info(f"✅ Model '{model}' verified as available on Ollama server")
                
        except Exception as e:
            if "not found" in str(e).lower() or "available" in str(e).lower():
                # Re-raise model availability errors
                raise
            else:
                # Log other errors but don't fail - server might still work
                logger.warning(f"⚠️ Could not verify model availability: {e}")
        
        return client
    
    else:
        error_msg = f"Unsupported LLM provider: {provider}. Supported providers: 'ollama', 'fake'."
        logger.error(error_msg)
        raise NotImplementedError(f"ارائه‌دهنده '{provider}' پشتیبانی نمی‌شود. ارائه‌دهندگان پشتیبانی‌شده: ollama, fake")


# Alias for backwards compatibility with rag_engine.py
def get_llm_client(config: Dict[str, Any]):
    """Alias for make_llm_client for backwards compatibility."""
    return make_llm_client(config)