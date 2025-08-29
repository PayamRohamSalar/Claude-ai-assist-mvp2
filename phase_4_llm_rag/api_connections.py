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
        
        logger.info(f"Initialized OllamaClient with model: {model}, base_url: {base_url}")
    
    def ping(self) -> bool:
        """Check if Ollama server is accessible.
        
        Returns:
            True if server responds to /api/tags, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=min(self.timeout, 10)  # Use shorter timeout for ping
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server ping failed: {e}")
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
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            logger.debug(f"Sending request to Ollama API: {endpoint}")
            logger.debug(f"Payload: model={self.model}, temperature={temperature}, max_tokens={max_tokens}")
            
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
        base_url = _pick("OLLAMA_BASE_URL", cfg, "base_url", "http://localhost:11434")
        timeout = int(_pick("OLLAMA_TIMEOUT", cfg, "timeout_s", 60))
        model = _pick("OLLAMA_MODEL_NAME", cfg, "model", None)
        backup = _pick("OLLAMA_BACKUP_MODEL", cfg, "backup_model", None)
        
        if model is None:
            raise KeyError("No primary LLM model configured (env OLLAMA_MODEL_NAME or llm.model).")
        
        # Normalize model names
        model = _normalize_model_tag(model)
        backup = _normalize_model_tag(backup) if backup else None
        
        logger.info(f"Creating Ollama client for model: {model}")
        
        # Create provisional client
        client = OllamaClient(model=model, base_url=base_url, timeout=timeout)
        
        # Validate model availability
        try:
            available_models = client.list_models()
            available = set(available_models)
            
            logger.debug(f"Available models: {available_models}")
            
            if model not in available:
                if backup and backup in available:
                    logger.warning(
                        f"Primary model '{model}' not found. Switching to backup model '{backup}'."
                    )
                    client = OllamaClient(model=backup, base_url=base_url, timeout=timeout)
                else:
                    error_msg = f"Requested model '{model}' not found on Ollama server"
                    if backup:
                        error_msg += f" and backup model '{backup}' is also not available"
                    else:
                        error_msg += " and no backup model configured"
                    
                    logger.error(error_msg)
                    raise ValueError(f"مدل '{model}' در سرور Ollama یافت نشد و مدل پشتیبان معتبری موجود نیست.")
            else:
                logger.info(f"Model '{model}' verified as available on Ollama server")
                
        except Exception as e:
            if "not found" in str(e).lower() or "available" in str(e).lower():
                # Re-raise model availability errors
                raise
            else:
                # Log other errors but don't fail - server might still work
                logger.warning(f"Could not verify model availability: {e}")
        
        return client
    
    else:
        error_msg = f"Unsupported LLM provider: {provider}. Supported providers: 'ollama', 'fake'."
        logger.error(error_msg)
        raise NotImplementedError(f"ارائه‌دهنده '{provider}' پشتیبانی نمی‌شود. ارائه‌دهندگان پشتیبانی‌شده: ollama, fake")


# Alias for backwards compatibility with rag_engine.py
def get_llm_client(config: Dict[str, Any]):
    """Alias for make_llm_client for backwards compatibility."""
    return make_llm_client(config)