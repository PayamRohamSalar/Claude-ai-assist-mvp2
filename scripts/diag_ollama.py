#!/usr/bin/env python3
"""
Ollama Diagnostic Script for Smart Legal Assistant
Reads Rag_config.json, tests connectivity, and validates model availability.
"""

import json
import sys
import requests
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "phase_4_llm_rag/Rag_config.json") -> Dict[str, Any]:
    """Load RAG configuration from JSON file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            return {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"✅ Config loaded from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return {}

def get_effective_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Get effective Ollama configuration with environment precedence."""
    import os
    
    def _pick(env_key: str, cfg_dict: dict, cfg_key: str, default=None):
        """Helper for configuration precedence: env > config > default."""
        v = os.getenv(env_key)
        if v not in (None, ""):
            return v
        if cfg_dict and cfg_key in cfg_dict and cfg_dict[cfg_key] not in (None, ""):
            return cfg_dict[cfg_key]
        return default
    
    cfg = config.get("llm", {})
    
    effective = {
        "provider": _pick("LLM_PROVIDER", cfg, "provider", "ollama"),
        "base_url": _pick("OLLAMA_BASE_URL", cfg, "base_url", "http://localhost:11434"),
        "timeout": int(_pick("OLLAMA_TIMEOUT", cfg, "timeout_s", 60)),
        "model": _pick("OLLAMA_MODEL_NAME", cfg, "model", "unknown"),
        "backup_model": _pick("OLLAMA_BACKUP_MODEL", cfg, "backup_model", None)
    }
    
    return effective

def test_ollama_ping(base_url: str, timeout: int = 10) -> bool:
    """Test if Ollama server is accessible."""
    try:
        logger.info(f"🔍 Pinging Ollama server at {base_url}")
        
        response = requests.get(
            f"{base_url}/api/tags",
            timeout=timeout
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Ping successful: HTTP {response.status_code}")
            return True
        else:
            logger.error(f"❌ Ping failed: HTTP {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        logger.error(f"❌ Ping failed: Connection timeout after {timeout}s")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Ping failed: Connection error - {e}")
        return False
    except requests.exceptions.Timeout:
        logger.error(f"❌ Ping failed: Request timeout after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"❌ Ping failed: {type(e).__name__}: {e}")
        return False

def get_available_models(base_url: str, timeout: int = 30) -> list:
    """Get list of available models from Ollama."""
    try:
        logger.info(f"📋 Fetching available models from {base_url}/api/tags")
        
        response = requests.get(
            f"{base_url}/api/tags",
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            models = []
            if "models" in data:
                for model_info in data["models"]:
                    if "name" in model_info:
                        models.append(model_info["name"].strip())
            
            logger.info(f"✅ Found {len(models)} models")
            return models
        else:
            logger.error(f"❌ Failed to get models: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Failed to get models: {e}")
        return []

def test_model_generate(base_url: str, model: str, timeout: int = 60) -> bool:
    """Test if model can generate text."""
    try:
        logger.info(f"🧪 Testing text generation with model: {model}")
        
        payload = {
            "model": model,
            "prompt": "سلام! این یک تست ساده است.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50
            }
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                generated = result["response"]
                logger.info(f"✅ Generation successful: {len(generated)} characters")
                logger.info(f"   Sample: {generated[:100]}...")
                return True
            else:
                logger.error(f"❌ Generation failed: No response in result")
                return False
        else:
            logger.error(f"❌ Generation failed: HTTP {response.status_code}")
            logger.error(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ Generation failed: Timeout after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False

def check_vector_store(config: Dict[str, Any]) -> bool:
    """Check if vector store files exist."""
    try:
        vector_config = config.get("vector_store", {})
        if not vector_config:
            logger.warning("⚠️ No vector_store config found")
            return False
        
        vs_type = vector_config.get("type", "unknown")
        logger.info(f"📁 Checking vector store: {vs_type}")
        
        if vs_type == "faiss":
            index_path = vector_config.get("index_path", "")
            embeddings_path = vector_config.get("embeddings_path", "")
            
            index_exists = Path(index_path).exists() if index_path else False
            embeddings_exist = Path(embeddings_path).exists() if embeddings_path else False
            
            logger.info(f"   Index file: {index_path} {'✅' if index_exists else '❌'}")
            logger.info(f"   Embeddings: {embeddings_path} {'✅' if embeddings_exist else '❌'}")
            
            return index_exists and embeddings_exist
        else:
            logger.warning(f"⚠️ Vector store type '{vs_type}' not supported in diagnostic")
            return False
            
    except Exception as e:
        logger.error(f"❌ Vector store check failed: {e}")
        return False

def main():
    """Main diagnostic function."""
    print("🚀 Ollama Diagnostic Script for Smart Legal Assistant")
    print("=" * 60)
    
    exit_code = 0
    
    # 1. Load configuration
    config = load_config()
    if not config:
        print("❌ FAILED: Could not load configuration")
        return 1
    
    # 2. Get effective configuration
    effective = get_effective_config(config)
    print(f"\n📊 Effective Configuration:")
    for key, value in effective.items():
        print(f"   • {key}: {value}")
    
    # 3. Test Ollama connectivity
    print(f"\n🔍 Testing Ollama Connectivity:")
    base_url = effective["base_url"]
    timeout = effective["timeout"]
    
    if not test_ollama_ping(base_url, min(timeout, 10)):
        print("❌ FAILED: Ollama server not accessible")
        exit_code = 1
    
    # 4. Get available models
    print(f"\n📋 Checking Available Models:")
    available_models = get_available_models(base_url, timeout)
    if available_models:
        print(f"   Found {len(available_models)} models:")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model}")
    else:
        print("❌ FAILED: No models available")
        exit_code = 1
    
    # 5. Test configured model
    print(f"\n🧪 Testing Configured Model:")
    requested_model = effective["model"]
    if requested_model == "unknown":
        print("❌ FAILED: No model configured")
        exit_code = 1
    elif requested_model not in available_models:
        print(f"❌ FAILED: Model '{requested_model}' not available")
        print(f"   💡 Suggestion: Run 'ollama pull {requested_model}'")
        exit_code = 1
    else:
        if test_model_generate(base_url, requested_model, timeout):
            print(f"✅ Model '{requested_model}' working correctly")
        else:
            print(f"❌ FAILED: Model '{requested_model}' cannot generate text")
            exit_code = 1
    
    # 6. Check vector store
    print(f"\n📁 Checking Vector Store:")
    if check_vector_store(config):
        print("✅ Vector store files found")
    else:
        print("❌ FAILED: Vector store files missing")
        print("   💡 Suggestion: Run phases 1-3 to generate vector store")
        exit_code = 1
    
    # 7. Summary
    print(f"\n📋 Diagnostic Summary:")
    if exit_code == 0:
        print("✅ All checks passed - System should work correctly")
    else:
        print("❌ Some checks failed - System may not work properly")
        print("   Check the errors above for specific issues")
    
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Diagnostic failed with error: {e}")
        sys.exit(1)
