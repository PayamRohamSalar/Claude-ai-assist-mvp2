#!/usr/bin/env python3
"""
Ollama Diagnostic Script for Smart Legal Assistant
Reads Rag_config.json, tests connectivity, and validates model availability.
"""

import json
import sys
import requests
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable proxies for local Ollama access to avoid corporate/local proxy interference
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')
os.environ.setdefault('no_proxy', 'localhost,127.0.0.1')

# Use a single session and ignore environment proxies entirely
SESSION = requests.Session()
SESSION.trust_env = False

def load_config(config_path: str = "phase_4_llm_rag/Rag_config.json") -> Dict[str, Any]:
    """Load RAG configuration from JSON file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"❌ فایل تنظیمات یافت نشد: {config_path}")
            return {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"✅ تنظیمات از مسیر زیر بارگذاری شد: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری تنظیمات: {e}")
        return {}

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
    
    # Get raw base_url and apply adaptive detection
    raw_base_url = _pick("OLLAMA_BASE_URL", cfg, "base_url", "http://localhost:11434")
    adaptive_base_url = _adaptive_base_url(raw_base_url)
    
    effective = {
        "provider": _pick("LLM_PROVIDER", cfg, "provider", "ollama"),
        "base_url": adaptive_base_url,
        "timeout": int(_pick("OLLAMA_TIMEOUT", cfg, "timeout_s", 60)),
        "model": _pick("OLLAMA_MODEL_NAME", cfg, "model", "unknown"),
        "backup_model": _pick("OLLAMA_BACKUP_MODEL", cfg, "backup_model", None)
    }
    
    return effective

def test_ollama_ping(base_url: str, timeout: int = 10) -> bool:
    """بررسی دسترسی‌پذیری سرور Ollama."""
    try:
        logger.info(f"🔍 ارسال پینگ به سرور Ollama در آدرس: {base_url}")
        
        response = SESSION.get(
            f"{base_url}/api/tags",
            timeout=timeout
        )
        
        if response.status_code == 200:
            logger.info(f"✅ پینگ موفق بود: کد {response.status_code}")
            return True
        else:
            logger.error(f"❌ پینگ ناموفق: کد {response.status_code}")
            logger.error(f"   پاسخ: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        logger.error(f"❌ پینگ ناموفق: اتمام زمان اتصال پس از {timeout} ثانیه")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ پینگ ناموفق: خطای اتصال - {e}")
        return False
    except requests.exceptions.Timeout:
        logger.error(f"❌ پینگ ناموفق: اتمام زمان درخواست پس از {timeout} ثانیه")
        return False
    except Exception as e:
        logger.error(f"❌ پینگ ناموفق: {type(e).__name__}: {e}")
        return False

def get_available_models(base_url: str, timeout: int = 30) -> list:
    """دریافت فهرست مدل‌های موجود از Ollama."""
    try:
        logger.info(f"📋 در حال دریافت فهرست مدل‌ها از {base_url}/api/tags")
        
        response = SESSION.get(
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
            
            logger.info(f"✅ تعداد مدل‌ها: {len(models)}")
            return models
        else:
            logger.error(f"❌ خطا در دریافت مدل‌ها: کد {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"❌ خطا در دریافت مدل‌ها: {e}")
        return []

def test_model_generate(base_url: str, model: str, timeout: int = 60) -> bool:
    """تست تولید متن توسط مدل."""
    try:
        logger.info(f"🧪 تست تولید متن با مدل: {model}")
        
        payload = {
            "model": model,
            "prompt": "سلام",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1
            }
        }
        
        response = SESSION.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                generated = result["response"]
                logger.info(f"✅ تولید موفق بود (طول پاسخ: {len(generated)} کاراکتر)")
                logger.info(f"   نمونه پاسخ: {generated[:100]}...")
                return True
            else:
                logger.error(f"❌ تولید ناموفق: کلید 'response' در نتیجه وجود ندارد")
                return False
        else:
            logger.error(f"❌ تولید ناموفق: کد {response.status_code}")
            logger.error(f"   پاسخ: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ تولید ناموفق: اتمام زمان پس از {timeout} ثانیه")
        return False
    except Exception as e:
        logger.error(f"❌ تولید ناموفق: {e}")
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
    """اجرای تست‌های تشخیصی اتصال به Ollama (خروجی فارسی)."""
    print("🧪 اسکریپت عیب‌یابی Ollama برای دستیار حقوقی هوشمند")
    print("=" * 60)

    exit_code = 0

    # 1) بارگذاری تنظیمات
    config = load_config()
    if not config:
        print("❌ خطا: امکان بارگذاری تنظیمات وجود ندارد")
        return 1

    # 2) اعمال تقدم ENV > config
    effective = get_effective_config(config)
    base_url = effective["base_url"]
    timeout = effective["timeout"]
    model = effective["model"]

    print("\n📊 تنظیمات موثر:")
    print(f"   آدرس سرور (base_url): {base_url}")
    print(f"   نام مدل (model): {model}")

    # 3) وضعیت پینگ
    print("\n🔍 وضعیت پینگ:")
    ping_ok = test_ollama_ping(base_url, min(timeout, 10))
    print("   نتیجه پینگ:", "موفق" if ping_ok else "ناموفق")
    if not ping_ok:
        exit_code = 1

    # 4) تعداد مدل‌ها از /api/tags
    available_models = get_available_models(base_url, timeout)
    print("\n📋 اطلاعات مدل‌ها:")
    print(f"   تعداد مدل‌های موجود: {len(available_models)}")

    # 5) تست تولید ۱ توکن با متن 'سلام'
    print("\n🧪 تست تولید تک‌توکن:")
    if model in (None, "", "unknown"):
        print("   ❌ خطا: هیچ مدلی در تنظیمات مشخص نشده است (OLLAMA_MODEL_NAME یا llm.model)")
        exit_code = 1
    else:
        can_generate = test_model_generate(base_url, model, timeout)
        print("   نتیجه تولید:", "موفق" if can_generate else "ناموفق")
        if not can_generate:
            exit_code = 1

    # 6) جمع‌بندی
    print("\n📋 نتیجه نهایی:")
    if exit_code == 0:
        print("✅ همه بررسی‌ها با موفقیت انجام شد.")
    else:
        print("❌ برخی از بررسی‌ها ناموفق بودند. لطفاً خطاهای فوق را بررسی کنید.")

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
