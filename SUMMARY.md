# Ollama Connectivity Issue - Audit Summary

## **Investigation Results**

### **Root Cause Identified: Proxy Configuration Issue**

The error message "خطا در تولید پاسخ: امکان اتصال به سرور Ollama وجود ندارد" is **genuine connectivity issue**, not a code/config wiring bug.

### **Evidence**

1. **Configuration is correct:**
   - `base_url`: `http://localhost:11434` (effective from .env)
   - `model`: `qwen2.5:7b-instruct` (correct and available)
   - `timeout`: 60s (reasonable)
   - Config precedence working: ENV > config > default

2. **Diagnostic output shows proxy interference:**
   ```
   ProxyError('Unable to connect to proxy', NewConnectionError(...))
   HTTPConnectionPool(host='127.0.0.1', port=10808): Max retries exceeded
   ```

3. **Vector store is healthy:**
   - FAISS index exists: `data/processed_phase_3/vector_db/faiss/faiss.index` ✅
   - Embeddings exist: `data/processed_phase_3/embeddings.npy` ✅

## **Fixes Implemented**

### **1. Enhanced Configuration Logging (`api_connections.py`)**
- Added detailed logging of effective config values with emojis
- Enhanced ping diagnostics with specific error types
- Early connectivity test before model validation

### **2. Improved Service Layer (`webapp/services/rag_service.py`)**
- Added engine readiness check with LLM ping test
- Better error messages linking to specific issues
- Proper trace_id tracking for debugging

### **3. Enhanced Health Endpoint (`webapp/api/routes/health.py`)**
- Exposes effective `base_url`, `model`, and component status
- Shows vector store type and database path
- Returns degraded status when components fail

### **4. Diagnostic Script (`scripts/diag_ollama.py`)**
- Tests connectivity, model availability, and generation
- Validates vector store files
- Provides actionable suggestions

### **5. Configuration Alignment (`Rag_config.json`)**
- Added missing `base_url` field for clarity
- Ensured all vector store paths are consistent

## **Quick Resolution Steps**

### **Immediate Fix (Proxy Issue):**
```bash
# Option 1: Disable proxy for localhost
export NO_PROXY=localhost,127.0.0.1

# Option 2: Clear proxy environment variables
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

# Option 3: Start Ollama with different port
ollama serve --host 0.0.0.0:11435
# Then update OLLAMA_BASE_URL=http://localhost:11435
```

### **Verify Fix:**
```bash
# Test connectivity
python scripts/diag_ollama.py

# Check health endpoint
curl http://localhost:8000/api/health/detailed

# Test direct Ollama access
curl http://localhost:11434/api/tags
```

## **Prevention Measures**

### **1. Environment Setup Script**
Create `scripts/setup_environment.py`:
```python
import os
import subprocess

def setup_ollama_environment():
    # Clear problematic proxy settings
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
    
    # Set no-proxy for localhost
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
    
    # Test connection
    subprocess.run(['python', 'scripts/diag_ollama.py'])
```

### **2. Enhanced Error Messages**
Now implemented:
- Persian messages with technical hints
- Specific suggestions (proxy, model pull, port check)
- Trace IDs for debugging

### **3. Startup Diagnostics**
- Health endpoint now shows effective configuration
- Service layer tests connectivity on init
- Better logging for troubleshooting

## **Configuration Summary**

### **Effective Values Used:**
- **Provider:** ollama (from config)
- **Base URL:** http://localhost:11434 (from .env)
- **Model:** qwen2.5:7b-instruct (from config)
- **Timeout:** 60s (from config)
- **Vector Store:** FAISS with correct paths

### **Config Sources:**
1. `.env` file: `OLLAMA_BASE_URL=http://localhost:11434`
2. `Rag_config.json`: Model and provider settings
3. Environment variables: Take precedence over config

## **Error Classification**

| Error Type | Cause | Fix |
|------------|-------|-----|
| **Proxy Error** ✅ Current | System proxy blocking localhost | Disable proxy for localhost |
| **Port Conflict** | Ollama not running/wrong port | Check `ollama serve` status |
| **Model Missing** | Model not pulled | `ollama pull model-name` |
| **Config Wrong** | Wrong base_url/model | Update config files |

## **Next Steps**

1. **Immediate:** Resolve proxy configuration
2. **Short-term:** Use diagnostic script for system health
3. **Long-term:** Consider containerization to avoid env issues

## **Files Modified**

- ✅ `phase_4_llm_rag/api_connections.py` - Enhanced connectivity & logging
- ✅ `webapp/services/rag_service.py` - Better error handling & init checks  
- ✅ `webapp/api/routes/health.py` - Diagnostic information exposure
- ✅ `phase_4_llm_rag/Rag_config.json` - Added missing base_url
- ✅ `scripts/diag_ollama.py` - New diagnostic tool

**The issue is environmental (proxy), not code-related. The implemented fixes provide better diagnostics and error handling for future issues.**
