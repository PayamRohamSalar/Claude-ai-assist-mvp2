import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from phase_4_llm_rag.api_connections import make_llm_client

def main():
    # minimal inline config; .env will override through make_llm_client()
    config = {
        "llm": {
            "provider": "ollama",
            # model/base_url/timeout will be overridden by .env if set
        }
    }
    client = make_llm_client(config)
    print("Ping:", client.ping())
    print("Models:", client.list_models()[:10])
    out = client.generate("سلام! این یک تست اتصال است.", temperature=0.1, max_tokens=128)
    print("Sample output:", out[:400])

if __name__ == "__main__":
    main()