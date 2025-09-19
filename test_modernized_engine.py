#!/usr/bin/env python3
"""
Test script for the modernized LegalRAGEngine
"""

import sys
sys.path.append('.')

def test_modernized_engine():
    """Test the modernized LegalRAGEngine with new config schema."""
    try:
        from rag_engine import LegalRAGEngine
        
        print("🧪 Testing modernized LegalRAGEngine...")
        
        # Test with the new config schema
        engine = LegalRAGEngine('phase_4_llm_rag/Rag_config.json')
        print('✅ Successfully initialized LegalRAGEngine with new config schema')
        
        # Test basic functionality
        stats = engine.get_stats()
        print(f'✅ Engine stats: {stats}')
        
        # Test retrieval (without LLM generation)
        chunks = engine.retrieve('مجازات سرقت چیست؟', top_k=3)
        print(f'✅ Retrieved {len(chunks)} chunks successfully')
        
        # Test prompt building
        if chunks:
            prompt = engine.build_prompt('مجازات سرقت چیست؟', chunks, 'default')
            print(f'✅ Built prompt successfully (length: {len(prompt)})')
        
        engine.close()
        print('✅ All tests passed!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_modernized_engine()
    sys.exit(0 if success else 1)
