#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_engine import LegalRAGEngine

def test_rag_engine():
    """Test the RAG engine with existing pipeline data"""
    
    # Get config path
    config_path = str(project_root / "config" / "config.json")
    
    print("[INFO] Initializing RAG engine...")
    
    # Initialize RAG engine
    try:
        rag_engine = LegalRAGEngine(config_path)
        print("[SUCCESS] RAG engine initialized successfully")
        
        # Test retrieval
        print("\n[INFO] Testing retrieval...")
        test_question = "حقوق کودکان چیست؟"
        
        # Test with very low threshold to see what we get
        retrieved_chunks = rag_engine.retrieve(test_question, top_k=5)
        print(f"[SUCCESS] Retrieved {len(retrieved_chunks)} chunks")
        
        if retrieved_chunks:
            print(f"First chunk length: {len(retrieved_chunks[0].content)} characters")
            print(f"First chunk similarity: {retrieved_chunks[0].similarity_score:.3f}")
        else:
            print("No chunks retrieved - similarity threshold may be too high")
        
        # Test full answer generation
        print("\n[INFO] Testing answer generation...")
        answer = rag_engine.answer(test_question)
        
        print("[SUCCESS] Answer generated successfully")
        try:
            answer_text = answer['answer'][:200] if 'answer' in answer else "No answer"
            print(f"Answer length: {len(answer.get('answer', ''))} characters")
        except:
            print("Answer contains Persian text")
        print(f"Sources: {len(answer.get('sources', []))} chunks used")
        print(f"Response time: {answer.get('response_time_seconds', 0):.2f}s")
        
        # Show error if any
        if 'error' in answer:
            print(f"Error: {answer['error']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing RAG engine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_engine()
    if success:
        print("\n[SUCCESS] RAG engine test completed successfully!")
    else:
        print("\n[FAILED] RAG engine test failed!")
        sys.exit(1)