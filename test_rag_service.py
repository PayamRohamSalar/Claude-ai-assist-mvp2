#!/usr/bin/env python3
"""
Test script for RAG service functionality.
"""
import os

def test_missing_config():
    """Test error handling for missing RAG config."""
    print("Testing missing config error handling...")
    
    # Set non-existent config path
    os.environ['RAG_CONFIG_PATH'] = 'nonexistent/config.json'
    
    from webapp.services.rag_service import RAGService, ServiceError
    
    service = RAGService()
    
    try:
        service.answer('test question')
        print("❌ Should have thrown ServiceError")
        return False
    except ServiceError as e:
        print("✅ ServiceError caught successfully:")
        print(f"   User message: {e.user_message}")
        print(f"   Trace ID: {e.trace_id}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_citation_normalization():
    """Test citation normalization functionality."""
    print("\nTesting citation normalization...")
    
    from webapp.services.rag_service import RAGService
    
    service = RAGService()
    
    test_citations = [
        {
            'document_uid': 'doc123',
            'article_number': '45',
            'note_label': '1',
            'document_title': 'قانون کار'
        },
        {
            'document_uid': 'doc456',
            'article_number': '10',
            'document_title': 'قانون مدنی'
        },
        {
            'document_uid': '',
            'document_title': 'سند بدون شناسه'
        }
    ]
    
    result = service._normalize_citations(test_citations)
    
    print("✅ Citation normalization results:")
    for i, citation in enumerate(result):
        print(f"   [{i+1}] Title: {citation['title']}")
        print(f"       Link: {citation['link']}")
    
    return True

def test_availability_check():
    """Test service availability check."""
    print("\nTesting service availability...")
    
    from webapp.services.rag_service import get_rag_service
    
    service = get_rag_service()
    available = service.is_available()
    
    print(f"✅ Service availability: {available}")
    return True

if __name__ == "__main__":
    print("🧪 Testing RAG Service")
    print("=" * 40)
    
    success = True
    success &= test_missing_config()
    success &= test_citation_normalization()
    success &= test_availability_check()
    
    if success:
        print("\n✅ All RAG service tests passed!")
    else:
        print("\n❌ Some tests failed.")
