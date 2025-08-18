# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_logger.py

"""
Test script for logger.py module
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_logger():
    """Test the logger module functionality"""
    
    print("🧪 Testing logger.py module...")
    
    try:
        # Import the logger module
        from shared_utils.logger import (
            get_logger, log_system_startup, log_system_shutdown,
            log_info, log_warning, log_error, LegalLogger
        )
        
        print("✅ Import successful")
        
        # Test system startup logging
        print("\n📝 Testing system startup...")
        log_system_startup()
        
        # Get main logger
        logger = get_logger("TestLogger", "DEBUG")
        print(f"✅ Logger created: {logger.name}")
        
        # Test basic logging levels
        print("\n📝 Testing basic logging levels...")
        logger.debug("Debug message", "پیام اشکال‌زدایی")
        logger.info("Info message", "پیام اطلاعاتی")
        logger.warning("Warning message", "پیام هشدار")
        logger.error("Error message", "پیام خطا")
        
        # Test convenience functions
        print("\n📝 Testing convenience functions...")
        log_info("Convenience info", "تابع راحت اطلاعات")
        log_warning("Convenience warning", "تابع راحت هشدار")
        log_error("Convenience error", "تابع راحت خطا")
        
        # Test specialized logging methods
        print("\n📝 Testing specialized logging...")
        
        # Document processing log
        logger.log_document_processing(
            document_name="قانون نمونه.pdf",
            status="completed",
            details={"pages": 25, "articles": 45}
        )
        
        # Search query log
        logger.log_search_query(
            query="ماده 15 قانون پژوهش",
            results_count=12,
            processing_time=0.85
        )
        
        # LLM interaction log
        logger.log_llm_interaction(
            model_name="qwen2.5:7b-instruct",
            prompt_length=150,
            response_length=300,
            success=True
        )
        
        # Test with extra data
        print("\n📝 Testing logging with extra data...")
        logger.info(
            "Processing completed",
            "پردازش تکمیل شد",
            user_id="test_user",
            operation="document_parse",
            duration=45.2
        )
        
        # Test system shutdown
        print("\n📝 Testing system shutdown...")
        log_system_shutdown()
        
        # Check if log files were created
        print("\n📁 Checking created log files...")
        from shared_utils.constants import LOGS_DIR
        
        log_files = list(LOGS_DIR.glob("*.log")) + list(LOGS_DIR.glob("*.jsonl"))
        
        if log_files:
            print("✅ Log files created:")
            for log_file in log_files:
                size = log_file.stat().st_size
                print(f"  📄 {log_file.name} ({size} bytes)")
        else:
            print("⚠️ No log files found")
        
        print("\n🎉 All logger tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_persian_formatting():
    """Test Persian date and digit formatting"""
    
    print("\n🇮🇷 Testing Persian formatting...")
    
    try:
        from shared_utils.logger import PersianFormatter, get_logger
        from shared_utils.constants import english_to_persian_digits
        import logging
        
        # Test Persian digit conversion
        english_text = "2024/01/15 14:30:25"
        persian_text = english_to_persian_digits(english_text)
        print(f"🔢 English: {english_text}")
        print(f"🔢 Persian: {persian_text}")
        
        # Test formatter
        formatter = PersianFormatter()
        
        # Create a test log record
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.persian_msg = "پیام تست"
        
        formatted = formatter.format(record)
        print(f"📝 Formatted log: {formatted}")
        
        print("✅ Persian formatting test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Persian formatting test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Legal Assistant AI - Logger Test Suite")
    print("=" * 60)
    
    success1 = test_logger()
    success2 = test_persian_formatting()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Logger is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)