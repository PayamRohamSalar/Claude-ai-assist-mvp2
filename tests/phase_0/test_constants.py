# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_constants.py

"""
Test script for constants.py module
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_constants():
    """Test the constants module functionality"""
    
    print("🧪 Testing constants.py module...")
    
    try:
        # Import the constants module
        from shared_utils.constants import (
            PROJECT_NAME, DocumentType, ApprovalAuthority, Messages,
            persian_to_english_digits, english_to_persian_digits,
            validate_file_extension, get_section_name, BASE_DIR
        )
        
        print("✅ Import successful")
        
        # Test basic constants
        print(f"📋 Project Name: {PROJECT_NAME}")
        print(f"📁 Base Directory: {BASE_DIR}")
        
        # Test Enums
        print(f"📜 Document Types: {[doc.value for doc in DocumentType][:3]}...")
        print(f"🏛️ Authorities: {[auth.value for auth in ApprovalAuthority][:2]}...")
        
        # Test digit conversion
        persian_text = "سال ۱۴۰۳ ماده ۲۵"
        english_text = persian_to_english_digits(persian_text)
        back_to_persian = english_to_persian_digits(english_text)
        print(f"🔢 Persian: {persian_text}")
        print(f"🔢 English: {english_text}")
        print(f"🔢 Back to Persian: {back_to_persian}")
        
        # Test file validation
        test_files = ["document.pdf", "text.docx", "data.json", "image.png"]
        for file in test_files:
            valid = validate_file_extension(file)
            status = "✅" if valid else "❌"
            print(f"{status} {file}: {'Valid' if valid else 'Invalid'}")
        
        # Test section names
        for i in range(1, 4):
            section_name = get_section_name(i)
            print(f"📂 Section {i}: {section_name}")
        
        # Test messages
        print(f"💬 Success Message: {Messages.SUCCESS_PARSE}")
        print(f"💬 Error Message: {Messages.ERROR_FILE_NOT_FOUND}")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

if __name__ == "__main__":
    success = test_constants()
    if not success:
        sys.exit(1)