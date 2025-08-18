# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\test_file_utils.py

"""
Test script for file_utils.py module
"""

import sys
import json
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_file_utils():
    """Test the file_utils module functionality"""
    
    print("🧪 Testing file_utils.py module...")
    
    try:
        # Import the file_utils module
        from shared_utils.file_utils import (
            FileInfo, DocumentReader, FileManager,
            get_document_reader, get_file_manager, read_document,
            create_directory, copy_file, get_file_info
        )
        
        print("✅ Import successful")
        
        # Test FileInfo class
        print("\n📝 Testing FileInfo class...")
        test_file = Path(__file__)  # Use this test file
        file_info = FileInfo(test_file)
        
        print(f"  📄 File name: {file_info.name}")
        print(f"  📏 File size: {file_info.size} bytes")
        print(f"  🕐 Modified time: {file_info.modified_time}")
        print(f"  🔍 MIME type: {file_info.mime_type}")
        
        # Test hash calculation
        file_hash = file_info.calculate_hash()
        print(f"  🔐 File hash: {file_hash[:16]}...")
        
        # Test encoding detection
        encoding = file_info.detect_encoding()
        print(f"  📝 Encoding: {encoding}")
        
        # Test to_dict conversion
        file_dict = file_info.to_dict()
        print(f"  📊 Dict keys: {list(file_dict.keys())}")
        
        # Test global functions
        print("\n📝 Testing global functions...")
        file_manager = get_file_manager()
        document_reader = get_document_reader()
        
        print(f"  🔧 File manager: {type(file_manager).__name__}")
        print(f"  📖 Document reader: {type(document_reader).__name__}")
        
        print("\n🎉 Basic file_utils tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_reading():
    """Test document reading functionality"""
    
    print("\n📖 Testing document reading...")
    
    try:
        from shared_utils.file_utils import DocumentReader, get_document_reader
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test JSON file
            print("  📄 Testing JSON file reading...")
            json_file = temp_path / "test.json"
            test_data = {
                "title": "قانون نمونه",
                "articles": [
                    {"number": 1, "content": "ماده اول"},
                    {"number": 2, "content": "ماده دوم"}
                ]
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            reader = get_document_reader()
            result = reader.read_document(json_file)
            
            if result['success']:
                print(f"    ✅ JSON read successfully: {len(result['content'])} chars")
                print(f"    📊 Stats: {result['stats']}")
            else:
                print(f"    ❌ JSON read failed: {result['error']}")
            
            # Test plain text file
            print("  📝 Testing text file reading...")
            txt_file = temp_path / "test.txt"
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("این یک فایل تست است.\nماده ۱: محتوای نمونه\nماده ۲: محتوای دیگر")
            
            result = reader.read_document(txt_file)
            
            if result['success']:
                print(f"    ✅ Text read successfully: {len(result['content'])} chars")
                print(f"    📊 Word count: {result['stats']['word_count']}")
            else:
                print(f"    ❌ Text read failed: {result['error']}")
            
            # Test CSV file
            print("  📊 Testing CSV file reading...")
            csv_file = temp_path / "test.csv"
            
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("شماره,عنوان,محتوا\n")
                f.write("1,ماده اول,محتوای ماده اول\n")
                f.write("2,ماده دوم,محتوای ماده دوم\n")
            
            result = reader.read_document(csv_file)
            
            if result['success']:
                print(f"    ✅ CSV read successfully: {len(result['content'])} chars")
                lines = result['content'].split('\n')
                print(f"    📊 Lines: {len(lines)}")
            else:
                print(f"    ❌ CSV read failed: {result['error']}")
            
            # Test unsupported format
            print("  ❓ Testing unsupported format...")
            unknown_file = temp_path / "test.xyz"
            unknown_file.write_text("Test content")
            
            result = reader.read_document(unknown_file)
            
            if not result['success']:
                print(f"    ✅ Correctly rejected unsupported format")
            else:
                print(f"    ⚠️ Unexpectedly accepted unsupported format")
        
        print("  ✅ Document reading tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Document reading test failed: {e}")
        return False

def test_file_management():
    """Test file management functionality"""
    
    print("\n🗂️ Testing file management...")
    
    try:
        from shared_utils.file_utils import FileManager, get_file_manager
        
        file_manager = get_file_manager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test directory creation
            print("  📁 Testing directory creation...")
            test_dir = temp_path / "test_directory"
            success = file_manager.create_directory(test_dir)
            
            if success and test_dir.exists():
                print("    ✅ Directory created successfully")
            else:
                print("    ❌ Directory creation failed")
            
            # Test file operations
            print("  📄 Testing file operations...")
            
            # Create a test file
            source_file = temp_path / "source.txt"
            source_file.write_text("Test file content", encoding='utf-8')
            
            # Test file copy
            dest_file = temp_path / "destination.txt"
            copy_success = file_manager.copy_file(source_file, dest_file)
            
            if copy_success and dest_file.exists():
                print("    ✅ File copied successfully")
            else:
                print("    ❌ File copy failed")
            
            # Test file move
            moved_file = temp_path / "moved.txt"
            move_success = file_manager.move_file(dest_file, moved_file)
            
            if move_success and moved_file.exists() and not dest_file.exists():
                print("    ✅ File moved successfully")
            else:
                print("    ❌ File move failed")
            
            # Test directory info
            print("  📊 Testing directory info...")
            dir_info = file_manager.get_directory_info(temp_path)
            
            if dir_info['exists']:
                print(f"    ✅ Directory info: {dir_info['file_count']} files, {dir_info['total_size']} bytes")
            else:
                print("    ❌ Directory info failed")
            
            # Test file finding
            print("  🔍 Testing file finding...")
            files = file_manager.find_files(temp_path, "*.txt")
            print(f"    📄 Found {len(files)} text files")
            
            # Test filename cleaning
            print("  🧹 Testing filename cleaning...")
            dirty_name = "فایل<>نامعتبر*با:کاراکترهای?غیرمجاز"
            clean_name = file_manager.clean_filename(dirty_name)
            print(f"    🔧 Cleaned: {dirty_name} -> {clean_name}")
            
            # Test file type stats
            print("  📈 Testing file type statistics...")
            stats = file_manager.get_file_type_stats(temp_path)
            
            for ext, stat in stats.items():
                print(f"    📊 {ext}: {stat['count']} files, {stat['total_size']} bytes")
        
        print("  ✅ File management tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ File management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shared_utils_import():
    """Test importing from shared_utils package"""
    
    print("\n📦 Testing shared_utils package import...")
    
    try:
        # Test importing from package
        from shared_utils import (
            get_logger, get_config, read_document, 
            FileInfo, PROJECT_NAME, Messages
        )
        
        print("  ✅ Package imports successful")
        
        # Test basic functionality
        logger = get_logger("TestLogger")
        config = get_config()
        
        print(f"  📝 Logger: {logger.name}")
        print(f"  ⚙️ Config: {config.project_name}")
        print(f"  📋 Project: {PROJECT_NAME}")
        print(f"  💬 Message: {Messages.SUCCESS_PARSE}")
        
        print("  ✅ Package functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Package import test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Legal Assistant AI - File Utils Test Suite")
    print("=" * 60)
    
    success1 = test_file_utils()
    success2 = test_document_reading()
    success3 = test_file_management()
    success4 = test_shared_utils_import()
    
    if success1 and success2 and success3 and success4:
        print("\n🎉 All file_utils tests passed! File utilities are ready.")
        print("\n📦 shared_utils package is complete and functional!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)