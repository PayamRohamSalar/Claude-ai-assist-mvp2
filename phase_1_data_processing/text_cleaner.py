# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\phase_1_data_processing\text_cleaner.py

"""
Legal Assistant AI - Text Cleaner
Cleans and standardizes Persian legal text content
Handles encoding issues, normalization, and formatting cleanup
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import unicodedata

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils import (
    get_logger, get_config, Messages,
    persian_to_english_digits, BASE_DIR
)


class PersianTextCleaner:
    """
    Advanced Persian text cleaner for legal documents
    Handles various encoding issues and text normalization
    """
    
    def __init__(self):
        self.logger = get_logger("PersianTextCleaner")
        self.config = get_config()
        
        # Persian character mappings for normalization
        self.persian_char_map = {
            # Normalize different forms of Persian characters
            'ي': 'ی',  # Arabic yeh to Persian yeh
            'ك': 'ک',  # Arabic kaf to Persian kaf
            'ء': 'ٔ',  # Hamza forms
            'أ': 'آ',  # Alef with hamza
            'إ': 'ا',  # Alef with hamza below
            'ؤ': 'و',  # Waw with hamza
            'ئ': 'ی',  # Yeh with hamza
            
            # Zero-width characters
            '\u200c': '‌',  # ZWNJ (zero-width non-joiner)
            '\u200d': '',   # ZWJ (zero-width joiner) - remove
            '\u200e': '',   # LTR mark - remove
            '\u200f': '',   # RTL mark - remove
            '\ufeff': '',   # BOM - remove
        }
        
        # Common encoding issues and fixes
        self.encoding_fixes = {
            # Common Word export issues
            'â€Œ': '‌',  # ZWNJ
            'â€‹': '',   # ZWSP (zero-width space)
            'â€': '"',   # Left double quotation
            'â€': '"',   # Right double quotation
            'â€™': "'",  # Right single quotation
            'â€œ': '"',  # Left double quotation
            'â€': '–',   # En dash
            'â€"': '—',  # Em dash
            
            # Persian text encoding issues from corrupted exports
            'Ù‚Ø§Ù†ÙˆÙ†': 'قانون',
            'Ù…Ø§Ø¯Ù‡': 'ماده',
            'ØªØ¨ØµØ±Ù‡': 'تبصره',
            'Ù…Ø¬Ù„Ø³': 'مجلس',
            'Ø´ÙˆØ±Ø§ÛŒ': 'شورای',
            'Ø§Ø³Ù„Ø§Ù…ÛŒ': 'اسلامی',
            'Ø¯Ø§Ù†Ø´Ú¯Ø§Ù‡': 'دانشگاه',
            'Ù‡ÛŒØ¦Øª': 'هیئت',
            'Ø¹Ù„Ù…ÛŒ': 'علمی',
            'ÙˆØ²ÛŒØ±Ø§Ù†': 'وزیران',
            'Ø¢Ù…ÙˆØ²Ø´': 'آموزش',
            'Ø¹Ø§Ù„ÛŒ': 'عالی',
        }
    
    def clean_text(self, text: str) -> str:
        """
        Main text cleaning method - applies all cleaning steps
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        
        if not text or not isinstance(text, str):
            return ""
        
        self.logger.debug("Starting text cleaning process")
        
        # Step 1: Fix encoding issues
        text = self._fix_encoding_issues(text)
        
        # Step 2: Normalize Persian characters
        text = self._normalize_persian_text(text)
        
        # Step 3: Standardize numbering
        text = self._standardize_numbering(text)
        
        # Step 4: Clean formatting
        text = self._clean_formatting(text)
        
        # Step 5: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 6: Fix legal document specific issues
        text = self._fix_legal_patterns(text)
        
        self.logger.debug("Text cleaning completed")
        
        return text.strip()
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues in Persian text"""
        
        # Apply encoding fixes
        for encoded, correct in self.encoding_fixes.items():
            text = text.replace(encoded, correct)
        
        # Normalize Unicode
        try:
            text = unicodedata.normalize('NFKC', text)
        except:
            pass  # Skip if normalization fails
        
        # Remove or fix problematic characters
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _normalize_persian_text(self, text: str) -> str:
        """Normalize Persian characters to standard forms"""
        
        # Apply character mappings
        for old_char, new_char in self.persian_char_map.items():
            text = text.replace(old_char, new_char)
        
        # Fix common Persian text issues
        text = re.sub(r'ه\u200c*ی\b', 'هی', text)  # Fix heh-yeh combinations
        text = re.sub(r'ت\u200c*ان\b', 'تان', text)  # Fix possessive endings
        
        return text
    
    def _standardize_numbering(self, text: str) -> str:
        """Standardize numbering throughout the text"""
        
        # Convert Persian digits to English in legal contexts
        # Use the shared utility function
        text = persian_to_english_digits(text)
        
        return text
    
    def _clean_formatting(self, text: str) -> str:
        """Clean Word formatting artifacts and inconsistencies"""
        
        # Remove excessive formatting marks
        text = re.sub(r'\*{3,}', '***', text)  # Normalize asterisk lines
        text = re.sub(r'-{3,}', '---', text)   # Normalize dash lines
        text = re.sub(r'={3,}', '===', text)   # Normalize equal lines
        
        # Clean markdown-like formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', text)  # Fix bold
        text = re.sub(r'\[([^\]]+)\]\{dir="rtl"\}', r'\1', text)  # Remove RTL markers
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipsis
        text = re.sub(r',{2,}', ',', text)     # Remove multiple commas
        
        # Clean up brackets and parentheses
        text = re.sub(r'\(\s*\)', '', text)    # Remove empty parentheses
        text = re.sub(r'\[\s*\]', '', text)    # Remove empty brackets
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([،؛:.!؟])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([،؛:.!؟])\s*([^\s])', r'\1 \2', text)  # Add space after punctuation
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace in the text"""
        
        # Replace various whitespace characters with standard space
        text = re.sub(r'[\t\f\v\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)     # Windows to Unix line endings
        text = re.sub(r'\r', '\n', text)       # Mac to Unix line endings
        
        # Clean up excessive whitespace
        text = re.sub(r' {2,}', ' ', text)     # Multiple spaces to single space
        text = re.sub(r'\n{3,}', '\n\n', text) # Multiple newlines to double newline
        
        # Fix spacing around ZWNJ
        text = re.sub(r'\s*‌\s*', '‌', text)   # Clean ZWNJ spacing
        
        # Remove trailing whitespace from lines
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _fix_legal_patterns(self, text: str) -> str:
        """Fix legal document specific patterns"""
        
        # Standardize article format
        text = re.sub(r'‌?(ماده)\s*(\d+)\.?', r'\1 \2.', text)
        
        # Standardize note format
        text = re.sub(r'‌?(تبصره)\s*(\d+)\.?', r'\1 \2.', text)
        
        # Standardize section headers
        text = re.sub(r'(فصل|بخش)\s*(اول|دوم|سوم|چهارم|پنجم)', r'\1 \2', text)
        
        # Fix approval date format
        text = re.sub(r'\(مصوب\s*(\d{1,2}/\d{1,2}/\d{4})\s*([^)]*)\)', 
                     r'(مصوب \1 \2)', text)
        
        return text
    
    def clean_document_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean text for a batch of documents
        
        Args:
            documents: List of document dictionaries with 'content' key
            
        Returns:
            List of documents with cleaned content
        """
        
        self.logger.info(
            f"Starting batch cleaning for {len(documents)} documents",
            f"شروع پاکسازی دسته‌ای برای {len(documents)} سند"
        )
        
        cleaned_documents = []
        
        for i, doc in enumerate(documents):
            try:
                if 'content' in doc:
                    # Clean the main content
                    original_length = len(doc['content'])
                    doc['content'] = self.clean_text(doc['content'])
                    cleaned_length = len(doc['content'])
                    
                    # Also clean title if present
                    if 'title' in doc:
                        doc['title'] = self.clean_text(doc['title'])
                    
                    # Update metadata
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    
                    doc['metadata'].update({
                        'text_cleaned': True,
                        'cleaning_date': datetime.now().isoformat(),
                        'original_length': original_length,
                        'cleaned_length': cleaned_length,
                        'length_reduction': original_length - cleaned_length
                    })
                    
                    self.logger.debug(
                        f"Document {i+1} cleaned: {original_length} -> {cleaned_length} chars"
                    )
                
                cleaned_documents.append(doc)
                
            except Exception as e:
                self.logger.error(
                    f"Error cleaning document {i+1}: {str(e)}",
                    f"خطا در پاکسازی سند {i+1}: {str(e)}"
                )
                # Keep original document in case of error
                cleaned_documents.append(doc)
        
        self.logger.info(
            f"Batch cleaning completed for {len(cleaned_documents)} documents",
            f"پاکسازی دسته‌ای برای {len(cleaned_documents)} سند تکمیل شد"
        )
        
        return cleaned_documents
    
    def process_split_documents(self, input_file: Path, output_dir: Path = None) -> Path:
        """
        Process documents from document_splitter output and clean all text
        
        Args:
            input_file: Path to JSON file from document_splitter
            output_dir: Output directory for cleaned results
            
        Returns:
            Path to cleaned output file
        """
        
        if output_dir is None:
            output_dir = BASE_DIR / "data" / "processed_phase_1"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"Processing split documents from: {input_file}",
            f"پردازش اسناد تفکیک شده از: {input_file}"
        )
        
        try:
            # Load split documents
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clean all documents
            if 'documents' in data:
                data['documents'] = self.clean_document_batch(data['documents'])
            
            # Add cleaning metadata to main data
            data['text_cleaning'] = {
                'processed': True,
                'processing_date': datetime.now().isoformat(),
                'cleaner_version': '1.0',
                'total_documents_processed': len(data.get('documents', []))
            }
            
            # Generate output filename
            input_stem = input_file.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"cleaned_{input_stem}_{timestamp}.json"
            
            # Save cleaned results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(
                f"Cleaned documents saved to: {output_file}",
                f"اسناد پاکسازی شده ذخیره شد در: {output_file}"
            )
            
            return output_file
            
        except Exception as e:
            error_msg = f"Error processing split documents: {str(e)}"
            self.logger.error(error_msg, f"خطا در پردازش اسناد تفکیک شده: {str(e)}")
            raise

    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """
        Generate statistics about the cleaning process
        
        Args:
            original_text: Text before cleaning
            cleaned_text: Text after cleaning
            
        Returns:
            Dictionary with cleaning statistics
        """
        
        stats = {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'reduction_chars': len(original_text) - len(cleaned_text),
            'reduction_percent': round(((len(original_text) - len(cleaned_text)) / len(original_text)) * 100, 2) if original_text else 0,
            'original_words': len(original_text.split()),
            'cleaned_words': len(cleaned_text.split()),
            'word_reduction': len(original_text.split()) - len(cleaned_text.split()),
            'encoding_issues_fixed': self._count_encoding_fixes(original_text),
            'persian_chars_normalized': self._count_persian_normalizations(original_text),
        }
        
        return stats
    
    def _count_encoding_fixes(self, text: str) -> int:
        """Count number of encoding issues found and fixed"""
        count = 0
        for encoded_char in self.encoding_fixes.keys():
            count += text.count(encoded_char)
        return count
    
    def _count_persian_normalizations(self, text: str) -> int:
        """Count number of Persian character normalizations"""
        count = 0
        for old_char in self.persian_char_map.keys():
            count += text.count(old_char)
        return count


def main():
    """Main execution function for testing"""
    
    # Test the text cleaner
    cleaner = PersianTextCleaner()
    
    # Test text with common issues - using clean Persian text for testing
    test_text = """
    قانون مقررات انتظامی هیئت علمی
    
    ماده 1. هیئت‌های رسیدگی عبارتند از:
    
    1. هیئت بدوی.
    2. هیئت تجدید نظر.
    
    تبصره 1. انتخاب مجدد اشخاص مزبور بلامانع است.
    
    **(مصوب 22/12/1364 مجلس شورای اسلامی)**
    """
    
    print("🧹 Testing Persian Text Cleaner...")
    print(f"📏 Original length: {len(test_text)} characters")
    print("\n📝 Original text (first 200 chars):")
    print(repr(test_text[:200]))
    
    # Clean the text
    cleaned_text = cleaner.clean_text(test_text)
    
    print(f"\n✨ Cleaned length: {len(cleaned_text)} characters")
    print("\n📝 Cleaned text:")
    print(cleaned_text)
    
    # Get statistics
    stats = cleaner.get_cleaning_stats(test_text, cleaned_text)
    print("\n📊 Cleaning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test with document structure
    print("\n" + "="*50)
    print("🔍 Testing with document batch...")
    
    test_docs = [
        {
            'title': 'قانون تستی',
            'content': test_text,
            'document_id': 'TEST_001'
        }
    ]
    
    cleaned_docs = cleaner.clean_document_batch(test_docs)
    
    print(f"✅ Processed {len(cleaned_docs)} documents")
    print(f"📋 First document title: {cleaned_docs[0]['title']}")
    print(f"📊 Metadata: {cleaned_docs[0].get('metadata', {})}")
    
    # Test file processing if split file exists
    split_files = list((BASE_DIR / "data" / "processed_phase_1").glob("split_*.json"))
    if split_files:
        print(f"\n🔄 Testing file processing with: {split_files[0].name}")
        try:
            output_file = cleaner.process_split_documents(split_files[0])
            print(f"✅ File processing completed: {output_file}")
        except Exception as e:
            print(f"⚠️ File processing test failed: {e}")
    else:
        print("\n💡 No split files found for file processing test")
        print("   Run document_splitter.py first to test file processing")
    
    print("\n✅ Text cleaner testing completed!")


if __name__ == "__main__":
    main()