# path: D:\OneDrive\AI-Project\Claude-ai-assist-mvp2\phase_1_data_processing\metadata_extractor.py

"""
Legal Assistant AI - Metadata Extractor
Extracts metadata from Persian legal documents including title, approval date, 
approval authority, document type, and section label.
Uses explicit text matching without interpretation beyond what's in the text.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Import shared utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils import (
    get_logger, get_config, Messages, BASE_DIR,
    DocumentType, ApprovalAuthority, DocumentSection
)


@dataclass
class DocumentMetadata:
    """Structured metadata for legal documents"""
    title: Optional[str] = None
    approval_date: Optional[str] = None
    approval_authority: Optional[str] = None
    document_type: Optional[str] = None
    section_label: Optional[str] = None
    extraction_confidence: float = 0.0
    extraction_method: str = "regex"
    extraction_timestamp: str = ""
    # Additional attributes expected by JsonFormatter
    effective_date: Optional[str] = None
    section_name: Optional[str] = None
    document_number: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = None
    related_docs: List[str] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        """Initialize default values for list fields."""
        if self.keywords is None:
            self.keywords = []
        if self.related_docs is None:
            self.related_docs = []
        # Use extraction_confidence as confidence_score if not set
        if self.confidence_score == 0.0 and self.extraction_confidence > 0.0:
            self.confidence_score = self.extraction_confidence


class PersianMetadataExtractor:
    """
    Extracts metadata from Persian legal documents using robust regex patterns
    Focuses on explicit text matching without interpretation
    """
    
    def __init__(self):
        self.logger = get_logger("MetadataExtractor")
        self.config = get_config()
        self._init_extraction_patterns()
        
    def _init_extraction_patterns(self):
        """Initialize all regex patterns for metadata extraction"""
        
        # Title patterns
        self.title_patterns = [
            r'^(قانون\s+[^\n\r]+?)(?:\n|$)',
            r'^(آیین‌نامه\s+[^\n\r]+?)(?:\n|$)',
            r'^(دستورالعمل\s+[^\n\r]+?)(?:\n|$)',
            r'^(مصوبه\s+[^\n\r]+?)(?:\n|$)',
            r'^(اساسنامه\s+[^\n\r]+?)(?:\n|$)',
            r'^(راهنما\s+[^\n\r]+?)(?:\n|$)',
            r'^(بخشنامه\s+[^\n\r]+?)(?:\n|$)',
            r'^(سیاست\s+[^\n\r]+?)(?:\n|$)',
        ]
        
        # Approval date patterns
        self.approval_date_patterns = [
            r'مصوب\s+(\d{1,2}/\d{1,2}/\d{4})',
            r'مصوب\s+(\d{4}/\d{1,2}/\d{1,2})',
            r'مصوب\s+(\d{1,2}\.\d{1,2}\.\d{4})',
            r'مصوب\s+سال\s+(\d{4})',
            r'\(مصوب\s+(\d{1,2}/\d{1,2}/\d{4})\)',
        ]
        
        # Approval authority patterns
        self.approval_authority_patterns = [
            r'مصوب\s+[^)]*?([^)]*?مجلس\s+شورای\s+اسلامی[^)]*?)(?:\)|$)',
            r'مصوب\s+[^)]*?([^)]*?هیئت\s+وزیران[^)]*?)(?:\)|$)',
            r'مصوب\s+[^)]*?([^)]*?شورای\s+عالی\s+انقلاب\s+فرهنگی[^)]*?)(?:\)|$)',
            r'مصوب\s+[^)]*?([^)]*?شورای\s+عالی\s+علوم[^)]*?)(?:\)|$)',
            r'مصوب\s+[^)]*?([^)]*?وزارت\s+علوم[^)]*?)(?:\)|$)',
            r'مصوب\s+[^)]*?([^)]*?قوه\s+قضاییه[^)]*?)(?:\)|$)',
        ]
        
        # Document type patterns
        self.document_type_patterns = [
            (r'^قانون\s+', 'قانون'),
            (r'^آیین‌نامه\s+', 'آیین‌نامه'),
            (r'^دستورالعمل\s+', 'دستورالعمل'),
            (r'^مصوبه\s+', 'مصوبه'),
            (r'^اساسنامه\s+', 'اساسنامه'),
            (r'^راهنما\s+', 'راهنما'),
            (r'^بخشنامه\s+', 'بخشنامه'),
            (r'^سیاست\s+', 'سیاست'),
            # Additional patterns for documents that start with title
            (r'قانون\s+[^مصوب\n\r]+?مصوب', 'قانون'),
            (r'آیین‌نامه\s+[^مصوب\n\r]+?مصوب', 'آیین‌نامه'),
            (r'دستورالعمل\s+[^مصوب\n\r]+?مصوب', 'دستورالعمل'),
        ]
        
        # Section label patterns
        self.section_label_patterns = [
            r'بخش\s+(اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم)',
            r'فصل\s+(اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم)',
            r'بخش\s+(\d+)',
            r'فصل\s+(\d+)',
            r'بخش\s+(اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم)\s*[-–]\s*([^\n\r]+)',
        ]
    
    def extract_metadata(self, text: str, document_id: str = None) -> DocumentMetadata:
        """Extract all metadata from a legal document"""
        
        if not text or not isinstance(text, str):
            return DocumentMetadata()
        
        self.logger.debug(f"Starting metadata extraction for document: {document_id}")
        
        metadata = DocumentMetadata()
        metadata.extraction_timestamp = datetime.now().isoformat()
        
        # Extract each metadata field
        metadata.title = self._extract_title(text)
        metadata.approval_date = self._extract_approval_date(text)
        metadata.approval_authority = self._extract_approval_authority(text)
        metadata.document_type = self._extract_document_type(text)
        metadata.section_label = self._extract_section_label(text)
        
        # Calculate confidence score
        metadata.extraction_confidence = self._calculate_confidence(metadata, text)
        
        return metadata
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract document title from the beginning of the text"""
        clean_text = text.strip()
        
        for pattern in self.title_patterns:
            match = re.search(pattern, clean_text, re.MULTILINE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'\s+', ' ', title)
                if len(title) >= 5:
                    return title
        return None
    
    def _extract_approval_date(self, text: str) -> Optional[str]:
        """Extract approval date from the text"""
        for pattern in self.approval_date_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                date_str = match.group(1).strip()
                date_str = re.sub(r'\s+', '', date_str)
                if self._is_valid_date_format(date_str):
                    return date_str
        return None
    
    def _extract_approval_authority(self, text: str) -> Optional[str]:
        """Extract approval authority from the text"""
        for pattern in self.approval_authority_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                authority = match.group(1).strip()
                authority = re.sub(r'\s+', ' ', authority)
                # Clean up authority text - remove date if present
                authority = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '', authority).strip()
                authority = re.sub(r'\d{4}/\d{1,2}/\d{1,2}', '', authority).strip()
                if len(authority) >= 5:
                    return authority
        return None
    
    def _extract_document_type(self, text: str) -> Optional[str]:
        """Extract document type from the text"""
        for pattern, doc_type in self.document_type_patterns:
            if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                return doc_type
        return None
    
    def _extract_section_label(self, text: str) -> Optional[str]:
        """Extract section label from the text"""
        for pattern in self.section_label_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                if len(match.groups()) == 2:
                    section_num = match.group(1)
                    section_name = match.group(2).strip()
                    return f"بخش {section_num} - {section_name}"
                else:
                    section_num = match.group(1)
                    return f"بخش {section_num}"
        return None
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Validate if the extracted date string has a valid format"""
        date_patterns = [
            r'^\d{1,2}/\d{1,2}/\d{4}$',
            r'^\d{4}/\d{1,2}/\d{1,2}$',
            r'^\d{1,2}\.\d{1,2}\.\d{4}$',
            r'^\d{4}\.\d{1,2}\.\d{1,2}$',
            r'^\d{4}$',
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, date_str):
                return True
        return False
    
    def _calculate_confidence(self, metadata: DocumentMetadata, text: str) -> float:
        """Calculate confidence score for the extracted metadata"""
        confidence = 0.0
        field_weights = {
            'title': 0.3,
            'approval_date': 0.2,
            'approval_authority': 0.2,
            'document_type': 0.2,
            'section_label': 0.1
        }
        
        if metadata.title:
            confidence += field_weights['title']
        if metadata.approval_date:
            confidence += field_weights['approval_date']
        if metadata.approval_authority:
            confidence += field_weights['approval_authority']
        if metadata.document_type:
            confidence += field_weights['document_type']
        if metadata.section_label:
            confidence += field_weights['section_label']
        
        if len(text.strip()) > 100:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def extract_metadata_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract metadata for a batch of documents"""
        self.logger.info(f"Starting batch metadata extraction for {len(documents)} documents")
        
        processed_documents = []
        
        for i, doc in enumerate(documents):
            try:
                if 'content' in doc:
                    metadata = self.extract_metadata(
                        doc['content'], 
                        doc.get('document_id', f'doc_{i}')
                    )
                    
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    
                    doc['metadata'].update(asdict(metadata))
                    doc['metadata']['metadata_extracted'] = True
                    doc['metadata']['extraction_date'] = datetime.now().isoformat()
                
                processed_documents.append(doc)
                
            except Exception as e:
                self.logger.error(f"Error extracting metadata for document {i+1}: {str(e)}")
                processed_documents.append(doc)
        
        return processed_documents
    
    def process_cleaned_documents(self, input_file: Path, output_dir: Path = None) -> Path:
        """Process cleaned documents and extract metadata"""
        if output_dir is None:
            output_dir = BASE_DIR / "data" / "processed_phase_1"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'documents' in data:
                data['documents'] = self.extract_metadata_batch(data['documents'])
            
            data['metadata_extraction'] = {
                'processed': True,
                'processing_date': datetime.now().isoformat(),
                'extractor_version': '1.0',
                'total_documents_processed': len(data.get('documents', []))
            }
            
            input_stem = input_file.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"metadata_{input_stem}_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error processing cleaned documents: {str(e)}")
            raise
    
    def get_extraction_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about metadata extraction"""
        stats = {
            'total_documents': len(documents),
            'documents_with_title': 0,
            'documents_with_date': 0,
            'documents_with_authority': 0,
            'documents_with_type': 0,
            'documents_with_section': 0,
            'average_confidence': 0.0,
            'document_types': {},
            'approval_authorities': {},
        }
        
        total_confidence = 0.0
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            
            if metadata.get('title'):
                stats['documents_with_title'] += 1
            if metadata.get('approval_date'):
                stats['documents_with_date'] += 1
            if metadata.get('approval_authority'):
                stats['documents_with_authority'] += 1
                authority = metadata['approval_authority']
                stats['approval_authorities'][authority] = stats['approval_authorities'].get(authority, 0) + 1
            if metadata.get('document_type'):
                stats['documents_with_type'] += 1
                doc_type = metadata['document_type']
                stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
            if metadata.get('section_label'):
                stats['documents_with_section'] += 1
            
            total_confidence += metadata.get('extraction_confidence', 0.0)
        
        if documents:
            stats['average_confidence'] = total_confidence / len(documents)
        
        return stats


def main():
    """Main execution function for testing"""
    extractor = PersianMetadataExtractor()
    
    test_text = """
    قانون مقررات انتظامی هیئت علمی دانشگاه‌ها و مراکز آموزش عالی
    
    ماده 1. هیئت‌های رسیدگی عبارتند از:
    
    1. هیئت بدوی.
    2. هیئت تجدید نظر.
    
    تبصره 1. انتخاب مجدد اشخاص مزبور بلامانع است.
    
    بخش اول - مقررات عمومی
    
    (مصوب 22/12/1364 مجلس شورای اسلامی)
    """
    
    print("🔍 Testing Persian Metadata Extractor...")
    
    metadata = extractor.extract_metadata(test_text, "TEST_001")
    
    print("\n📊 Extracted Metadata:")
    print(f"  Title: {metadata.title}")
    print(f"  Approval Date: {metadata.approval_date}")
    print(f"  Approval Authority: {metadata.approval_authority}")
    print(f"  Document Type: {metadata.document_type}")
    print(f"  Section Label: {metadata.section_label}")
    print(f"  Confidence: {metadata.extraction_confidence:.2f}")
    
    print("\n✅ Metadata extractor testing completed!")


if __name__ == "__main__":
    main()
