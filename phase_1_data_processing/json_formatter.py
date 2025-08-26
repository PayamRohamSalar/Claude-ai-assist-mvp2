"""
JSON Formatter

Role: Build the canonical JSON schema for each processed document and update global indexes.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for shared_utils imports
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils.logger import get_logger
from shared_utils.constants import PROCESSED_PHASE_1_DIR

logger = get_logger(__name__)


class JsonFormatter:
    """Formatter for creating canonical JSON documents and managing indexes."""
    
    def __init__(self, out_dir: Optional[Path] = None):
        """
        Initialize the JSON formatter.
        
        Args:
            out_dir: Output directory for processed files. Defaults to PROCESSED_PHASE_1_DIR.
        """
        self.out_dir = out_dir or PROCESSED_PHASE_1_DIR
        self.out_dir = Path(self.out_dir)
        
        # Ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"JsonFormatter initialized with output directory: {self.out_dir}")
    
    def build_document_json(
        self,
        metadata,  # Metadata object
        clean,     # CleanResult object  
        parsed,    # ParsedStructure object
        source_file: str,
        processing_ms: float,
        pipeline_versions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Build the canonical JSON representation of a processed document.
        
        Args:
            metadata: Metadata object with document information
            clean: CleanResult object with cleaned text variants
            parsed: ParsedStructure object with document structure
            source_file: Original source filename
            processing_ms: Processing duration in milliseconds
            pipeline_versions: Optional pipeline component versions
            
        Returns:
            Complete document JSON dictionary
        """
        logger.info(f"Building document JSON for: {metadata.title}")
        
        # Build metadata section
        metadata_dict = {
            "title": metadata.title,
            "document_type": metadata.document_type,
            "approval_authority": metadata.approval_authority,
            "approval_date": metadata.approval_date,
            "effective_date": metadata.effective_date,
            "section_name": metadata.section_name,
            "document_number": metadata.document_number,
            "subject": metadata.subject,
            "keywords": metadata.keywords,
            "related_docs": metadata.related_docs,
            "confidence_score": metadata.confidence_score
        }
        
        # Build content variants section
        # Handle both dictionary and object formats for clean parameter
        if isinstance(clean, dict):
            content_variants = {
                "normalized": clean.get("normalized_text", ""),
                "ascii_digits": clean.get("ascii_digits_text", ""),
                "persian_digits": clean.get("persian_digits_text", "")
            }
        else:
            content_variants = {
                "normalized": clean.normalized_text,
                "ascii_digits": clean.ascii_digits_text,
                "persian_digits": clean.persian_digits_text
            }
        
        # Build structure section
        structure = {
            "chapters": [],
            "totals": {
                "chapters": parsed.total_chapters,
                "articles": parsed.total_articles,
                "notes": parsed.total_notes
            }
        }
        
        # Convert chapters to JSON format
        for chapter in parsed.chapters:
            chapter_dict = {
                "title": chapter.title,
                "articles": []
            }
            
            for article in chapter.articles:
                article_dict = {
                    "number": article.number,
                    "text": article.text,
                    "notes": article.notes,
                    "clauses": article.clauses
                }
                chapter_dict["articles"].append(article_dict)
            
            structure["chapters"].append(chapter_dict)
        
        # Build stats section
        # Handle both dictionary and object formats for clean parameter
        if isinstance(clean, dict):
            stats = {
                "chars_original": len(clean.get("original_text", "")),
                "chars_clean": len(clean.get("normalized_text", ""))
            }
        else:
            stats = {
                "chars_original": len(clean.original_text) if hasattr(clean, 'original_text') else 0,
                "chars_clean": len(clean.normalized_text)
            }
        
        # Build processing section
        processing = {
            "source_filename": source_file,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "duration_ms": processing_ms,
            "pipeline_versions": pipeline_versions or {}
        }
        
        # Compose the complete document JSON
        document_json = {
            "metadata": metadata_dict,
            "content_variants": content_variants,
            "structure": structure,
            "stats": stats,
            "processing": processing
        }
        
        logger.debug(f"Document JSON built successfully for: {metadata.title}")
        return document_json
    
    def save_document(self, doc_json: Dict[str, Any], idx: int) -> Path:
        """
        Save a document JSON to file with sanitized filename.
        
        Args:
            doc_json: Document JSON dictionary
            idx: Document index for filename
            
        Returns:
            Path to the saved file
        """
        # Get title and sanitize filename
        title = doc_json.get("metadata", {}).get("title", "")
        sanitized_title = self._sanitize_filename(title)
        
        # If title is empty, use fallback
        if not sanitized_title:
            sanitized_title = "doc"
        
        # Create filename with index
        filename = f"{sanitized_title}__{idx:03d}.json"
        file_path = self.out_dir / filename
        
        # Save JSON file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_json, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Document saved: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save document {filename}: {e}")
            raise
    
    def update_indexes(self, all_docs: List[Dict[str, Any]]) -> None:
        """
        Update global index files with aggregated data.
        
        Args:
            all_docs: List of all document JSON dictionaries
        """
        logger.info(f"Updating indexes for {len(all_docs)} documents")
        
        # Build documents metadata index
        metadata_index = []
        for doc in all_docs:
            metadata = doc.get("metadata", {})
            stats = doc.get("stats", {})
            processing = doc.get("processing", {})
            structure_totals = doc.get("structure", {}).get("totals", {})
            
            doc_metadata = {
                "title": metadata.get("title", ""),
                "document_type": metadata.get("document_type", ""),
                "approval_authority": metadata.get("approval_authority", ""),
                "approval_date": metadata.get("approval_date", ""),
                "section_name": metadata.get("section_name", ""),
                "document_number": metadata.get("document_number", ""),
                "subject": metadata.get("subject", ""),
                "confidence_score": metadata.get("confidence_score", 0.0),
                "chars_clean": stats.get("chars_clean", 0),
                "total_articles": structure_totals.get("articles", 0),
                "total_notes": structure_totals.get("notes", 0),
                "source_filename": processing.get("source_filename", ""),
                "timestamp_utc": processing.get("timestamp_utc", "")
            }
            metadata_index.append(doc_metadata)
        
        # Save documents metadata index
        metadata_file = self.out_dir / "documents_metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_index, f, ensure_ascii=False, indent=2)
            logger.info(f"Documents metadata index saved: {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata index: {e}")
            raise
        
        # Build statistics aggregation
        total_documents = len(all_docs)
        total_articles = sum(
            doc.get("structure", {}).get("totals", {}).get("articles", 0) 
            for doc in all_docs
        )
        total_notes = sum(
            doc.get("structure", {}).get("totals", {}).get("notes", 0)
            for doc in all_docs
        )
        total_chars_clean = sum(
            doc.get("stats", {}).get("chars_clean", 0)
            for doc in all_docs
        )
        
        statistics = {
            "summary": {
                "total_documents": total_documents,
                "total_articles": total_articles,
                "total_notes": total_notes,
                "total_chars_clean": total_chars_clean
            },
            "averages": {
                "articles_per_document": total_articles / total_documents if total_documents > 0 else 0,
                "notes_per_document": total_notes / total_documents if total_documents > 0 else 0,
                "chars_per_document": total_chars_clean / total_documents if total_documents > 0 else 0
            },
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "documents_processed": total_documents
        }
        
        # Save statistics
        stats_file = self.out_dir / "statistics.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            logger.info(f"Statistics saved: {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
            raise
        
        logger.info(f"Indexes updated successfully. Total documents: {total_documents}")
    
    def _sanitize_filename(self, title: str) -> str:
        """
        Sanitize a title string for use as filename.
        
        Args:
            title: Original title string
            
        Returns:
            Sanitized filename string
        """
        if not title:
            return ""
        
        # Remove or replace problematic characters
        # Keep Persian characters, letters, digits, spaces, and safe punctuation
        sanitized = re.sub(r'[^\w\s\u0600-\u06FF\u200C\u200D.-]', '_', title)
        
        # Replace multiple spaces/underscores with single underscore
        sanitized = re.sub(r'[\s_]+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Limit length to reasonable filename size
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized
    
    def validate_json_output(self, file_path: Path) -> bool:
        """
        Validate that a JSON file is valid UTF-8 and properly formatted.
        
        Args:
            file_path: Path to JSON file to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            logger.debug(f"JSON validation successful: {file_path}")
            return True
        except (json.JSONDecodeError, UnicodeDecodeError, Exception) as e:
            logger.error(f"JSON validation failed for {file_path}: {e}")
            return False
    
    def get_next_document_index(self) -> int:
        """
        Get the next available document index by checking existing files.
        
        Returns:
            Next available index number
        """
        existing_files = list(self.out_dir.glob("*__*.json"))
        if not existing_files:
            return 1
        
        # Extract indices from existing files
        indices = []
        for file_path in existing_files:
            match = re.search(r'__(\d+)\.json$', file_path.name)
            if match:
                indices.append(int(match.group(1)))
        
        return max(indices) + 1 if indices else 1
