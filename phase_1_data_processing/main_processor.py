"""
Main Processor

Role: Orchestrate the Phase-1 pipeline end-to-end.
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for shared_utils imports
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils.logger import get_logger
from shared_utils.constants import RAW_DATA_DIR, PROCESSED_PHASE_1_DIR, Messages

# Import pipeline components
from .legal_structure_parser import LegalStructureParser, ParsedStructure
from .json_formatter import JsonFormatter

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single document split."""
    source_file: str
    split_index: int
    success: bool
    duration_ms: float
    error_message: Optional[str] = None
    document_path: Optional[str] = None
    metadata_title: Optional[str] = None
    articles_count: int = 0
    notes_count: int = 0
    chars_clean: int = 0


@dataclass
class Phase1Report:
    """Final report of Phase-1 processing."""
    processed_files: int
    processed_documents: int
    errors: List[str]
    total_articles: int
    total_notes: int
    total_chars_clean: int
    processing_time_seconds: float
    timestamp_utc: str


class Phase1Processor:
    """Main orchestrator for Phase-1 document processing pipeline."""
    
    def __init__(self, in_dir: Optional[Path] = None, out_dir: Optional[Path] = None):
        """
        Initialize the Phase-1 processor.
        
        Args:
            in_dir: Input directory for raw files. Defaults to RAW_DATA_DIR.
            out_dir: Output directory for processed files. Defaults to PROCESSED_PHASE_1_DIR.
        """
        self.in_dir = Path(in_dir or RAW_DATA_DIR)
        self.out_dir = Path(out_dir or PROCESSED_PHASE_1_DIR)
        
        # Ensure directories exist
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline components
        self.legal_parser = LegalStructureParser()
        self.json_formatter = JsonFormatter(self.out_dir)
        
        # Processing state
        self.processing_results: List[ProcessingResult] = []
        self.all_documents: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        
        # Pipeline versions for tracking
        self.pipeline_versions = {
            "legal_structure_parser": "1.0.0",
            "json_formatter": "1.0.0",
            "main_processor": "1.0.0"
        }
        
        logger.info(f"Phase1Processor initialized")
        logger.info(f"Input directory: {self.in_dir}")
        logger.info(f"Output directory: {self.out_dir}")
    
    def run(self) -> Phase1Report:
        """
        Run the complete Phase-1 processing pipeline.
        
        Returns:
            Phase1Report with processing results
        """
        start_time = time.time()
        logger.info("Starting Phase-1 processing pipeline")
        
        try:
            # Get all input files
            input_files = self._get_input_files()
            logger.info(f"Found {len(input_files)} input files to process")
            
            if not input_files:
                logger.warning("No input files found to process")
                return self._create_report(start_time)
            
            # Process each file
            for file_path in sorted(input_files):
                self._process_file(file_path)
            
            # Update aggregate indexes
            self._update_aggregate_files()
            
            # Write processing logs
            self._write_processing_logs()
            
            logger.info("Phase-1 processing pipeline completed successfully")
            
        except Exception as e:
            error_msg = f"Critical error in Phase-1 pipeline: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
        
        return self._create_report(start_time)
    
    def _get_input_files(self) -> List[Path]:
        """Get all input files with supported extensions."""
        supported_extensions = {'.docx', '.doc', '.pdf', '.txt', '.json'}
        
        input_files = []
        for ext in supported_extensions:
            input_files.extend(self.in_dir.glob(f"*{ext}"))
        
        return sorted(input_files)
    
    def _process_file(self, file_path: Path) -> None:
        """Process a single input file."""
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Read file content
            content = self._read_file_content(file_path)
            if not content:
                logger.warning(f"Empty or unreadable file: {file_path.name}")
                return
            
            # For now, treat each file as a single document
            # In a full implementation, DocumentSplitter would be used here
            splits = [{"content": content, "metadata": {}}]
            
            # Process each split
            for split_idx, split in enumerate(splits):
                self._process_split(file_path, split_idx, split)
                
        except Exception as e:
            error_msg = f"Error processing file {file_path.name}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read content from a file."""
        try:
            if file_path.suffix.lower() in {'.txt', '.json'}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # For other formats, return a placeholder
                # In full implementation, proper document readers would be used
                return f"Content from {file_path.name}"
        except Exception as e:
            logger.error(f"Failed to read file {file_path.name}: {e}")
            return None
    
    def _process_split(self, file_path: Path, split_idx: int, split: Dict[str, Any]) -> None:
        """Process a single document split."""
        start_time = time.time()
        result = ProcessingResult(
            source_file=file_path.name,
            split_index=split_idx,
            success=False,
            duration_ms=0.0
        )
        
        try:
            content = split["content"]
            
            # Step 1: Clean text (simplified - in full implementation, TextCleaner would be used)
            clean_result = self._clean_text(content)
            
            # Step 2: Extract metadata (simplified - in full implementation, MetadataExtractor would be used)
            metadata = self._extract_metadata(content, file_path.name)
            
            # Step 3: Parse legal structure
            parsed_structure = self.legal_parser.parse(clean_result["normalized"])
            
            # Step 4: Build and save JSON
            doc_json = self.json_formatter.build_document_json(
                metadata=metadata,
                clean=clean_result,
                parsed=parsed_structure,
                source_file=file_path.name,
                processing_ms=(time.time() - start_time) * 1000,
                pipeline_versions=self.pipeline_versions
            )
            
            # Save document
            doc_path = self.json_formatter.save_document(doc_json, len(self.all_documents) + 1)
            
            # Update result
            result.success = True
            result.document_path = str(doc_path)
            result.metadata_title = metadata.title
            result.articles_count = parsed_structure.total_articles
            result.notes_count = parsed_structure.total_notes
            result.chars_clean = len(clean_result["normalized"])
            
            # Add to collections
            self.all_documents.append(doc_json)
            
            logger.info(f"Successfully processed split {split_idx} from {file_path.name}")
            
        except Exception as e:
            error_msg = f"Error processing split {split_idx} from {file_path.name}: {e}"
            logger.error(error_msg)
            result.error_message = error_msg
            self.errors.append(error_msg)
        
        finally:
            result.duration_ms = (time.time() - start_time) * 1000
            self.processing_results.append(result)
    
    def _clean_text(self, content: str) -> Dict[str, str]:
        """Clean and normalize text (simplified implementation)."""
        # In full implementation, this would use TextCleaner
        normalized = content.strip()
        
        # Convert Persian digits to ASCII
        ascii_digits = self._persian_to_ascii_digits(normalized)
        
        # Keep Persian digits version
        persian_digits = normalized
        
        return {
            "original_text": content,
            "normalized_text": normalized,
            "ascii_digits_text": ascii_digits,
            "persian_digits_text": persian_digits
        }
    
    def _persian_to_ascii_digits(self, text: str) -> str:
        """Convert Persian digits to ASCII digits."""
        persian_to_ascii = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
        
        for persian, ascii_digit in persian_to_ascii.items():
            text = text.replace(persian, ascii_digit)
        
        return text
    
    def _extract_metadata(self, content: str, filename: str) -> Any:
        """Extract metadata from content (simplified implementation)."""
        # In full implementation, this would use MetadataExtractor
        # For now, create a simple metadata object
        
        class SimpleMetadata:
            def __init__(self):
                self.title = f"Document from {filename}"
                self.document_type = "قانون"
                self.approval_authority = "مجلس شورای اسلامی"
                self.approval_date = ""
                self.effective_date = ""
                self.section_name = "بخش دوم - قوانین"
                self.document_number = ""
                self.subject = ""
                self.keywords = []
                self.related_docs = []
                self.confidence_score = 0.8
        
        return SimpleMetadata()
    
    def _update_aggregate_files(self) -> None:
        """Update aggregate index files."""
        if self.all_documents:
            self.json_formatter.update_indexes(self.all_documents)
            logger.info("Aggregate files updated successfully")
    
    def _write_processing_logs(self) -> None:
        """Write processing logs and validation reports."""
        # Write processing log
        processing_log = {
            "pipeline_info": {
                "version": "1.0.0",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "input_directory": str(self.in_dir),
                "output_directory": str(self.out_dir)
            },
            "processing_results": [asdict(result) for result in self.processing_results],
            "summary": {
                "total_files_processed": len(set(r.source_file for r in self.processing_results)),
                "total_splits_processed": len(self.processing_results),
                "successful_splits": len([r for r in self.processing_results if r.success]),
                "failed_splits": len([r for r in self.processing_results if not r.success])
            }
        }
        
        processing_log_path = self.out_dir / "processing_log.json"
        with open(processing_log_path, 'w', encoding='utf-8') as f:
            json.dump(processing_log, f, ensure_ascii=False, indent=2)
        
        # Write validation report
        validation_report = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "validation_errors": self.errors,
            "schema_issues": [],
            "file_validation": {
                "total_files": len(set(r.source_file for r in self.processing_results)),
                "valid_files": len([r for r in self.processing_results if r.success]),
                "invalid_files": len([r for r in self.processing_results if not r.success])
            }
        }
        
        validation_report_path = self.out_dir / "validation_report.json"
        with open(validation_report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        logger.info("Processing logs and validation reports written")
    
    def _create_report(self, start_time: float) -> Phase1Report:
        """Create the final processing report."""
        processing_time = time.time() - start_time
        
        successful_results = [r for r in self.processing_results if r.success]
        
        return Phase1Report(
            processed_files=len(set(r.source_file for r in self.processing_results)),
            processed_documents=len(successful_results),
            errors=self.errors,
            total_articles=sum(r.articles_count for r in successful_results),
            total_notes=sum(r.notes_count for r in successful_results),
            total_chars_clean=sum(r.chars_clean for r in successful_results),
            processing_time_seconds=processing_time,
            timestamp_utc=datetime.utcnow().isoformat() + "Z"
        )


def main():
    """Main entry point for CLI execution."""
    processor = Phase1Processor()
    report = processor.run()
    
    # Print report as JSON
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
