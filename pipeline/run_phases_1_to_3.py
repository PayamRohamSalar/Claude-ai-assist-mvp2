#!/usr/bin/env python3
"""
End-to-End Pipeline Runner for Legal Assistant AI (Phases 1-3)

This module orchestrates the complete pipeline execution from raw document processing
through database import to RAG vector store creation with Persian user-facing logs
and English docstrings.

Author: Legal Assistant AI Team
Version: 1.0.0
"""

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, Tuple

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import phase modules
from phase_1_data_processing.main_processor import Phase1Processor
from phase_2_database.database_creator import init_database
from phase_2_database.data_importer import process_single_document, ImportStats

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class PhasesPipeline:
    """
    End-to-end pipeline orchestrator for Legal Assistant AI phases 1-3.
    
    Provides idempotent execution with configurable start/end phases,
    Persian user logs, and comprehensive error handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file. Defaults to config/config.json.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/config.json"
        self.config = self._load_config()
        
        # Set up paths from config
        self.raw_dir = Path("data/raw")
        self.phase1_dir = Path("data/processed_phase_1")
        self.phase3_dir = Path("data/processed_phase_3")
        self.logs_dir = Path("logs")
        self.db_path = Path(self.config.get("database", {}).get("path", "data/db/legal_assistant.db"))
        self.schema_sql = Path("phase_2_database/schema.sql")
        
        # Create necessary directories
        self._ensure_directories()
        
        # Pipeline state
        self.start_time = None
        self.errors: List[str] = []
        self.artifacts: List[str] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise PipelineError(f"Error loading config: {e}")
    
    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        directories = [
            self.raw_dir,
            self.phase1_dir, 
            self.phase3_dir,
            self.logs_dir,
            self.db_path.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Verify write permissions
        for directory in directories:
            if not directory.exists() or not directory.is_dir():
                raise PipelineError(f"Directory not accessible: {directory}")
                
        self.logger.info("تمام پوشه‌های مورد نیاز بررسی و ایجاد شدند")
    
    def run(self, from_phase: str = "phase1", to_phase: str = "phase3", 
            rebuild: bool = False, db_path_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete pipeline from specified start to end phase.
        
        Args:
            from_phase: Starting phase (phase1, phase2, phase3)
            to_phase: Ending phase (phase1, phase2, phase3) 
            rebuild: Whether to purge outputs before running
            db_path_override: Override database path from config
            
        Returns:
            Dict containing execution results and artifacts
        """
        self.start_time = time.time()
        
        try:
            # Override DB path if provided
            if db_path_override:
                self.db_path = Path(db_path_override)
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate phase arguments
            valid_phases = ["phase1", "phase2", "phase3"]
            if from_phase not in valid_phases or to_phase not in valid_phases:
                raise PipelineError(f"Invalid phase. Valid phases: {valid_phases}")
                
            phase_order = {"phase1": 1, "phase2": 2, "phase3": 3}
            if phase_order[from_phase] > phase_order[to_phase]:
                raise PipelineError("Start phase cannot be after end phase")
            
            self.logger.info(f"شروع اجرای خط تولید از {from_phase} تا {to_phase}")
            if rebuild:
                self.logger.info("حالت بازسازی فعال - خروجی‌های موجود حذف خواهند شد")
            
            # Execute phases sequentially
            executed_phases = []
            
            if phase_order[from_phase] <= 1 <= phase_order[to_phase]:
                self._run_phase1(rebuild)
                executed_phases.append("phase1")
                
            if phase_order[from_phase] <= 2 <= phase_order[to_phase]:
                self._run_phase2(rebuild)
                executed_phases.append("phase2")
                
            if phase_order[from_phase] <= 3 <= phase_order[to_phase]:
                self._run_phase3(rebuild)
                executed_phases.append("phase3")
            
            # Generate final report
            duration = time.time() - self.start_time
            
            report = {
                "success": True,
                "executed_phases": executed_phases,
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "artifacts": self.artifacts,
                "errors": self.errors,
                "config_used": self.config_path,
                "db_path": str(self.db_path)
            }
            
            # Write pipeline report
            report_path = self.logs_dir / "pipeline_phase1_3.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # Success message in Persian
            self.logger.info(f"خط تولید با موفقیت تکمیل شد در {duration:.2f} ثانیه")
            self._log_artifacts()
            
            return report
            
        except Exception as e:
            error_msg = f"خطای حین اجرای خط تولید: {e}"
            self.logger.error(error_msg)
            self.errors.append(str(e))
            
            # Generate error report
            duration = time.time() - (self.start_time or time.time())
            report = {
                "success": False,
                "error": str(e),
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.utcnow().isoformat() + "Z", 
                "artifacts": self.artifacts,
                "errors": self.errors
            }
            
            return report
    
    def _run_phase1(self, rebuild: bool = False) -> None:
        """
        Execute Phase 1: Document processing and JSON generation.
        
        Args:
            rebuild: Whether to purge Phase 1 outputs before processing
        """
        self.logger.info("شروع فاز ۱: پردازش اسناد و تولید JSON")
        
        try:
            # Handle rebuild
            if rebuild and self.phase1_dir.exists():
                self.logger.info("حذف خروجی‌های فاز ۱ موجود...")
                shutil.rmtree(self.phase1_dir)
                self.phase1_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if phase already completed (unless rebuild)
            if not rebuild and self._is_phase1_complete():
                self.logger.info("فاز ۱ قبلاً تکمیل شده - رد شدن...")
                return
                
            # Initialize and run Phase 1 processor
            processor = Phase1Processor(self.raw_dir, self.phase1_dir)
            report = processor.run()
            
            # Add artifacts
            self.artifacts.extend([
                str(self.phase1_dir / "documents_metadata.json"),
                str(self.phase1_dir / "statistics.json"), 
                str(self.phase1_dir / "processing_log.json"),
                str(self.phase1_dir / "validation_report.json")
            ])
            
            # Add processed document JSONs
            for json_file in self.phase1_dir.glob("*.json"):
                if json_file.name not in ["documents_metadata.json", "statistics.json", 
                                         "processing_log.json", "validation_report.json"]:
                    self.artifacts.append(str(json_file))
            
            self.logger.info(f"فاز ۱ تکمیل شد - {report.processed_documents} سند پردازش شد")
            
        except Exception as e:
            raise PipelineError(f"Error in Phase 1: {e}") from e
    
    def _run_phase2(self, rebuild: bool = False) -> None:
        """
        Execute Phase 2: Database creation and data import.
        
        Args:
            rebuild: Whether to recreate database before importing
        """
        self.logger.info("شروع فاز ۲: ایجاد پایگاه داده و واردات داده‌ها")
        
        try:
            # Handle rebuild
            recreate_db = rebuild
            if rebuild and self.db_path.exists():
                self.logger.info("حذف پایگاه داده موجود...")
                self.db_path.unlink()
                recreate_db = True
            
            # Check if phase already completed (unless rebuild)
            if not rebuild and self._is_phase2_complete():
                self.logger.info("فاز ۲ قبلاً تکمیل شده - رد شدن...")
                return
            
            # Initialize database with schema
            self.logger.info("ایجاد و راه‌اندازی پایگاه داده...")
            connection = init_database(str(self.db_path), recreate_db)
            
            # Import Phase 1 data
            self.logger.info("واردات داده‌های فاز ۱ به پایگاه داده...")
            
            # Get all JSON files from Phase 1
            json_files = list(self.phase1_dir.glob("*.json"))
            document_files = [f for f in json_files if f.name not in [
                "documents_metadata.json", "statistics.json", 
                "processing_log.json", "validation_report.json"
            ]]
            
            import_count = 0
            for json_file in document_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        document_data = json.load(f)
                    
                    # Use the process_single_document function
                    stats = process_single_document(document_data, str(json_file), connection)
                    if stats.documents_inserted > 0 or stats.documents_updated > 0:
                        import_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"خطا در واردات {json_file.name}: {e}")
                    self.errors.append(f"Import error for {json_file.name}: {e}")
            
            connection.close()
            
            # Add database to artifacts
            self.artifacts.append(str(self.db_path))
            
            # Write import report
            import_report = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "imported_documents": import_count,
                "total_files_processed": len(document_files),
                "database_path": str(self.db_path)
            }
            
            report_path = self.logs_dir / "phase2_import_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(import_report, f, ensure_ascii=False, indent=2)
                
            self.artifacts.append(str(report_path))
            
            self.logger.info(f"فاز ۲ تکمیل شد - {import_count} سند وارد پایگاه داده شد")
            
        except Exception as e:
            raise PipelineError(f"Error in Phase 2: {e}") from e
    
    def _run_phase3(self, rebuild: bool = False) -> None:
        """
        Execute Phase 3: RAG pipeline with chunking, embeddings, and vector store.
        
        Args:
            rebuild: Whether to purge Phase 3 outputs before processing
        """
        self.logger.info("شروع فاز ۳: خط تولید RAG و ایجاد فروشگاه بردار")
        
        try:
            # Handle rebuild
            if rebuild and self.phase3_dir.exists():
                self.logger.info("حذف خروجی‌های فاز ۳ موجود...")
                shutil.rmtree(self.phase3_dir)
                self.phase3_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if phase already completed (unless rebuild)
            if not rebuild and self._is_phase3_complete():
                self.logger.info("فاز ۳ قبلاً تکمیل شده - رد شدن...")
                return
            
            # Determine vector store backend from config
            backend = self.config.get("rag", {}).get("vector_backend", "faiss")
            
            # Step 1: Run chunker
            self.logger.info("گام ۱: تقسیم‌بندی اسناد...")
            chunker_result = self._run_chunker()
            
            # Step 2: Generate embeddings  
            self.logger.info("گام ۲: تولید جاسازی‌ها...")
            embedding_result = self._run_embedding_generator()
            
            # Step 3: Build vector store
            self.logger.info("گام ۳: ساخت فروشگاه بردار...")
            vector_store_result = self._run_vector_store_builder(backend)
            
            # Step 4: Run quality evaluation
            self.logger.info("گام ۴: ارزیابی کیفیت...")
            quality_result = self._run_quality_evaluation()
            
            # Add Phase 3 artifacts
            phase3_artifacts = [
                self.phase3_dir / "chunks.json",
                self.phase3_dir / "chunking_config.json", 
                self.phase3_dir / "embeddings.npy",
                self.phase3_dir / "embeddings_meta.json",
                self.phase3_dir / "vector_db",
                self.phase3_dir / "embedding_report.json",
                self.phase3_dir / "retrieval_sanity.json"
            ]
            
            for artifact in phase3_artifacts:
                if artifact.exists():
                    self.artifacts.append(str(artifact))
            
            self.logger.info("فاز ۳ تکمیل شد - فروشگاه بردار آماده استفاده است")
            
        except Exception as e:
            raise PipelineError(f"Error in Phase 3: {e}") from e
    
    def _run_chunker(self) -> Dict[str, Any]:
        """Run the document chunker programmatically."""
        try:
            # Import chunker functions
            from phase_3_rag.chunker import (
                load_chunking_config, 
                create_chunks_from_database, 
                write_chunks_json
            )
            
            # Load configuration
            config = load_chunking_config()
            
            # Create chunks from database
            chunks = create_chunks_from_database(
                db_path=str(self.db_path),
                config=config
            )
            
            if not chunks:
                raise PipelineError("No chunks were created from database")
            
            # Write chunks to output directory
            chunks_output_path = self.phase3_dir / "chunks.json"
            write_chunks_json(chunks, str(chunks_output_path))
            
            # Save the chunking configuration used
            config_file = self.phase3_dir / "chunking_config.json"
            import json
            from dataclasses import asdict
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(config), f, ensure_ascii=False, indent=2)
            
            # Verify outputs
            if not chunks_output_path.exists() or not config_file.exists():
                raise PipelineError("Chunker outputs were not generated")
            
            return {"success": True, "chunks_file": str(chunks_output_path), "chunks_count": len(chunks)}
            
        except Exception as e:
            raise PipelineError(f"Error in chunker: {e}") from e
    
    def _run_embedding_generator(self) -> Dict[str, Any]:
        """Run the embedding generator programmatically."""
        try:
            # Run as subprocess for better isolation
            embedding_script = Path(__file__).parent.parent / "phase_3_rag" / "embedding_generator.py"
            cmd = [
                sys.executable, 
                str(embedding_script),
                "--chunks-file", str(self.phase3_dir / "chunks.json"),
                "--output-dir", str(self.phase3_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                raise PipelineError(f"Embedding generator failed: {result.stderr}")
            
            # Verify outputs
            embeddings_file = self.phase3_dir / "embeddings.npy"
            meta_file = self.phase3_dir / "embeddings_meta.json"
            
            if not embeddings_file.exists() or not meta_file.exists():
                raise PipelineError("خروجی‌های embedding generator تولید نشدند")
            
            return {"success": True, "embeddings_file": str(embeddings_file)}
            
        except Exception as e:
            raise PipelineError(f"Error in embedding generator: {e}") from e
    
    def _run_vector_store_builder(self, backend: str) -> Dict[str, Any]:
        """Run the vector store builder programmatically."""
        try:
            vector_store_script = Path(__file__).parent.parent / "phase_3_rag" / "vector_store_builder.py"
            cmd = [
                sys.executable,
                str(vector_store_script), 
                "--embeddings-file", str(self.phase3_dir / "embeddings.npy"),
                "--metadata-file", str(self.phase3_dir / "embeddings_meta.json"),
                "--output-dir", str(self.phase3_dir / "vector_db"),
                "--backend", backend
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                raise PipelineError(f"Vector store builder failed: {result.stderr}")
            
            # Verify output directory exists
            vector_db_dir = self.phase3_dir / "vector_db"
            if not vector_db_dir.exists():
                raise PipelineError("دایرکتوری vector store تولید نشد")
            
            return {"success": True, "vector_db_dir": str(vector_db_dir)}
            
        except Exception as e:
            raise PipelineError(f"Error in vector store builder: {e}") from e
    
    def _run_quality_evaluation(self) -> Dict[str, Any]:
        """Run quality evaluation programmatically."""
        try:
            quality_eval_script = Path(__file__).parent.parent / "phase_3_rag" / "quality_eval.py"
            cmd = [
                sys.executable,
                str(quality_eval_script),
                "--vector-db-dir", str(self.phase3_dir / "vector_db"), 
                "--chunks-file", str(self.phase3_dir / "chunks.json"),
                "--output-dir", str(self.phase3_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                raise PipelineError(f"Quality evaluation failed: {result.stderr}")
            
            # Verify outputs
            embedding_report = self.phase3_dir / "embedding_report.json"
            retrieval_sanity = self.phase3_dir / "retrieval_sanity.json"
            
            if not embedding_report.exists() or not retrieval_sanity.exists():
                raise PipelineError("گزارش‌های کیفیت تولید نشدند")
            
            return {"success": True, "reports_generated": 2}
            
        except Exception as e:
            raise PipelineError(f"Error in quality evaluation: {e}") from e
    
    def _is_phase1_complete(self) -> bool:
        """Check if Phase 1 has been completed."""
        required_files = [
            "documents_metadata.json",
            "statistics.json", 
            "processing_log.json",
            "validation_report.json"
        ]
        
        for filename in required_files:
            if not (self.phase1_dir / filename).exists():
                return False
        
        # Check if we have at least one processed document
        doc_files = [f for f in self.phase1_dir.glob("*.json") 
                    if f.name not in required_files]
        
        return len(doc_files) > 0
    
    def _is_phase2_complete(self) -> bool:
        """Check if Phase 2 has been completed."""
        if not self.db_path.exists():
            return False
            
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except sqlite3.Error:
            return False
    
    def _is_phase3_complete(self) -> bool:
        """Check if Phase 3 has been completed."""
        required_files = [
            "chunks.json",
            "chunking_config.json",
            "embeddings.npy", 
            "embeddings_meta.json",
            "embedding_report.json",
            "retrieval_sanity.json"
        ]
        
        for filename in required_files:
            if not (self.phase3_dir / filename).exists():
                return False
        
        # Check vector_db directory exists
        return (self.phase3_dir / "vector_db").exists()
    
    def _log_artifacts(self) -> None:
        """Log all generated artifacts in Persian."""
        self.logger.info("فهرست تولیدات نهایی:")
        for artifact in self.artifacts:
            artifact_path = Path(artifact)
            if artifact_path.exists():
                if artifact_path.is_file():
                    size = artifact_path.stat().st_size
                    self.logger.info(f"  📄 {artifact} ({size:,} بایت)")
                else:
                    self.logger.info(f"  📁 {artifact}/")
            else:
                self.logger.warning(f"  ❌ {artifact} (یافت نشد)")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser with help text."""
    parser = argparse.ArgumentParser(
        description="Run complete Legal Assistant AI pipeline from phases 1-3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python pipeline/run_phases_1_to_3.py                    # Complete run from phase 1-3
  python pipeline/run_phases_1_to_3.py --from phase2      # Start from phase 2
  python pipeline/run_phases_1_to_3.py --to phase1        # Only phase 1
  python pipeline/run_phases_1_to_3.py --rebuild          # Rebuild all outputs
        """
    )
    
    parser.add_argument(
        '--from', 
        dest='from_phase',
        choices=['phase1', 'phase2', 'phase3'],
        default='phase1',
        help='Starting phase (default: phase1)'
    )
    
    parser.add_argument(
        '--to',
        dest='to_phase', 
        choices=['phase1', 'phase2', 'phase3'],
        default='phase3',
        help='Ending phase (default: phase3)'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Remove existing outputs before running'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.json',
        help='Configuration file path (default: config/config.json)'
    )
    
    parser.add_argument(
        '--db-path',
        help='Override database path (otherwise use config)'
    )
    
    return parser


def main():
    """Main entry point for CLI execution."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = PhasesPipeline(args.config)
        
        # Run pipeline
        result = pipeline.run(
            from_phase=args.from_phase,
            to_phase=args.to_phase,
            rebuild=args.rebuild,
            db_path_override=args.db_path
        )
        
        if result["success"]:
            print(f"\n[SUCCESS] Pipeline completed successfully!")
            print(f"Duration: {result['duration_seconds']} seconds")
            print(f"Artifacts generated: {len(result['artifacts'])}")
            
            if result.get("artifacts"):
                print("\nGenerated artifacts:")
                for artifact in result["artifacts"]:
                    print(f"  - {artifact}")
            
            sys.exit(0)
        else:
            print(f"\n[ERROR] Pipeline failed: {result.get('error', 'Unknown error')}")
            if result.get("errors"):
                print("\nError details:")
                for error in result["errors"]:
                    print(f"  - {error}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Execution stopped by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()