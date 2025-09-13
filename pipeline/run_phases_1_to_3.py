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
import hashlib
import json
import logging
import os
import zipfile
import xml.etree.ElementTree as ET
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, Tuple, Iterable, Set

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import phase modules
from phase_1_data_processing.main_processor import Phase1Processor
from phase_2_database.database_creator import init_database
from phase_2_database.data_importer import process_single_document, ImportStats

# Reuse helpers from Phase 3 chunker where possible
from phase_3_rag.chunker import (
    load_chunking_config,
    normalize_text as chunk_normalize_text,
    split_text_into_chunks,
    compute_chunk_uid,
    estimate_token_count,
)

# Text/metadata extraction on raw files
from shared_utils.file_utils import DocumentReader
from phase_1_data_processing.metadata_extractor import PersianMetadataExtractor

try:
    import numpy as np
except Exception:
    np = None  # Will validate at runtime if embeddings are requested

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # Optional until embeddings step

try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

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
        
        # Incremental processing state
        self.embedding_model_name = (
            self.config.get("rag", {}).get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
        )
        self.vector_db_dir = self.phase3_dir / "vector_db" / "faiss"
        
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

    # =============== Incremental multi-file processing ===============
    def run_incremental(
        self,
        raw_dir: Path,
        force: bool = False,
        workers: int = 1,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Incrementally process .docx files end-to-end: text->normalize->chunk->embed->FAISS->SQLite.
        """
        start_ts = time.time()
        raw_dir = Path(raw_dir)
        self.logger.info(f"Raw dir: {str(raw_dir.resolve())}")
        self.logger.info(f"Phase3 dir: {str(self.phase3_dir.resolve())}")
        self.logger.info(f"DB path: {str(self.db_path.resolve())}")
        self.logger.info(f"Vector DB dir: {str(self.vector_db_dir.resolve())}")

        self.phase3_dir.mkdir(parents=True, exist_ok=True)
        (self.phase3_dir / "chunks_by_doc").mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)

        existing_doc_uids: Set[str] = self._load_existing_document_uids()

        doc_paths = sorted(raw_dir.glob("*.docx"))
        if not doc_paths:
            self.logger.info("هیچ فایل DOCX برای پردازش یافت نشد")
            return {
                "success": True,
                "processed_docs": 0,
                "processed_chunks": 0,
                "updated_chunks": 0,
                "duration_seconds": round(time.time() - start_ts, 2),
            }

        # Stage 1: per-file text+metadata+chunking (parallelizable)
        per_doc_results: List[Dict[str, Any]] = []

        def _process_one(file_path: Path) -> Dict[str, Any]:
            t0 = time.time()
            document_uid = self._compute_document_uid(file_path)
            skipped = False
            was_update = document_uid in existing_doc_uids
            prev_count = 0
            if (not force) and was_update:
                skipped = True
                return {
                    "document_uid": document_uid,
                    "file": str(file_path),
                    "chunks": [],
                    "skipped": True,
                    "was_update": was_update,
                    "prev_count": 0,
                    "elapsed": round(time.time() - t0, 3),
                }

            # Extract text
            reader = DocumentReader()
            r = reader.read_document(file_path)
            if not r.get("success"):
                # Fallback for DOCX when python-docx not available
                if file_path.suffix.lower() == ".docx" and "docx" in str(r.get("error", "")).lower():
                    text = self._read_docx_fallback(file_path)
                else:
                    raise PipelineError(f"Read failure: {r.get('error')}")
            else:
                text = (r.get("content") or "").strip()
            if not text:
                return {
                    "document_uid": document_uid,
                    "file": str(file_path),
                    "chunks": [],
                    "skipped": True,
                    "prev_count": 0,
                    "elapsed": round(time.time() - t0, 3),
                }

            # Metadata
            meta_extractor = PersianMetadataExtractor()
            md = meta_extractor.extract_metadata(text, document_id=document_uid)
            title = md.title or file_path.stem

            # Normalize and chunk
            cfg = load_chunking_config()
            normalized_all = chunk_normalize_text(text, cfg)
            raw_chunks = split_text_into_chunks(normalized_all, cfg)

            chunks: List[Dict[str, Any]] = []
            for idx, chunk_text in enumerate(raw_chunks):
                norm = chunk_normalize_text(chunk_text, cfg)
                uid = compute_chunk_uid(
                    document_uid=document_uid,
                    article_number="NA",
                    note_label="NA",
                    chunk_index=idx,
                    normalized_text=norm,
                )
                chunks.append({
                    "chunk_uid": uid,
                    "document_uid": document_uid,
                    "document_title": title,
                    "document_type": md.document_type or "",
                    "section": md.section_label or "",
                    "approval_date": md.approval_date or "",
                    "approval_authority": md.approval_authority or "",
                    "chapter_title": "",
                    "article_number": "",
                    "note_label": "",
                    "clause_label": "",
                    "text": chunk_text,
                    "normalized_text": norm,
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    "token_count": estimate_token_count(chunk_text),
                    "source_type": "article" if not chunk_text.strip().startswith("تبصره") else "note",
                    "metadata": {
                        "source_file": file_path.name,
                        "abs_path": str(file_path.resolve()),
                    },
                })

            elapsed = round(time.time() - t0, 3)
            return {
                "document_uid": document_uid,
                "file": str(file_path),
                "title": title,
                "chunks": chunks,
                "skipped": skipped,
                "was_update": was_update,
                "prev_count": prev_count,
                "elapsed": elapsed,
            }

        if workers and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_process_one, p): p for p in doc_paths}
                for fut in as_completed(futs):
                    per_doc_results.append(fut.result())
        else:
            for p in doc_paths:
                per_doc_results.append(_process_one(p))

        # Stage 2: merge chunks per doc into per-doc files and combined chunks.json
        processed_doc_uids = [r["document_uid"] for r in per_doc_results if r.get("chunks")]
        total_new_chunks = sum(len(r.get("chunks", [])) for r in per_doc_results)

        if not dry_run:
            for r in per_doc_results:
                if not r.get("chunks"):
                    continue
                self._write_chunks_per_doc(r["document_uid"], r["chunks"])  
            self._merge_chunks_json(processed_doc_uids)

        # Stage 3: embeddings append
        embedding_dim = 0
        emb_added = 0
        if total_new_chunks > 0:
            if SentenceTransformer is None or np is None:
                raise PipelineError("sentence-transformers and numpy are required for embeddings")
            texts = [c["text"] or c["normalized_text"] for r in per_doc_results for c in r.get("chunks", [])]
            uids = [c["chunk_uid"] for r in per_doc_results for c in r.get("chunks", [])]
            if not dry_run:
                embedding_dim = self._append_embeddings(texts, uids)
                emb_added = len(texts)

        # Stage 4: rebuild FAISS index from embeddings
        if not dry_run and emb_added > 0:
            if not HAS_FAISS:
                self.logger.warning("FAISS not available; skipping index rebuild")
            else:
                self._rebuild_faiss_index()

        # Stage 5: upsert into SQLite `chunks`
        upserts = 0
        if not dry_run and total_new_chunks > 0:
            upserts = self._upsert_chunks_into_db(per_doc_results)

        # Per-doc logging
        for r in per_doc_results:
            created_count = len(r.get("chunks", [])) if not r.get("was_update") else 0
            updated_count = len(r.get("chunks", [])) if r.get("was_update") else 0
            self.logger.info(
                f"Doc {r.get('title', Path(r['file']).stem)} | uid={r['document_uid']} | "
                f"created={created_count} updated={updated_count} | chunks={len(r.get('chunks', []))} | "
                f"skipped={r.get('skipped', False)} | elapsed={r.get('elapsed', 0)}s"
            )

        duration = round(time.time() - start_ts, 2)
        self.logger.info(
            f"Processed docs: {len(per_doc_results)} | new_docs={len(processed_doc_uids)} | "
            f"new_chunks={total_new_chunks} | embeddings_dim={embedding_dim} | upserts={upserts} | "
            f"elapsed={duration}s"
        )

        return {
            "success": True,
            "processed_docs": len(per_doc_results),
            "new_docs": len(processed_doc_uids),
            "new_chunks": total_new_chunks,
            "embeddings_dim": embedding_dim,
            "db_upserts": upserts,
            "duration_seconds": duration,
        }

    def _compute_document_uid(self, file_path: Path) -> str:
        st = file_path.stat()
        basis = f"{file_path.name}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    def _read_docx_fallback(self, file_path: Path) -> str:
        # Minimal DOCX text extractor using zip + XML parsing
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_bytes = z.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in root.findall('.//w:p', ns):
                runs = []
                for t in p.findall('.//w:t', ns):
                    if t.text:
                        runs.append(t.text)
                if runs:
                    paragraphs.append(''.join(runs))
            return '\n\n'.join(paragraphs)
        except Exception as e:
            raise PipelineError(f"DOCX fallback read failed: {e}")

    def _load_existing_document_uids(self) -> Set[str]:
        chunks_file = self.phase3_dir / "chunks.json"
        doc_uids: Set[str] = set()
        if chunks_file.exists():
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chunks = data.get("chunks", data if isinstance(data, list) else [])
                for c in chunks:
                    uid = c.get("document_uid")
                    if uid:
                        doc_uids.add(uid)
            except Exception as e:
                self.logger.warning(f"Failed to read existing chunks.json: {e}")
        return doc_uids

    def _write_chunks_per_doc(self, document_uid: str, chunks: List[Dict[str, Any]]) -> None:
        per_doc_dir = self.phase3_dir / "chunks_by_doc"
        per_doc_dir.mkdir(parents=True, exist_ok=True)
        out = per_doc_dir / f"{document_uid}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

    def _merge_chunks_json(self, document_uids: Iterable[str]) -> None:
        chunks_file = self.phase3_dir / "chunks.json"
        # Load existing
        existing_chunks: List[Dict[str, Any]] = []
        if chunks_file.exists():
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                existing_chunks = data.get("chunks", data if isinstance(data, list) else [])
            except Exception as e:
                self.logger.warning(f"Failed reading existing chunks.json: {e}")
        # Remove any chunks for document_uids
        doc_uid_set = set(document_uids)
        existing_chunks = [c for c in existing_chunks if c.get("document_uid") not in doc_uid_set]
        # Append per-doc chunks
        per_doc_dir = self.phase3_dir / "chunks_by_doc"
        for uid in doc_uid_set:
            p = per_doc_dir / f"{uid}.json"
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    existing_chunks.extend(data.get("chunks", []))
                except Exception as e:
                    self.logger.warning(f"Failed merging {p.name}: {e}")
        # Write merged with stats
        stats = {
            "total_chunks": len(existing_chunks),
            "documents_represented": len(set(c.get("document_uid") for c in existing_chunks)),
        }
        with chunks_file.open("w", encoding="utf-8") as f:
            json.dump({"chunks": existing_chunks, "statistics": stats}, f, ensure_ascii=False, indent=2)

    def _append_embeddings(self, texts: List[str], chunk_uids: List[str]) -> int:
        model = SentenceTransformer(self.embedding_model_name)
        embeddings = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]

        emb_path = self.phase3_dir / "embeddings.npy"
        meta_path = self.phase3_dir / "embeddings_meta.json"

        if emb_path.exists():
            existing = np.load(emb_path)
            if existing.shape[1] != dim:
                raise PipelineError(f"Embedding dimension mismatch: {existing.shape[1]} != {dim}")
            combined = np.concatenate([existing, embeddings], axis=0)
        else:
            combined = embeddings

        # Write embeddings
        np.save(emb_path, combined)

        # Update meta
        meta = {"count": int(combined.shape[0]), "dimension": int(dim), "chunk_uid_order": []}
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta_old = json.load(f)
                # preserve previous order
                meta["chunk_uid_order"] = meta_old.get("chunk_uid_order", [])
            except Exception:
                pass
        meta["chunk_uid_order"].extend(chunk_uids)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Track artifacts
        self.artifacts.extend([str(emb_path), str(meta_path)])
        return dim

    def _rebuild_faiss_index(self) -> None:
        emb_path = self.phase3_dir / "embeddings.npy"
        meta_path = self.phase3_dir / "embeddings_meta.json"
        if not emb_path.exists() or not meta_path.exists():
            raise PipelineError("Embeddings not found for FAISS index rebuild")
        embeddings = np.load(emb_path).astype(np.float32)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        d = int(meta.get("dimension", embeddings.shape[1]))
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        # Write index and mapping
        out_dir = self.vector_db_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out_dir / "faiss.index"))
        mapping = {"ntotal": int(index.ntotal), "dimension": int(d), "chunk_uid_order": meta.get("chunk_uid_order", [])}
        with (out_dir / "mapping.json").open("w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        self.artifacts.extend([str(out_dir / "faiss.index"), str(out_dir / "mapping.json")])

    def _upsert_chunks_into_db(self, per_doc_results: List[Dict[str, Any]]) -> int:
        # Connect directly without executing full schema to avoid incompatibilities
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        # Ensure table exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_uid TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_uid TEXT NOT NULL,
                text TEXT,
                normalized_text TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_uid, chunk_index)
            )
            """
        )
        conn.commit()

        total = 0
        for r in per_doc_results:
            chunks = r.get("chunks", [])
            if not chunks:
                continue
            # Replace all rows for this document_uid to keep chunk_index uniqueness stable
            cur.execute("DELETE FROM chunks WHERE document_uid = ?", (r["document_uid"],))
            for c in chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (document_uid, chunk_index, chunk_uid, text, normalized_text, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_uid, chunk_index) DO UPDATE SET
                        chunk_uid=excluded.chunk_uid,
                        text=excluded.text,
                        normalized_text=excluded.normalized_text,
                        metadata_json=excluded.metadata_json,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        c["document_uid"],
                        int(c["chunk_index"]),
                        c["chunk_uid"],
                        c.get("text", ""),
                        c.get("normalized_text", ""),
                        json.dumps(c.get("metadata", {}), ensure_ascii=False),
                    ),
                )
                total += 1
        conn.commit()
        conn.close()
        return total
    
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
        description="Run Legal Assistant AI pipeline (incremental multi-file or legacy phases 1-3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Incremental multi-file processing (default)
  python pipeline/run_phases_1_to_3.py --raw_dir data/raw --workers 4

  # Legacy phases 1-3 (kept for compatibility)
  python pipeline/run_phases_1_to_3.py --from phase2 --to phase3
  python pipeline/run_phases_1_to_3.py --rebuild
        """
    )
    
    # Incremental mode flags
    parser.add_argument('--raw_dir', default='data/raw', help='Directory containing raw .docx files (default: data/raw)')
    parser.add_argument('--force', action='store_true', help='Reprocess documents even if seen (default: false)')
    parser.add_argument('--workers', type=int, default=1, help='Parallel worker count for per-file processing')
    parser.add_argument('--dry-run', action='store_true', help='Plan only, perform no writes')

    # Legacy flags (kept)
    parser.add_argument('--from', dest='from_phase', choices=['phase1', 'phase2', 'phase3'], default='phase1', help='Legacy: start phase')
    parser.add_argument('--to', dest='to_phase', choices=['phase1', 'phase2', 'phase3'], default='phase3', help='Legacy: end phase')
    parser.add_argument('--rebuild', action='store_true', help='Legacy: remove outputs before running')
    parser.add_argument('--config', default='config/config.json', help='Configuration file path (default: config/config.json)')
    parser.add_argument('--db-path', help='Override database path (otherwise use config)')
    
    return parser


def main():
    """Main entry point for CLI execution."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        pipeline = PhasesPipeline(args.config)
        # Prefer incremental path by default; legacy can still be invoked explicitly
        inc_result = pipeline.run_incremental(
            raw_dir=Path(args.raw_dir),
            force=bool(args.force),
            workers=int(args.workers or 1),
            dry_run=bool(args.dry_run),
        )

        print("\n[SUCCESS] Incremental pipeline completed")
        print(f"Processed docs: {inc_result.get('processed_docs', 0)}")
        print(f"New chunks: {inc_result.get('new_chunks', 0)}")
        if inc_result.get('embeddings_dim'):
            print(f"Embedding dimension: {inc_result['embeddings_dim']}")
        print(f"Duration: {inc_result.get('duration_seconds', 0)} seconds")
        sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Execution stopped by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()