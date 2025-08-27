#!/usr/bin/env python3
"""
سازنده فروشگاه بردار برای سیستم RAG
Vector Store Builder for RAG System

این ماژول بردارهای متنی را بارگذاری کرده و فروشگاه بردار ایجاد می‌کند
با پشتیبانی از FAISS و Chroma

This module loads embeddings and creates vector stores
with support for FAISS and Chroma backends
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# Optional imports for backends
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_store_building.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VectorStoreBuilder:
    """
    کلاس سازنده فروشگاه بردار
    Vector Store Builder Class
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        راه‌اندازی سازنده فروشگاه بردار
        Initialize the vector store builder
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.rag_config = config.get('rag', {})
        
        # Backend configuration
        self.backend = self.rag_config.get('index_backend', 'faiss').lower()
        
        if self.backend not in ['faiss', 'chroma']:
            raise ValueError(
                f"پشتیبان نامعتبر: {self.backend}. پشتیبان‌های معتبر: faiss, chroma\n"
                f"Invalid backend: {self.backend}. Valid backends: faiss, chroma"
            )
        
        # Check backend availability
        if self.backend == 'faiss' and not HAS_FAISS:
            raise ImportError(
                "خطا: کتابخانه faiss در دسترس نیست.\n"
                "Error: faiss library is not available.\n"
                "برای نصب از دستور زیر استفاده کنید:\n"
                "To install, use: pip install faiss-cpu"
            )
        
        if self.backend == 'chroma' and not HAS_CHROMA:
            raise ImportError(
                "خطا: کتابخانه chromadb در دسترس نیست.\n"
                "Error: chromadb library is not available.\n"
                "برای نصب از دستور زیر استفاده کنید:\n"
                "To install, use: pip install chromadb"
            )
        
        # Paths
        self.embeddings_path = Path("data/processed_phase_3/embeddings.npy")
        self.meta_path = Path("data/processed_phase_3/embeddings_meta.json")
        self.chunks_path = Path("data/processed_phase_3/chunks.json")
        self.vector_db_dir = Path("data/processed_phase_3/vector_db")
        
        # Ensure output directory exists
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.chunks: Optional[List[Dict[str, Any]]] = None
        
    def _load_data(self) -> None:
        """
        بارگذاری داده‌های مورد نیاز
        Load required data
        """
        logger.info("در حال بارگذاری داده‌های مورد نیاز...")
        logger.info("Loading required data...")
        
        # Load embeddings
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"فایل بردارها یافت نشد: {self.embeddings_path}\n"
                f"Embeddings file not found: {self.embeddings_path}\n"
                "لطفاً ابتدا embedding_generator.py را اجرا کنید.\n"
                "Please run embedding_generator.py first."
            )
        
        self.embeddings = np.load(self.embeddings_path)
        logger.info(f"بردارها بارگذاری شد: شکل {self.embeddings.shape}")
        logger.info(f"Embeddings loaded: shape {self.embeddings.shape}")
        
        # Load metadata
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"فایل متادیتای بردارها یافت نشد: {self.meta_path}\n"
                f"Embeddings metadata file not found: {self.meta_path}"
            )
        
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        logger.info(f"متادیتا بارگذاری شد: {self.metadata['count']} بردار با {self.metadata['dimension']} بعد")
        logger.info(f"Metadata loaded: {self.metadata['count']} embeddings with {self.metadata['dimension']} dimensions")
        
        # Load chunks
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"فایل قطعات یافت نشد: {self.chunks_path}\n"
                f"Chunks file not found: {self.chunks_path}"
            )
        
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        if isinstance(chunks_data, dict):
            self.chunks = chunks_data.get('chunks', [])
        elif isinstance(chunks_data, list):
            self.chunks = chunks_data
        else:
            raise ValueError("فرمت نامعتبر برای فایل chunks.json / Invalid format for chunks.json")
        
        logger.info(f"قطعات بارگذاری شد: {len(self.chunks)} قطعه")
        logger.info(f"Chunks loaded: {len(self.chunks)} chunks")
        
        # Validate data consistency
        if len(self.chunks) != self.metadata['count']:
            raise ValueError(
                f"تعداد قطعات ({len(self.chunks)}) با تعداد بردارها ({self.metadata['count']}) مطابقت ندارد\n"
                f"Number of chunks ({len(self.chunks)}) doesn't match number of embeddings ({self.metadata['count']})"
            )
        
        if self.embeddings.shape[0] != self.metadata['count']:
            raise ValueError(
                f"تعداد ردیف‌های بردار ({self.embeddings.shape[0]}) با متادیتا ({self.metadata['count']}) مطابقت ندارد\n"
                f"Number of embedding rows ({self.embeddings.shape[0]}) doesn't match metadata ({self.metadata['count']})"
            )
    
    def _build_faiss_index(self) -> None:
        """
        ایجاد فهرست FAISS
        Build FAISS index
        """
        logger.info("در حال ایجاد فهرست FAISS...")
        logger.info("Building FAISS index...")
        
        start_time = time.time()
        
        # Get dimensions
        d = self.embeddings.shape[1]
        
        # Create index - using IndexFlatIP (Inner Product) for similarity search
        # Note: Inner Product is good for normalized embeddings
        index = faiss.IndexFlatIP(d)
        
        # Optional: Use IVF for larger datasets
        # For now, we'll stick with Flat index for simplicity and accuracy
        logger.info(f"ایجاد فهرست FAISS با {d} بعد (IndexFlatIP)")
        logger.info(f"Creating FAISS index with {d} dimensions (IndexFlatIP)")
        
        # Add embeddings to index
        # FAISS expects float32
        embeddings_float32 = self.embeddings.astype(np.float32)
        
        # Normalize embeddings for better similarity search with Inner Product
        faiss.normalize_L2(embeddings_float32)
        
        logger.info("در حال افزودن بردارها به فهرست...")
        logger.info("Adding embeddings to index...")
        
        index.add(embeddings_float32)
        
        build_time = time.time() - start_time
        logger.info(f"فهرست FAISS با موفقیت ایجاد شد در {build_time:.2f} ثانیه")
        logger.info(f"FAISS index built successfully in {build_time:.2f} seconds")
        logger.info(f"تعداد بردارهای فهرست شده: {index.ntotal}")
        logger.info(f"Number of indexed vectors: {index.ntotal}")
        
        # Create mapping from row ID to chunk UID
        # The order should match the insertion order
        chunk_uid_order = self.metadata.get('chunk_uid_order', [])
        if not chunk_uid_order:
            # Fallback: extract from chunks in order
            chunk_uid_order = []
            for chunk in self.chunks:
                uid = chunk.get('uid', chunk.get('chunk_uid', ''))
                if uid:
                    chunk_uid_order.append(uid)
        
        # Create mapping: rowid -> chunk_uid
        rowid_to_uid_mapping = {}
        for rowid, chunk_uid in enumerate(chunk_uid_order):
            rowid_to_uid_mapping[str(rowid)] = chunk_uid
        
        # Save index and mapping atomically
        faiss_dir = self.vector_db_dir / "faiss"
        faiss_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = faiss_dir / "faiss.index"
        mapping_path = faiss_dir / "mapping.json"
        
        # Use temporary files for atomic operation
        temp_index_path = faiss_dir / "faiss.index.tmp"
        temp_mapping_path = faiss_dir / "mapping.json.tmp"
        
        try:
            # Save index
            logger.info(f"در حال ذخیره فهرست FAISS در: {temp_index_path}")
            logger.info(f"Saving FAISS index to: {temp_index_path}")
            faiss.write_index(index, str(temp_index_path))
            
            # Save mapping
            logger.info(f"در حال ذخیره نقشه‌برداری در: {temp_mapping_path}")
            logger.info(f"Saving mapping to: {temp_mapping_path}")
            with open(temp_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(rowid_to_uid_mapping, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            if temp_index_path.exists():
                if index_path.exists():
                    index_path.unlink()
                temp_index_path.rename(index_path)
            
            if temp_mapping_path.exists():
                if mapping_path.exists():
                    mapping_path.unlink()
                temp_mapping_path.rename(mapping_path)
            
            logger.info("فایل‌های FAISS با موفقیت ذخیره شد")
            logger.info("FAISS files saved successfully")
            
        except Exception as e:
            # Cleanup temp files on error
            if temp_index_path.exists():
                temp_index_path.unlink()
            if temp_mapping_path.exists():
                temp_mapping_path.unlink()
            raise e
    
    def _build_chroma_index(self) -> None:
        """
        ایجاد فهرست Chroma
        Build Chroma index
        """
        logger.info("در حال ایجاد فهرست Chroma...")
        logger.info("Building Chroma index...")
        
        start_time = time.time()
        
        # Setup Chroma client
        chroma_dir = self.vector_db_dir / "chroma"
        
        # Remove existing directory for idempotent operation
        if chroma_dir.exists():
            logger.info("حذف فهرست موجود برای عملیات ایدمپوتنت")
            logger.info("Removing existing index for idempotent operation")
            shutil.rmtree(chroma_dir)
        
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Create persistent client
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Create or get collection
        collection_name = "legal_chunks"
        
        logger.info(f"ایجاد مجموعه Chroma: {collection_name}")
        logger.info(f"Creating Chroma collection: {collection_name}")
        
        # Delete collection if exists (for idempotent operation)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection doesn't exist
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Persian Legal Documents Chunks for RAG"}
        )
        
        # Prepare data for insertion
        chunk_uid_order = self.metadata.get('chunk_uid_order', [])
        if not chunk_uid_order:
            # Fallback: extract from chunks in order
            chunk_uid_order = []
            for chunk in self.chunks:
                uid = chunk.get('uid', chunk.get('chunk_uid', ''))
                if uid:
                    chunk_uid_order.append(uid)
        
        # Create mapping from chunk_uid to chunk data
        chunks_by_uid = {}
        for chunk in self.chunks:
            uid = chunk.get('uid', chunk.get('chunk_uid', ''))
            if uid:
                chunks_by_uid[uid] = chunk
        
        # Prepare batch data
        ids = []
        documents = []
        embeddings_list = []
        metadatas = []
        
        logger.info("در حال آماده‌سازی داده‌ها برای درج...")
        logger.info("Preparing data for insertion...")
        
        for i, chunk_uid in enumerate(chunk_uid_order):
            chunk = chunks_by_uid.get(chunk_uid)
            if not chunk:
                logger.warning(f"قطعه با UID یافت نشد: {chunk_uid}")
                logger.warning(f"Chunk not found for UID: {chunk_uid}")
                continue
            
            ids.append(chunk_uid)
            
            # Get document text (support multiple field names)
            content = chunk.get('text', chunk.get('normalized_text', chunk.get('content', '')))
            documents.append(content)
            
            # Get embedding
            embeddings_list.append(self.embeddings[i].tolist())
            
            # Create metadata
            metadata = {
                'document_uid': chunk.get('document_uid', ''),
                'document_title': chunk.get('document_title', ''),
                'document_type': chunk.get('document_type', ''),
                'article_number': chunk.get('article_number', ''),
                'note_label': chunk.get('note_label', ''),
                'clause_label': chunk.get('clause_label', ''),
                'chapter_title': chunk.get('chapter_title', ''),
                'source_type': chunk.get('source_type', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'char_count': chunk.get('char_count', 0),
                'token_count': chunk.get('token_count', 0),
                'approval_date': chunk.get('approval_date', ''),
                'approval_authority': chunk.get('approval_authority', '')
            }
            
            # Remove empty values for cleaner storage
            metadata = {k: v for k, v in metadata.items() if v}
            metadatas.append(metadata)
        
        # Batch insert with embeddings
        logger.info(f"در حال درج {len(ids)} قطعه در Chroma...")
        logger.info(f"Inserting {len(ids)} chunks into Chroma...")
        
        # Chroma has a batch size limit, so we'll insert in batches
        batch_size = 500  # Safe batch size for Chroma
        total_inserted = 0
        
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            batch_ids = ids[i:batch_end]
            batch_documents = documents[i:batch_end]
            batch_embeddings = embeddings_list[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            collection.upsert(
                ids=batch_ids,
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            
            total_inserted += len(batch_ids)
            logger.info(f"درج شده: {total_inserted}/{len(ids)} قطعه")
            logger.info(f"Inserted: {total_inserted}/{len(ids)} chunks")
        
        build_time = time.time() - start_time
        logger.info(f"فهرست Chroma با موفقیت ایجاد شد در {build_time:.2f} ثانیه")
        logger.info(f"Chroma index built successfully in {build_time:.2f} seconds")
        logger.info(f"تعداد قطعات فهرست شده: {collection.count()}")
        logger.info(f"Number of indexed chunks: {collection.count()}")
    
    def build_index(self) -> None:
        """
        ایجاد فهرست بردار بر اساس پشتیبان انتخاب شده
        Build vector index based on selected backend
        """
        logger.info(f"شروع ایجاد فهرست با پشتیبان: {self.backend}")
        logger.info(f"Starting index build with backend: {self.backend}")
        
        start_total_time = time.time()
        
        # Load data
        self._load_data()
        
        # Build index based on backend
        if self.backend == 'faiss':
            self._build_faiss_index()
        elif self.backend == 'chroma':
            self._build_chroma_index()
        else:
            raise ValueError(f"پشتیبان پشتیبانی نشده: {self.backend}")
        
        total_time = time.time() - start_total_time
        
        logger.info("خلاصه نهایی:")
        logger.info("Final Summary:")
        logger.info(f"  پشتیبان: {self.backend}")
        logger.info(f"  Backend: {self.backend}")
        logger.info(f"  تعداد بردارها: {self.metadata['count']}")
        logger.info(f"  Number of vectors: {self.metadata['count']}")
        logger.info(f"  ابعاد: {self.metadata['dimension']}")
        logger.info(f"  Dimensions: {self.metadata['dimension']}")
        logger.info(f"  زمان کل: {total_time:.2f} ثانیه")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info(f"  مسیر ذخیره: {self.vector_db_dir}")
        logger.info(f"  Storage path: {self.vector_db_dir}")
        
        logger.info("ایجاد فروشگاه بردار با موفقیت کامل شد!")
        logger.info("Vector store building completed successfully!")
    
    def run(self) -> None:
        """
        اجرای کامل فرآیند ایجاد فروشگاه بردار
        Run the complete vector store building process
        """
        try:
            self.build_index()
            
        except Exception as e:
            logger.error(f"خطا در فرآیند ایجاد فروشگاه بردار: {str(e)}")
            logger.error(f"Error in vector store building process: {str(e)}")
            raise


def load_config() -> Dict[str, Any]:
    """
    بارگذاری تنظیمات از فایل کانفیگ
    Load configuration from config file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config_paths = [
        "config.json",
        "data/config.json",
        "phase_3_rag/config.json"
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"خطا در بارگذاری کانفیگ از {config_path}: {e}")
                continue
    
    # Default configuration
    logger.info("از تنظیمات پیش‌فرض استفاده می‌شود")
    logger.info("Using default configuration")
    
    return {
        "rag": {
            "index_backend": "faiss"
        }
    }


def main():
    """
    تابع اصلی برای اجرای مستقل
    Main function for standalone execution
    """
    try:
        config = load_config()
        builder = VectorStoreBuilder(config)
        builder.run()
        
    except Exception as e:
        logger.error(f"خطای کلی: {str(e)}")
        logger.error(f"General error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())