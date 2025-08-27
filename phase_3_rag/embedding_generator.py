#!/usr/bin/env python3
"""
تولیدکننده بردارهای متنی برای سیستم RAG
Embedding Generator for RAG System

این ماژول قطعات متنی را از chunks.json بارگذاری کرده و با استفاده از
مدل‌های Sentence Transformers، بردارهای متنی تولید می‌کند.

This module loads text chunks from chunks.json and generates embeddings
using Sentence Transformers models.
"""

import argparse
import json
import logging
import os
import hashlib
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # Create a dummy class for type hints when not available
    class SentenceTransformer:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    کلاس تولیدکننده بردارهای متنی
    Text Embedding Generator Class
    """
    
    def __init__(self, config: Dict[str, Any], chunks_file: Optional[str] = None, 
                 output_dir: Optional[str] = None, model_name: Optional[str] = None,
                 batch_size: Optional[int] = None):
        """
        راه‌اندازی تولیدکننده بردارها
        Initialize the embedding generator
        
        Args:
            config: Dictionary containing configuration parameters
            chunks_file: Override path to chunks.json file
            output_dir: Override output directory
            model_name: Override model name
            batch_size: Override batch size
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "خطا: کتابخانه sentence-transformers در دسترس نیست.\n"
                "Error: sentence-transformers library is not available.\n"
                "برای نصب از دستور زیر استفاده کنید:\n"
                "To install, use: pip install sentence-transformers"
            )
        
        self.config = config
        self.rag_config = config.get('rag', {})
        
        # Model configuration with CLI overrides
        self.model_name = model_name or self.rag_config.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.batch_size = batch_size or self.rag_config.get('batch_size', 256)
        
        # Paths with CLI overrides
        chunks_file = chunks_file or "data/processed_phase_3/chunks.json"
        output_dir = output_dir or "data/processed_phase_3"
        
        self.chunks_path = Path(chunks_file)
        self.output_dir = Path(output_dir)
        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.meta_path = self.output_dir / "embeddings_meta.json"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[SentenceTransformer] = None
        self.chunks: List[Dict[str, Any]] = []
        
    def _load_model(self) -> SentenceTransformer:
        """
        بارگذاری مدل Sentence Transformers
        Load Sentence Transformers model
        
        Returns:
            SentenceTransformer: The loaded model
            
        Raises:
            Exception: If model cannot be loaded
        """
        try:
            logger.info(f"در حال بارگذاری مدل: {self.model_name}")
            logger.info(f"Loading model: {self.model_name}")
            
            # Check if it's a local path
            if os.path.exists(self.model_name):
                model = SentenceTransformer(self.model_name)
                logger.info(f"مدل محلی با موفقیت بارگذاری شد از مسیر: {self.model_name}")
                logger.info(f"Local model loaded successfully from: {self.model_name}")
            else:
                # Try to load from Hugging Face Hub
                model = SentenceTransformer(self.model_name)
                logger.info(f"مدل از Hugging Face Hub بارگذاری شد: {self.model_name}")
                logger.info(f"Model loaded from Hugging Face Hub: {self.model_name}")
            
            return model
            
        except Exception as e:
            error_msg = (
                f"خطا در بارگذاری مدل '{self.model_name}': {str(e)}\n"
                f"Error loading model '{self.model_name}': {str(e)}\n"
                "پیشنهادات:\n"
                "Suggestions:\n"
                "1. اتصال اینترنت خود را بررسی کنید\n"
                "   Check your internet connection\n"
                "2. نام مدل را در تنظیمات config.rag.embedding_model بررسی کنید\n"
                "   Verify the model name in config.rag.embedding_model\n"
                "3. برای استفاده از مدل محلی، مسیر کامل را وارد کنید\n"
                "   For local models, provide the full path\n"
                "مثال مدل محلی: /path/to/your/local/model\n"
                "Local model example: /path/to/your/local/model"
            )
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """
        بارگذاری قطعات متنی از فایل JSON
        Load text chunks from JSON file
        
        Returns:
            List[Dict[str, Any]]: List of chunks
            
        Raises:
            FileNotFoundError: If chunks.json file not found
            Exception: If JSON parsing fails
        """
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"فایل قطعات یافت نشد: {self.chunks_path}\n"
                f"Chunks file not found: {self.chunks_path}\n"
                "لطفاً ابتدا مرحله chunking را اجرا کنید.\n"
                "Please run the chunking phase first."
            )
        
        try:
            logger.info(f"در حال بارگذاری قطعات از: {self.chunks_path}")
            logger.info(f"Loading chunks from: {self.chunks_path}")
            
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                chunks = data.get('chunks', [])
            elif isinstance(data, list):
                chunks = data
            else:
                raise ValueError("فرمت نامعتبر برای فایل chunks.json / Invalid format for chunks.json")
            
            logger.info(f"تعداد {len(chunks)} قطعه بارگذاری شد")
            logger.info(f"Loaded {len(chunks)} chunks")
            
            if not chunks:
                raise ValueError(
                    "هیچ قطعه‌ای در فایل chunks.json یافت نشد\n"
                    "No chunks found in chunks.json"
                )
            
            return chunks
            
        except json.JSONDecodeError as e:
            raise Exception(
                f"خطا در پردازش JSON: {str(e)}\n"
                f"JSON parsing error: {str(e)}"
            )
        except Exception as e:
            raise Exception(
                f"خطا در بارگذاری قطعات: {str(e)}\n"
                f"Error loading chunks: {str(e)}"
            )
    
    def _calculate_chunks_md5(self, chunks: List[Dict[str, Any]]) -> str:
        """
        محاسبه MD5 hash برای قطعات متنی
        Calculate MD5 hash for text chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            str: MD5 hash string
        """
        # Create a consistent string representation of chunks for hashing
        chunk_texts = []
        for chunk in chunks:
            # Use text first, fallback to normalized_text
            text = chunk.get('text', '')
            if not text:
                text = chunk.get('normalized_text', '')
            chunk_texts.append(text)
        
        combined_text = '\n'.join(chunk_texts)
        return hashlib.md5(combined_text.encode('utf-8')).hexdigest()
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        تولید بردارها به صورت دسته‌ای
        Generate embeddings in batches
        
        Args:
            texts: List of text strings
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not self.model:
            raise ValueError("مدل بارگذاری نشده است / Model not loaded")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        # Ensure float32 dtype
        return embeddings.astype(np.float32)
    
    def generate_embeddings(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        تولید بردارهای متنی برای همه قطعات
        Generate embeddings for all chunks
        
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Embeddings array and metadata
        """
        # Load model and chunks
        logger.info("شروع فرآیند تولید بردارها...")
        logger.info("Starting embedding generation process...")
        
        self.model = self._load_model()
        self.chunks = self._load_chunks()
        
        # Extract text content and UIDs
        chunk_texts = []
        chunk_uids = []
        
        for chunk in self.chunks:
            # Use text as primary content, fallback to normalized_text
            content = chunk.get('text', '').strip()
            if not content:
                content = chunk.get('normalized_text', '').strip()
            
            # Support both 'uid' and 'chunk_uid' fields for UID
            uid = chunk.get('uid', chunk.get('chunk_uid', ''))
            
            # Validate content
            if not content:
                error_msg = f"قطعه بدون متن یافت شد با UID: {uid}\nChunk with no text content found with UID: {uid}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not uid:
                logger.error("قطعه بدون UID یافت شد")
                logger.error("Chunk without UID found")
                raise ValueError("همه قطعات باید دارای UID باشند / All chunks must have UIDs")
            
            chunk_texts.append(content)
            chunk_uids.append(uid)
        
        # Calculate MD5 hash of chunks
        chunks_md5 = self._calculate_chunks_md5(self.chunks)
        
        logger.info(f"در حال تولید بردارهای {self.model.get_sentence_embedding_dimension()} بعدی برای {len(chunk_texts)} قطعه...")
        logger.info(f"Generating {self.model.get_sentence_embedding_dimension()}-dimensional embeddings for {len(chunk_texts)} chunks...")
        
        # Generate embeddings
        embeddings = self._generate_embeddings_batch(chunk_texts)
        
        # Create metadata
        metadata = {
            "model_name": self.model_name,
            "dimension": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
            "chunks_md5": chunks_md5,
            "dtype": str(embeddings.dtype),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "chunk_uid_order": chunk_uids
        }
        
        # Validate alignment
        if len(chunk_uids) != embeddings.shape[0]:
            raise ValueError(
                f"تعداد UIDs ({len(chunk_uids)}) با تعداد بردارها ({embeddings.shape[0]}) مطابقت ندارد\n"
                f"Number of UIDs ({len(chunk_uids)}) doesn't match number of embeddings ({embeddings.shape[0]})"
            )
        
        logger.info(f"تولید بردارها با موفقیت انجام شد: {embeddings.shape}")
        logger.info(f"Embeddings generated successfully: {embeddings.shape}")
        
        return embeddings, metadata
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        ذخیره بردارها و متادیتا
        Save embeddings and metadata
        
        Args:
            embeddings: NumPy array of embeddings
            metadata: Metadata dictionary
        """
        # Save embeddings as .npy file
        logger.info(f"در حال ذخیره بردارها در: {self.embeddings_path}")
        logger.info(f"Saving embeddings to: {self.embeddings_path}")
        np.save(self.embeddings_path, embeddings)
        
        # Save metadata as JSON
        logger.info(f"در حال ذخیره متادیتا در: {self.meta_path}")
        logger.info(f"Saving metadata to: {self.meta_path}")
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Log summary
        logger.info("خلاصه نتایج:")
        logger.info("Summary:")
        logger.info(f"  تعداد بردارها: {metadata['count']}")
        logger.info(f"  Number of embeddings: {metadata['count']}")
        logger.info(f"  ابعاد: {metadata['dimension']}")
        logger.info(f"  Dimensions: {metadata['dimension']}")
        logger.info(f"  مدل: {metadata['model_name']}")
        logger.info(f"  Model: {metadata['model_name']}")
        logger.info(f"  نوع داده: {metadata['dtype']}")
        logger.info(f"  Data type: {metadata['dtype']}")
        
        file_size_mb = os.path.getsize(self.embeddings_path) / (1024 * 1024)
        logger.info(f"  حجم فایل: {file_size_mb:.2f} MB")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
    
    def run(self) -> None:
        """
        اجرای کامل فرآیند تولید بردارها
        Run the complete embedding generation process
        """
        try:
            embeddings, metadata = self.generate_embeddings()
            self.save_embeddings(embeddings, metadata)
            
            logger.info("فرآیند تولید بردارها با موفقیت کامل شد!")
            logger.info("Embedding generation process completed successfully!")
            
        except Exception as e:
            logger.error(f"خطا در فرآیند تولید بردارها: {str(e)}")
            logger.error(f"Error in embedding generation process: {str(e)}")
            raise


def create_cli_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for embedding generator.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate embeddings for text chunks using Sentence Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python embedding_generator.py
  python embedding_generator.py --chunks-file data/chunks.json
  python embedding_generator.py --output-dir output/ --model-name all-MiniLM-L6-v2
  python embedding_generator.py --batch-size 128 --model-name paraphrase-multilingual-MiniLM-L12-v2
        """
    )
    
    parser.add_argument(
        '--chunks-file',
        default='data/processed_phase_3/chunks.json',
        help='Path to chunks.json file (default: data/processed_phase_3/chunks.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/processed_phase_3',
        help='Output directory for embeddings.npy and embeddings_meta.json (default: data/processed_phase_3)'
    )
    
    parser.add_argument(
        '--model-name',
        help='Sentence Transformers model name (default: from config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for embedding generation (default: from config)'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.json',
        help='Configuration file path (default: config/config.json)'
    )
    
    return parser


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    بارگذاری تنظیمات از فایل کانفیگ
    Load configuration from config file
    
    Args:
        config_path: Specific config file path to load
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config_paths = []
    
    # If specific path provided, try it first
    if config_path and os.path.exists(config_path):
        config_paths.append(config_path)
    
    # Fallback paths
    config_paths.extend([
        "config/config.json",
        "config.json",
        "data/config.json",
        "phase_3_rag/config.json"
    ])
    
    for path in config_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading config from: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"خطا در بارگذاری کانفیگ از {path}: {e}")
                logger.warning(f"Error loading config from {path}: {e}")
                continue
    
    # Default configuration
    logger.info("از تنظیمات پیش‌فرض استفاده می‌شود")
    logger.info("Using default configuration")
    
    return {
        "rag": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "batch_size": 256
        }
    }


def main():
    """
    تابع اصلی برای اجرای مستقل
    Main function for standalone execution
    """
    try:
        # Parse command line arguments
        parser = create_cli_parser()
        args = parser.parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Log CLI overrides if provided
        if args.model_name:
            logger.info(f"Model name override: {args.model_name}")
        if args.batch_size:
            logger.info(f"Batch size override: {args.batch_size}")
        if args.chunks_file != 'data/processed_phase_3/chunks.json':
            logger.info(f"Chunks file override: {args.chunks_file}")
        if args.output_dir != 'data/processed_phase_3':
            logger.info(f"Output directory override: {args.output_dir}")
        
        # Create generator with CLI overrides
        generator = EmbeddingGenerator(
            config=config,
            chunks_file=args.chunks_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
        
        # Run the generator
        generator.run()
        
        logger.info("Embedding generation completed successfully!")
        logger.info("تولید بردارها با موفقیت انجام شد!")
        print("[SUCCESS] Embedding generation completed successfully!")
        
    except KeyboardInterrupt:
        logger.error("Process interrupted by user")
        logger.error("فرآیند توسط کاربر متوقف شد")
        print("[INTERRUPTED] Process interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"خطای کلی: {str(e)}")
        logger.error(f"General error: {str(e)}")
        print(f"[ERROR] Embedding generation failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())