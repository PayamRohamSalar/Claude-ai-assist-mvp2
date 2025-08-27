#!/usr/bin/env python3
"""
CLI برای سیستم RAG فارسی
Persian RAG System Command Line Interface

این ابزار خط فرمان برای مدیریت کامل فرآیند RAG شامل chunking، embedding generation،
vector store building و quality evaluation می‌باشد.

This command line tool provides complete RAG pipeline management including
chunking, embedding generation, vector store building, and quality evaluation.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def safe_print(message: str):
    """Print message safely handling Unicode encoding issues"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback to ASCII representation
        print(message.encode('ascii', errors='replace').decode('ascii'))

# Import shared utilities
try:
    from shared_utils.config_manager import ConfigManager
    from shared_utils.logger import get_logger
except ImportError as e:
    try:
        print(f"Error loading shared utilities: {e}")
    except UnicodeEncodeError:
        print("Error loading shared utilities")
    sys.exit(1)

# Import phase 3 modules
try:
    from phase_3_rag import chunker
    from phase_3_rag.embedding_generator import EmbeddingGenerator
    from phase_3_rag.vector_store_builder import VectorStoreBuilder
    from phase_3_rag.quality_eval import QualityEvaluator
except ImportError as e:
    try:
        print(f"Error loading phase 3 modules: {e}")
    except UnicodeEncodeError:
        print("Error loading phase 3 modules")
    sys.exit(1)


class PersianRAGCLI:
    """
    کلاس CLI اصلی برای سیستم RAG فارسی
    Main CLI class for Persian RAG system
    """
    
    def __init__(self):
        """Initialize CLI with config and logger"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Setup logger
        self.logger = get_logger(
            name='phase3_cli',
            level='INFO'
        )
        
        self.logger.info("راه‌اندازی CLI سیستم RAG...")
        self.logger.info("Initializing RAG system CLI...")
    
    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create argument parser with Persian help strings
        
        Returns:
            argparse.ArgumentParser: Configured parser
        """
        parser = argparse.ArgumentParser(
            prog='rag-cli',
            description='ابزار خط فرمان برای سیستم RAG فارسی',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
مثال‌های استفاده:
  python cli.py chunk --db data/db/legal_assistant.db --out data/processed_phase_3
  python cli.py embed --chunks data/processed_phase_3/chunks.json
  python cli.py build-index --backend faiss --out data/processed_phase_3/vector_db
  python cli.py eval --out .
  python cli.py all

برای دریافت راهنمای هر دستور:
  python cli.py <command> --help
            """
        )
        
        subparsers = parser.add_subparsers(
            dest='command',
            title='دستورات موجود',
            description='دستورات قابل اجرا:',
            help='انتخاب دستور برای اجرا'
        )
        
        # Chunk command
        chunk_parser = subparsers.add_parser(
            'chunk',
            help='تبدیل اسناد دیتابیس به قطعات متنی',
            description='این دستور اسناد موجود در دیتابیس را به قطعات کوچک‌تر تقسیم می‌کند'
        )
        chunk_parser.add_argument(
            '--db',
            type=str,
            default='data/db/legal_assistant.db',
            help='مسیر فایل دیتابیس (پیش‌فرض: data/db/legal_assistant.db)'
        )
        chunk_parser.add_argument(
            '--out',
            type=str,
            default='data/processed_phase_3',
            help='مسیر خروجی برای ذخیره قطعات (پیش‌فرض: data/processed_phase_3)'
        )
        
        # Embed command
        embed_parser = subparsers.add_parser(
            'embed',
            help='تولید بردارهای متنی از قطعات',
            description='این دستور برای هر قطعه متنی یک بردار عددی تولید می‌کند'
        )
        embed_parser.add_argument(
            '--chunks',
            type=str,
            default='data/processed_phase_3/chunks.json',
            help='مسیر فایل قطعات JSON (پیش‌فرض: data/processed_phase_3/chunks.json)'
        )
        embed_parser.add_argument(
            '--out',
            type=str,
            default='data/processed_phase_3',
            help='مسیر خروجی برای ذخیره بردارها (پیش‌فرض: data/processed_phase_3)'
        )
        
        # Build-index command
        index_parser = subparsers.add_parser(
            'build-index',
            help='ایجاد فهرست جستجوی بردارها',
            description='این دستور از بردارهای تولید شده فهرست جستجوی سریع می‌سازد'
        )
        index_parser.add_argument(
            '--backend',
            type=str,
            choices=['faiss', 'chroma'],
            default='faiss',
            help='نوع پشتیبان فهرست‌سازی (پیش‌فرض: faiss)'
        )
        index_parser.add_argument(
            '--out',
            type=str,
            default='data/processed_phase_3/vector_db',
            help='مسیر خروجی برای ذخیره فهرست (پیش‌فرض: data/processed_phase_3/vector_db)'
        )
        
        # Eval command
        eval_parser = subparsers.add_parser(
            'eval',
            help='ارزیابی کیفیت سیستم RAG',
            description='این دستور کیفیت بردارها و عملکرد جستجو را بررسی می‌کند'
        )
        eval_parser.add_argument(
            '--out',
            type=str,
            default='.',
            help='مسیر خروجی برای ذخیره گزارش‌های ارزیابی (پیش‌فرض: .)'
        )
        
        # All command
        all_parser = subparsers.add_parser(
            'all',
            help='اجرای کامل تمام مراحل به ترتیب',
            description='این دستور تمام مراحل chunk → embed → build-index → eval را به ترتیب اجرا می‌کند'
        )
        all_parser.add_argument(
            '--backend',
            type=str,
            choices=['faiss', 'chroma'],
            default='faiss',
            help='نوع پشتیبان فهرست‌سازی (پیش‌فرض: faiss)'
        )
        
        return parser
    
    def run_chunk(self, db_path: str, output_dir: str) -> bool:
        """
        Run chunking process
        
        Args:
            db_path: Database file path
            output_dir: Output directory for chunks
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"شروع فرآیند chunking با دیتابیس: {db_path}")
            self.logger.info(f"Starting chunking process with database: {db_path}")
            
            # Check if database exists
            if not Path(db_path).exists():
                raise FileNotFoundError(f"فایل دیتابیس یافت نشد: {db_path}")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Set environment for chunker
            original_dir = os.getcwd()
            try:
                os.chdir(output_dir)  # Change to output directory for chunker
                
                # Load chunking configuration
                config = chunker.load_chunking_config()
                
                # Create chunks from database
                chunks = chunker.create_chunks_from_database(db_path=db_path, config=config)
                
                if not chunks:
                    raise ValueError("No chunks were created from the database")
                
                # Write chunks to JSON file
                chunker.write_chunks_json(chunks)
                
            finally:
                os.chdir(original_dir)
            
            self.logger.info("فرآیند chunking با موفقیت کامل شد")
            self.logger.info("Chunking process completed successfully")
            safe_print("✅ تقسیم‌بندی اسناد با موفقیت انجام شد")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در فرآیند chunking: {str(e)}")
            self.logger.error(f"Error in chunking process: {str(e)}")
            safe_print(f"❌ خطا در تقسیم‌بندی اسناد: {str(e)}")
            return False
    
    def run_embed(self, chunks_path: str, output_dir: str) -> bool:
        """
        Run embedding generation process
        
        Args:
            chunks_path: Path to chunks JSON file
            output_dir: Output directory for embeddings
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"شروع فرآیند تولید embedding با قطعات: {chunks_path}")
            self.logger.info(f"Starting embedding generation with chunks: {chunks_path}")
            
            # Check if chunks file exists
            if not Path(chunks_path).exists():
                raise FileNotFoundError(f"فایل قطعات یافت نشد: {chunks_path}")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Update config with paths
            config = self.config.copy()
            
            # Initialize and run embedding generator
            generator = EmbeddingGenerator(config)
            generator.chunks_path = Path(chunks_path)
            generator.embeddings_path = Path(output_dir) / "embeddings.npy"
            generator.meta_path = Path(output_dir) / "embeddings_meta.json"
            generator.run()
            
            self.logger.info("فرآیند تولید embedding با موفقیت کامل شد")
            self.logger.info("Embedding generation process completed successfully")
            safe_print("✅ تولید بردارها با موفقیت انجام شد")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در فرآیند تولید embedding: {str(e)}")
            self.logger.error(f"Error in embedding generation process: {str(e)}")
            safe_print(f"❌ خطا در تولید بردارها: {str(e)}")
            return False
    
    def run_build_index(self, backend: str, output_dir: str) -> bool:
        """
        Run vector store building process
        
        Args:
            backend: Vector store backend (faiss/chroma)
            output_dir: Output directory for vector store
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"شروع فرآیند ایجاد فهرست با پشتیبان: {backend}")
            self.logger.info(f"Starting index building with backend: {backend}")
            
            # Check if embeddings exist
            embeddings_path = Path("data/processed_phase_3/embeddings.npy")
            if not embeddings_path.exists():
                raise FileNotFoundError(f"فایل بردارها یافت نشد: {embeddings_path}")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Update config with backend
            config = self.config.copy()
            config['rag'] = config.get('rag', {})
            config['rag']['index_backend'] = backend
            
            # Initialize and run vector store builder
            builder = VectorStoreBuilder(config)
            builder.vector_db_dir = Path(output_dir)
            builder.run()
            
            self.logger.info("فرآیند ایجاد فهرست با موفقیت کامل شد")
            self.logger.info("Index building process completed successfully")
            safe_print("✅ ایجاد فهرست جستجو با موفقیت انجام شد")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در فرآیند ایجاد فهرست: {str(e)}")
            self.logger.error(f"Error in index building process: {str(e)}")
            safe_print(f"❌ خطا در ایجاد فهرست جستجو: {str(e)}")
            return False
    
    def run_eval(self, output_dir: str) -> bool:
        """
        Run quality evaluation process
        
        Args:
            output_dir: Output directory for evaluation reports
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"شروع فرآیند ارزیابی کیفیت")
            self.logger.info(f"Starting quality evaluation process")
            
            # Check if required files exist
            required_files = [
                "data/processed_phase_3/embeddings.npy",
                "data/processed_phase_3/embeddings_meta.json",
                "data/processed_phase_3/chunks.json"
            ]
            
            for file_path in required_files:
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"فایل مورد نیاز یافت نشد: {file_path}")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save current directory and change to output directory
            original_dir = os.getcwd()
            try:
                os.chdir(output_dir)
                
                # Initialize and run quality evaluator
                evaluator = QualityEvaluator(self.config)
                evaluator.run()
                
            finally:
                os.chdir(original_dir)
            
            self.logger.info("فرآیند ارزیابی کیفیت با موفقیت کامل شد")
            self.logger.info("Quality evaluation process completed successfully")
            safe_print("✅ ارزیابی کیفیت با موفقیت انجام شد")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در فرآیند ارزیابی کیفیت: {str(e)}")
            self.logger.error(f"Error in quality evaluation process: {str(e)}")
            safe_print(f"❌ خطا در ارزیابی کیفیت: {str(e)}")
            return False
    
    def run_all(self, backend: str) -> bool:
        """
        Run complete RAG pipeline
        
        Args:
            backend: Vector store backend (faiss/chroma)
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("شروع اجرای کامل پایپ‌لاین RAG")
            self.logger.info("Starting complete RAG pipeline execution")
            
            safe_print("🚀 شروع اجرای کامل سیستم RAG...")
            safe_print("=" * 50)
            
            # Step 1: Chunking
            safe_print("مرحله ۱: تقسیم‌بندی اسناد...")
            if not self.run_chunk(
                db_path='data/db/legal_assistant.db',
                output_dir='data/processed_phase_3'
            ):
                return False
            
            # Step 2: Embedding generation
            safe_print("\nمرحله ۲: تولید بردارها...")
            if not self.run_embed(
                chunks_path='data/processed_phase_3/chunks.json',
                output_dir='data/processed_phase_3'
            ):
                return False
            
            # Step 3: Index building
            safe_print(f"\nمرحله ۳: ایجاد فهرست جستجو ({backend})...")
            if not self.run_build_index(
                backend=backend,
                output_dir='data/processed_phase_3/vector_db'
            ):
                return False
            
            # Step 4: Quality evaluation
            safe_print("\nمرحله ۴: ارزیابی کیفیت...")
            if not self.run_eval(output_dir='.'):
                return False
            
            safe_print("\n" + "=" * 50)
            safe_print("🎉 تمام مراحل با موفقیت کامل شد!")
            self.logger.info("تمام مراحل پایپ‌لاین RAG با موفقیت کامل شد")
            self.logger.info("Complete RAG pipeline execution finished successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در اجرای کامل پایپ‌لاین: {str(e)}")
            self.logger.error(f"Error in complete pipeline execution: {str(e)}")
            safe_print(f"❌ خطا در اجرای کامل سیستم: {str(e)}")
            return False
    
    def run(self, args: argparse.Namespace) -> int:
        """
        Run CLI command based on arguments
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            if args.command == 'chunk':
                success = self.run_chunk(args.db, args.out)
            elif args.command == 'embed':
                success = self.run_embed(args.chunks, args.out)
            elif args.command == 'build-index':
                success = self.run_build_index(args.backend, args.out)
            elif args.command == 'eval':
                success = self.run_eval(args.out)
            elif args.command == 'all':
                success = self.run_all(args.backend)
            else:
                safe_print("❌ دستور نامعتبر. از --help استفاده کنید.")
                safe_print("❌ Invalid command. Use --help for usage information.")
                return 1
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            safe_print("\n⏸️  عملیات توسط کاربر متوقف شد")
            safe_print("\n⏸️  Operation interrupted by user")
            self.logger.info("عملیات توسط کاربر متوقف شد")
            return 130
        except Exception as e:
            self.logger.error(f"خطای غیرمنتظره: {str(e)}")
            self.logger.error(f"Unexpected error: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            safe_print(f"❌ خطای غیرمنتظره: {str(e)}")
            return 1


def main() -> int:
    """
    Main entry point for CLI
    
    Returns:
        int: Exit code
    """
    try:
        # Initialize CLI
        cli = PersianRAGCLI()
        
        # Create and parse arguments
        parser = cli.create_parser()
        args = parser.parse_args()
        
        # Show help if no command provided
        if not args.command:
            parser.print_help()
            return 0
        
        # Run command
        return cli.run(args)
        
    except Exception as e:
        safe_print(f"❌ خطا در راه‌اندازی CLI: {str(e)}")
        safe_print(f"❌ Error initializing CLI: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())