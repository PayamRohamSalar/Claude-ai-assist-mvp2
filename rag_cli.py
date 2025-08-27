#!/usr/bin/env python3
"""
Persian RAG System CLI Tool

Command line interface for managing the complete RAG pipeline including
chunking, embedding generation, vector store building, and quality evaluation.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def safe_print(message: str):
    """Print message safely handling Unicode encoding issues"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', errors='replace').decode('ascii'))


def run_python_script(script_path: str, description: str) -> bool:
    """
    Run a Python script and return success status
    
    Args:
        script_path: Path to Python script
        description: Description of the operation
        
    Returns:
        bool: Success status
    """
    try:
        safe_print(f"Starting: {description}")
        
        # Use the current Python executable
        python_exe = sys.executable
        result = subprocess.run(
            [python_exe, script_path],
            cwd=str(project_root),
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            safe_print(f"✅ Completed: {description}")
            return True
        else:
            safe_print(f"❌ Failed: {description} (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        safe_print(f"❌ Error running {description}: {str(e)}")
        return False


def run_chunk(db_path: str, output_dir: str) -> bool:
    """Run chunking process"""
    safe_print(f"Running chunker with database: {db_path}")
    safe_print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run chunker
    return run_python_script(
        "phase_3_rag/chunker.py",
        "Document chunking process"
    )


def run_embed(chunks_path: str, output_dir: str) -> bool:
    """Run embedding generation process"""
    safe_print(f"Running embedding generator with chunks: {chunks_path}")
    safe_print(f"Output directory: {output_dir}")
    
    # Check if chunks file exists
    if not Path(chunks_path).exists():
        safe_print(f"❌ Chunks file not found: {chunks_path}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run embedding generator
    return run_python_script(
        "phase_3_rag/embedding_generator.py",
        "Embedding generation process"
    )


def run_build_index(backend: str, output_dir: str) -> bool:
    """Run vector store building process"""
    safe_print(f"Building vector index with backend: {backend}")
    safe_print(f"Output directory: {output_dir}")
    
    # Check if embeddings exist
    embeddings_path = Path("data/processed_phase_3/embeddings.npy")
    if not embeddings_path.exists():
        safe_print(f"❌ Embeddings file not found: {embeddings_path}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run vector store builder
    return run_python_script(
        "phase_3_rag/vector_store_builder.py",
        f"Vector index building ({backend})"
    )


def run_eval(output_dir: str) -> bool:
    """Run quality evaluation process"""
    safe_print(f"Running quality evaluation")
    safe_print(f"Output directory: {output_dir}")
    
    # Check if required files exist
    required_files = [
        "data/processed_phase_3/embeddings.npy",
        "data/processed_phase_3/embeddings_meta.json", 
        "data/processed_phase_3/chunks.json"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            safe_print(f"❌ Required file not found: {file_path}")
            return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Change to output directory and run evaluation
    original_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        success = run_python_script(
            str(project_root / "phase_3_rag" / "quality_eval.py"),
            "RAG quality evaluation"
        )
        return success
    finally:
        os.chdir(original_dir)


def run_all(backend: str) -> bool:
    """Run complete RAG pipeline"""
    safe_print("🚀 Starting complete RAG pipeline...")
    safe_print("=" * 50)
    
    # Step 1: Chunking
    safe_print("Step 1: Document chunking...")
    if not run_chunk(
        db_path='data/db/legal_assistant.db',
        output_dir='data/processed_phase_3'
    ):
        return False
    
    # Step 2: Embedding generation
    safe_print("\nStep 2: Embedding generation...")
    if not run_embed(
        chunks_path='data/processed_phase_3/chunks.json',
        output_dir='data/processed_phase_3'
    ):
        return False
    
    # Step 3: Index building  
    safe_print(f"\nStep 3: Index building ({backend})...")
    if not run_build_index(
        backend=backend,
        output_dir='data/processed_phase_3/vector_db'
    ):
        return False
    
    # Step 4: Quality evaluation
    safe_print("\nStep 4: Quality evaluation...")
    if not run_eval(output_dir='.'):
        return False
    
    safe_print("\n" + "=" * 50)
    safe_print("🎉 Complete RAG pipeline finished successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        prog='rag-cli',
        description='Persian RAG System CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_cli.py chunk --db data/db/legal_assistant.db --out data/processed_phase_3
  python rag_cli.py embed --chunks data/processed_phase_3/chunks.json
  python rag_cli.py build-index --backend faiss --out data/processed_phase_3/vector_db  
  python rag_cli.py eval --out .
  python rag_cli.py all

For help with each command:
  python rag_cli.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        title='Available Commands',
        description='Available commands:',
        help='Select a command to execute'
    )
    
    # Chunk command
    chunk_parser = subparsers.add_parser(
        'chunk',
        help='Convert database documents to text chunks',
        description='Convert legal documents from database into smaller text chunks'
    )
    chunk_parser.add_argument(
        '--db',
        type=str,
        default='data/db/legal_assistant.db',
        help='Database file path (default: data/db/legal_assistant.db)'
    )
    chunk_parser.add_argument(
        '--out',
        type=str,
        default='data/processed_phase_3',
        help='Output directory for chunks (default: data/processed_phase_3)'
    )
    
    # Embed command
    embed_parser = subparsers.add_parser(
        'embed',
        help='Generate embeddings from chunks',
        description='Generate numerical vector embeddings for each text chunk'
    )
    embed_parser.add_argument(
        '--chunks',
        type=str,
        default='data/processed_phase_3/chunks.json',
        help='Input chunks JSON file (default: data/processed_phase_3/chunks.json)'
    )
    embed_parser.add_argument(
        '--out',
        type=str,
        default='data/processed_phase_3',
        help='Output directory for embeddings (default: data/processed_phase_3)'
    )
    
    # Build-index command
    index_parser = subparsers.add_parser(
        'build-index',
        help='Build vector search index',
        description='Create fast search index from embeddings'
    )
    index_parser.add_argument(
        '--backend',
        type=str,
        choices=['faiss', 'chroma'],
        default='faiss',
        help='Vector store backend type (default: faiss)'
    )
    index_parser.add_argument(
        '--out',
        type=str,
        default='data/processed_phase_3/vector_db',
        help='Output directory for index (default: data/processed_phase_3/vector_db)'
    )
    
    # Eval command
    eval_parser = subparsers.add_parser(
        'eval',
        help='Evaluate RAG system quality',
        description='Run quality assessment tests on embeddings and retrieval'
    )
    eval_parser.add_argument(
        '--out',
        type=str,
        default='.',
        help='Output directory for evaluation reports (default: .)'
    )
    
    # All command
    all_parser = subparsers.add_parser(
        'all', 
        help='Run complete pipeline: chunk -> embed -> build-index -> eval',
        description='Execute all pipeline stages in sequence'
    )
    all_parser.add_argument(
        '--backend',
        type=str,
        choices=['faiss', 'chroma'],
        default='faiss',
        help='Vector store backend type (default: faiss)'
    )
    
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        # Execute selected command
        if args.command == 'chunk':
            success = run_chunk(args.db, args.out)
        elif args.command == 'embed':
            success = run_embed(args.chunks, args.out)
        elif args.command == 'build-index':
            success = run_build_index(args.backend, args.out)
        elif args.command == 'eval':
            success = run_eval(args.out)
        elif args.command == 'all':
            success = run_all(args.backend)
        else:
            safe_print("❌ Invalid command. Use --help for usage information.")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        safe_print("\n⏸️  Operation interrupted by user")
        return 130
    except Exception as e:
        safe_print(f"❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())