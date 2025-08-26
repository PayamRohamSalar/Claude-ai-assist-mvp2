"""
Phase 3 RAG Module for Legal Assistant AI

This module provides text chunking functionality for retrieval-augmented
generation (RAG) processing of Persian legal documents.

Author: Legal Assistant AI Team
Version: 3.0
"""

from .chunker import (
    DocumentChunk,
    ChunkingConfig,
    DocumentChunkerError,
    create_chunks_from_database,
    write_chunks_json,
    load_chunking_config,
    compute_chunk_uid,
    normalize_text,
    split_text_into_chunks
)

__version__ = "3.0"
__all__ = [
    "DocumentChunk",
    "ChunkingConfig", 
    "DocumentChunkerError",
    "create_chunks_from_database",
    "write_chunks_json",
    "load_chunking_config",
    "compute_chunk_uid",
    "normalize_text",
    "split_text_into_chunks"
]