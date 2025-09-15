"""
Document Chunker Module for Legal Assistant AI (Phase 3)

This module reads legal documents from SQLite database and creates
canonical chunks with stable IDs for RAG processing.

Author: Legal Assistant AI Team
Version: 3.0
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared_utils.config_manager import get_config
from shared_utils.logger import get_logger


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    max_chunk_chars: int = 1000
    min_chunk_chars: int = 100
    overlap_chars: int = 150
    overlap_ratio: float = 0.15
    sentence_boundaries: Dict[str, List[str]] = None
    preserve_headers: List[str] = None
    normalize_spaces: bool = True
    preserve_original: bool = True
    splitting_strategy: str = "sentence_aware"
    prefer_article_chunks: bool = True
    preserve_article_note_relationships: bool = True
    max_notes_per_chunk: int = 1
    strict_header_boundaries: bool = True
    
    def __post_init__(self):
        if self.sentence_boundaries is None:
            self.sentence_boundaries = {
                "primary": [".", "؟", "!", "…"],
                "secondary": ["،", "؛"]
            }
        if self.preserve_headers is None:
            self.preserve_headers = ["تبصره", "بند", "ماده", "فصل"]


@dataclass
class DocumentChunk:
    """Represents a single document chunk."""
    chunk_uid: str
    document_uid: str
    document_title: str
    document_type: str
    section: str
    approval_date: str
    approval_authority: str
    chapter_title: str
    article_number: str
    note_label: str
    clause_label: str
    text: str
    normalized_text: str
    chunk_index: int
    char_count: int
    token_count: int
    source_type: str  # 'article', 'note', 'clause'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentChunkerError(Exception):
    """Custom exception for chunking errors."""
    pass


def load_chunking_config() -> ChunkingConfig:
    """Load chunking configuration from config file."""
    try:
        config_path = Path(__file__).parent.parent / "config" / "chunking_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            chunking_data = config_data.get('chunking', {})
            return ChunkingConfig(
                max_chunk_chars=chunking_data.get('max_chunk_chars', 1000),
                min_chunk_chars=chunking_data.get('min_chunk_chars', 100),
                overlap_chars=chunking_data.get('overlap_chars', 150),
                overlap_ratio=chunking_data.get('overlap_ratio', 0.15),
                sentence_boundaries=chunking_data.get('sentence_boundaries'),
                preserve_headers=chunking_data.get('preserve_headers'),
                normalize_spaces=chunking_data.get('whitespace_normalization', {}).get('normalize_spaces', True),
                preserve_original=chunking_data.get('whitespace_normalization', {}).get('preserve_original', True),
                splitting_strategy=chunking_data.get('splitting_strategy', 'sentence_aware'),
                prefer_article_chunks=chunking_data.get('prefer_article_chunks', True),
                preserve_article_note_relationships=chunking_data.get('preserve_article_note_relationships', True),
                max_notes_per_chunk=chunking_data.get('max_notes_per_chunk', 1),
                strict_header_boundaries=chunking_data.get('strict_header_boundaries', True)
            )
        else:
            return ChunkingConfig()
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Could not load chunking config, using defaults: {e}")
        return ChunkingConfig()


def normalize_text(text: str, config: ChunkingConfig) -> str:
    """Normalize text according to configuration."""
    if not text:
        return ""
    
    normalized = text
    
    if config.normalize_spaces:
        # Normalize multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        
        # Normalize Persian/Arabic characters if needed
        normalized = re.sub(r'[ي]', 'ی', normalized)  # Normalize ya
        normalized = re.sub(r'[ك]', 'ک', normalized)  # Normalize kaf
    
    return normalized


def compute_chunk_uid(document_uid: str, article_number: str, note_label: str, 
                     chunk_index: int, normalized_text: str) -> str:
    """
    Compute stable chunk UID using SHA-1 hash.
    
    Args:
        document_uid: Document unique identifier
        article_number: Article number or 'NA'
        note_label: Note label or 'NA'
        chunk_index: Index of chunk within article/note
        normalized_text: Normalized chunk text
        
    Returns:
        str: SHA-1 hash as chunk UID
    """
    # Prepare components
    article_part = article_number or 'NA'
    note_part = note_label or 'NA'
    text_prefix = normalized_text.encode('ascii', errors='ignore').decode('ascii')[:64]
    
    # Create hash input
    hash_input = f"{document_uid}|{article_part}|{note_part}|{chunk_index}|{text_prefix}"
    
    # Generate SHA-1 hash
    sha1_hash = hashlib.sha1(hash_input.encode('utf-8')).hexdigest()
    return sha1_hash


def estimate_token_count(text: str) -> int:
    """Estimate token count for Persian text."""
    if not text:
        return 0
    
    # Simple estimation: split by whitespace and punctuation
    # Persian typically has fewer tokens per character than English
    words = re.findall(r'\S+', text)
    return len(words)


def find_sentence_boundaries(text: str, config: ChunkingConfig) -> List[int]:
    """Find sentence boundary positions in text."""
    boundaries = []
    
    primary_pattern = '|'.join(re.escape(punct) for punct in config.sentence_boundaries['primary'])
    secondary_pattern = '|'.join(re.escape(punct) for punct in config.sentence_boundaries['secondary'])
    
    # Find primary boundaries (preferred)
    for match in re.finditer(f'({primary_pattern})(?=\\s|$)', text):
        boundaries.append(match.end())
    
    # Find secondary boundaries if no primary ones found
    if not boundaries:
        for match in re.finditer(f'({secondary_pattern})(?=\\s|$)', text):
            boundaries.append(match.end())
    
    # Always include text end as boundary
    if boundaries and boundaries[-1] != len(text):
        boundaries.append(len(text))
    elif not boundaries:
        boundaries.append(len(text))
    
    return sorted(set(boundaries))


def has_header_marker(text: str, config: ChunkingConfig) -> bool:
    """Check if text starts with a header marker that should not be split."""
    text_stripped = text.strip()
    for header in config.preserve_headers:
        if text_stripped.startswith(header):
            return True
    return False


def find_header_markers(text: str) -> List[Tuple[int, str]]:
    """
    Find all header markers (تبصره and بند) in text using regex.
    
    Args:
        text: Text to search for header markers
        
    Returns:
        List of tuples (position, matched_text) for each header marker found
    """
    header_pattern = r'\b(تبصره\s+\d+|بند\s+[\u0627-\u06CC]+)\b'
    matches = []
    
    for match in re.finditer(header_pattern, text):
        matches.append((match.start(), match.group()))
    
    return matches


def count_header_markers_in_chunk(chunk_text: str) -> int:
    """
    Count the number of header markers in a chunk.
    
    Args:
        chunk_text: Text chunk to analyze
        
    Returns:
        int: Number of header markers found
    """
    markers = find_header_markers(chunk_text)
    return len(markers)


def split_text_into_chunks(text: str, config: ChunkingConfig) -> List[str]:
    """
    Split text into chunks according to configuration rules.
    Enhanced to handle legal structure (تبصره and بند) properly.
    
    Args:
        text: Text to split
        config: Chunking configuration
        
    Returns:
        List of text chunks
    """
    logger = get_logger(__name__)
    
    if not text or len(text) <= config.min_chunk_chars:
        return [text] if text else []
    
    # Step 1: Legal structure-aware splitting
    if getattr(config, 'splitting_strategy', 'sentence_aware') == 'legal_structure_aware':
        return _split_legal_structure_aware(text, config)
    
    # Step 2: Find header markers (تبصره and بند)
    header_markers = find_header_markers(text)
    
    # Step 3: Strict header boundary enforcement
    if getattr(config, 'strict_header_boundaries', True) and len(header_markers) > 1:
        if getattr(config, 'max_notes_per_chunk', 1) == 1:
            logger.debug(f"Found {len(header_markers)} header markers, enforcing strict boundaries")
            return _split_at_header_boundaries(text, header_markers, config)
    
    # Step 4: If multiple header markers found, split at each header position
    if len(header_markers) > 1:
        logger.debug(f"Found {len(header_markers)} header markers, splitting text")
        
        subtexts = []
        last_pos = 0
        
        for pos, marker in header_markers:
            if pos > last_pos:
                # Add text before this marker
                subtexts.append(text[last_pos:pos].strip())
            last_pos = pos
        
        # Add remaining text after last marker
        if last_pos < len(text):
            subtexts.append(text[last_pos:].strip())
        
        # Remove empty subtexts
        subtexts = [subtext for subtext in subtexts if subtext.strip()]
        
        # Recursively process each subtext
        all_chunks = []
        for subtext in subtexts:
            subtext_chunks = split_text_into_chunks(subtext, config)
            all_chunks.extend(subtext_chunks)
        
        return all_chunks
    
    # Step 3: If text is short enough, return as single chunk
    if len(text) <= config.max_chunk_chars:
        return [text]
    
    # Step 4: Check if text has header that should not be split
    if has_header_marker(text, config):
        # Find end of header line
        first_newline = text.find('\n')
        if first_newline > 0:
            header = text[:first_newline + 1]
            rest = text[first_newline + 1:]
            
            # Recursively chunk the rest
            rest_chunks = split_text_into_chunks(rest, config)
            if rest_chunks:
                rest_chunks[0] = header + rest_chunks[0]
            else:
                rest_chunks = [header]
            return rest_chunks
    
    # Step 5: Apply sentence-based or size-based chunking
    chunks = []
    boundaries = find_sentence_boundaries(text, config)
    
    if not boundaries:
        # Fallback: split by character count
        start = 0
        while start < len(text):
            end = min(start + config.max_chunk_chars, len(text))
            chunks.append(text[start:end])
            start = end - config.overlap_chars if end < len(text) else end
        return chunks
    
    # Split by sentence boundaries with overlap
    start = 0
    current_chunk = ""
    
    for boundary in boundaries:
        sentence = text[start:boundary]
        
        # If adding this sentence would exceed max chars, finalize current chunk
        if current_chunk and len(current_chunk + sentence) > config.max_chunk_chars:
            if len(current_chunk) >= config.min_chunk_chars:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - config.overlap_chars)
                current_chunk = current_chunk[overlap_start:] + sentence
            else:
                current_chunk += sentence
        else:
            current_chunk += sentence
        
        start = boundary
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Step 6: Validate that each chunk has at most one header marker
    final_chunks = []
    for chunk in chunks:
        if chunk.strip():
            header_count = count_header_markers_in_chunk(chunk)
            if header_count > 1:
                logger.warning(f"Chunk contains {header_count} header markers, may affect retrieval granularity")
            final_chunks.append(chunk.strip())
    
    return final_chunks


def _split_legal_structure_aware(text: str, config: ChunkingConfig) -> List[str]:
    """
    Split text with full awareness of legal document structure.
    
    This ensures articles and their related notes stay together when possible.
    """
    logger = get_logger(__name__)
    
    # Find all legal structure markers
    article_pattern = r'\b(ماده\s+[\d۰-۹]+|ماده\s+واحده)\b'
    note_pattern = r'\b(تبصره\s+[\d۰-۹]+)\b'
    clause_pattern = r'\b(بند\s+[\u0627-\u06CC]+)\b'
    
    # Find all markers with their positions
    markers = []
    
    for match in re.finditer(article_pattern, text):
        markers.append((match.start(), 'article', match.group()))
    
    for match in re.finditer(note_pattern, text):
        markers.append((match.start(), 'note', match.group()))
    
    for match in re.finditer(clause_pattern, text):
        markers.append((match.start(), 'clause', match.group()))
    
    # Sort by position
    markers.sort(key=lambda x: x[0])
    
    if not markers:
        # No legal structure found, fall back to regular chunking
        return _split_by_sentences(text, config)
    
    # Split text into logical legal units
    chunks = []
    last_pos = 0
    current_article = None
    current_article_text = ""
    
    for pos, marker_type, marker_text in markers:
        if marker_type == 'article':
            # Save previous article if exists
            if current_article and current_article_text.strip():
                article_chunks = _finalize_article_chunk(current_article_text, config)
                chunks.extend(article_chunks)
            
            # Start new article
            current_article = marker_text
            current_article_text = text[last_pos:pos].strip()
            if current_article_text:
                current_article_text += "\n\n"
            current_article_text += text[pos:]
            last_pos = len(text)  # Mark end
            break
        
        elif marker_type == 'note' and current_article:
            # Add note to current article if preservation is enabled
            if getattr(config, 'preserve_article_note_relationships', True):
                continue  # Keep note with article
            else:
                # Split note separately
                note_start = pos
                note_end = _find_next_marker_pos(text, pos + len(marker_text), markers)
                note_text = text[note_start:note_end].strip()
                if note_text and len(note_text) >= config.min_chunk_chars:
                    chunks.append(note_text)
    
    # Handle remaining text
    if current_article and current_article_text.strip():
        article_chunks = _finalize_article_chunk(current_article_text, config)
        chunks.extend(article_chunks)
    elif not current_article and last_pos < len(text):
        remaining_text = text[last_pos:].strip()
        if remaining_text:
            remaining_chunks = _split_by_sentences(remaining_text, config)
            chunks.extend(remaining_chunks)
    
    return [chunk for chunk in chunks if chunk.strip()]


def _split_at_header_boundaries(text: str, header_markers: List[Tuple[int, str]], config: ChunkingConfig) -> List[str]:
    """Split text strictly at header boundaries, ensuring one header per chunk."""
    chunks = []
    last_pos = 0
    
    for pos, marker in header_markers:
        # Add text before this marker as a chunk
        if pos > last_pos:
            chunk_text = text[last_pos:pos].strip()
            if chunk_text and len(chunk_text) >= config.min_chunk_chars:
                chunks.append(chunk_text)
        
        # Find end of this header section
        next_pos = len(text)
        current_idx = header_markers.index((pos, marker))
        if current_idx + 1 < len(header_markers):
            next_pos = header_markers[current_idx + 1][0]
        
        # Add this header section as a chunk
        header_chunk = text[pos:next_pos].strip()
        if header_chunk:
            chunks.append(header_chunk)
        
        last_pos = next_pos
    
    return [chunk for chunk in chunks if chunk.strip()]


def _finalize_article_chunk(article_text: str, config: ChunkingConfig) -> List[str]:
    """Finalize an article chunk, splitting if too large."""
    if len(article_text) <= config.max_chunk_chars:
        return [article_text]
    
    # Article is too large, need to split carefully
    return _split_by_sentences(article_text, config)


def _split_by_sentences(text: str, config: ChunkingConfig) -> List[str]:
    """Split text by sentences with overlap."""
    chunks = []
    boundaries = find_sentence_boundaries(text, config)
    
    if not boundaries:
        # Fallback: split by character count
        start = 0
        while start < len(text):
            end = min(start + config.max_chunk_chars, len(text))
            chunks.append(text[start:end])
            start = end - config.overlap_chars if end < len(text) else end
        return chunks
    
    # Split by sentence boundaries with overlap
    start = 0
    current_chunk = ""
    
    for boundary in boundaries:
        sentence = text[start:boundary]
        
        # If adding this sentence would exceed max chars, finalize current chunk
        if current_chunk and len(current_chunk + sentence) > config.max_chunk_chars:
            if len(current_chunk) >= config.min_chunk_chars:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - config.overlap_chars)
                current_chunk = current_chunk[overlap_start:] + sentence
            else:
                current_chunk += sentence
        else:
            current_chunk += sentence
        
        start = boundary
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _find_next_marker_pos(text: str, start_pos: int, markers: List[Tuple[int, str, str]]) -> int:
    """Find position of next marker after start_pos."""
    for pos, _, _ in markers:
        if pos > start_pos:
            return pos
    return len(text)


def get_database_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get database connection with proper configuration."""
    logger = get_logger(__name__)
    
    if db_path is None:
        try:
            config = get_config()
            if hasattr(config, 'get'):
                db_path = config.get('database', {}).get('path', 'data/db/legal_assistant.db')
            else:
                db_path = 'data/db/legal_assistant.db'
        except Exception as e:
            logger.warning(f"Could not load config, using default database path: {e}")
            db_path = 'data/db/legal_assistant.db'
    
    # Convert to absolute path
    if not os.path.isabs(db_path):
        project_root = Path(__file__).parent.parent
        db_path = project_root / db_path
    
    logger.info(f"Connecting to database: {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_document_data(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Fetch document data with joins for chunking.
    
    Returns list of rows with document, chapter, article, and note information.
    """
    query = """
    SELECT 
        d.document_uid,
        d.title as document_title,
        d.document_type,
        d.section_name AS section,
        d.approval_date,
        d.approval_authority,
        c.chapter_title,
        a.article_number,
        a.article_text,
        n.note_label,
        n.note_text,
        cl.clause_label,
        cl.clause_text
    FROM documents d
    LEFT JOIN chapters c ON d.id = c.document_id
    LEFT JOIN articles a ON c.id = a.chapter_id
    LEFT JOIN notes n ON a.id = n.article_id
    LEFT JOIN clauses cl ON n.id = cl.note_id
    ORDER BY d.document_uid, c.chapter_index, a.article_number, n.note_label, cl.clause_label
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def process_document_row(row: Dict[str, Any], config: ChunkingConfig) -> List[DocumentChunk]:
    """
    Process a single database row into chunks.
    
    Args:
        row: Database row with joined document data
        config: Chunking configuration
        
    Returns:
        List of document chunks
    """
    chunks = []
    
    # Determine what text to chunk and source type
    if row['clause_text']:
        # Chunk clause text
        text = row['clause_text']
        source_type = 'clause'
        primary_text = text
    elif row['note_text']:
        # Chunk note text
        text = row['note_text']
        source_type = 'note'
        primary_text = text
    elif row['article_text']:
        # Chunk article text
        text = row['article_text']
        source_type = 'article'
        primary_text = text
    else:
        # No text content
        return chunks
    
    if not text or not text.strip():
        return chunks
    
    # Normalize text
    normalized_text = normalize_text(text, config)
    
    # Split into chunks
    if config.prefer_article_chunks and source_type == 'article':
        # For articles, prefer to keep whole if possible
        if len(text) <= config.max_chunk_chars * 1.2:  # Allow 20% overage for articles
            text_chunks = [text]
        else:
            text_chunks = split_text_into_chunks(text, config)
    else:
        text_chunks = split_text_into_chunks(text, config)
    
    # Create chunk objects
    for chunk_index, chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue
            
        chunk_normalized = normalize_text(chunk_text, config)
        
        chunk_uid = compute_chunk_uid(
            row['document_uid'],
            row['article_number'],
            row['note_label'], 
            chunk_index,
            chunk_normalized
        )
        
        chunk = DocumentChunk(
            chunk_uid=chunk_uid,
            document_uid=row['document_uid'],
            document_title=row['document_title'] or '',
            document_type=row['document_type'] or '',
            section=row['section'] or '',
            approval_date=row['approval_date'] or '',
            approval_authority=row['approval_authority'] or '',
            chapter_title=row['chapter_title'] or '',
            article_number=row['article_number'] or '',
            note_label=row['note_label'] or '',
            clause_label=row['clause_label'] or '',
            text=chunk_text,
            normalized_text=chunk_normalized,
            chunk_index=chunk_index,
            char_count=len(chunk_text),
            token_count=estimate_token_count(chunk_text),
            source_type=source_type,
            metadata={
                'original_text_length': len(primary_text),
                'total_chunks_for_source': len(text_chunks),
                'has_header_marker': has_header_marker(chunk_text, config),
                'header_marker_count': count_header_markers_in_chunk(chunk_text)
            }
        )
        
        chunks.append(chunk)
    
    return chunks


def create_chunks_from_database(db_path: Optional[str] = None, 
                               config: Optional[ChunkingConfig] = None) -> List[Dict[str, Any]]:
    """
    Create chunks from database content.
    
    Args:
        db_path: Path to SQLite database
        config: Chunking configuration
        
    Returns:
        List of chunk dictionaries
    """
    logger = get_logger(__name__)
    
    if config is None:
        config = load_chunking_config()
    
    logger.info("Starting document chunking process")
    logger.info("شروع فرآیند تکه‌بندی اسناد")
    
    # Connect to database
    conn = get_database_connection(db_path)
    
    try:
        # Fetch document data
        logger.info("Fetching document data from database...")
        document_rows = fetch_document_data(conn)
        
        if not document_rows:
            logger.warning("No document data found in database")
            return []
        
        logger.info(f"Processing {len(document_rows)} database rows")
        
        # Process rows into chunks
        all_chunks = []
        documents_processed = set()
        
        for row in document_rows:
            try:
                chunks = process_document_row(row, config)
                all_chunks.extend(chunks)
                
                if row['document_uid']:
                    documents_processed.add(row['document_uid'])
                    
            except Exception as e:
                logger.error(f"Error processing row for document {row.get('document_uid', 'unknown')}: {e}")
                continue
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents_processed)} documents")
        logger.info(f"{len(all_chunks)} تکه از {len(documents_processed)} سند ایجاد شد")
        
        # Convert chunks to dictionaries
        chunk_dicts = []
        for chunk in all_chunks:
            chunk_dict = {
                "chunk_uid": chunk.chunk_uid,
                "document_uid": chunk.document_uid,
                "document_title": chunk.document_title,
                "document_type": chunk.document_type,
                "section": chunk.section,
                "approval_date": chunk.approval_date,
                "approval_authority": chunk.approval_authority,
                "chapter_title": chunk.chapter_title,
                "article_number": chunk.article_number,
                "note_label": chunk.note_label,
                "clause_label": chunk.clause_label,
                "text": chunk.text,
                "normalized_text": chunk.normalized_text,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "token_count": chunk.token_count,
                "source_type": chunk.source_type,
                "metadata": chunk.metadata
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
        
    finally:
        conn.close()


def write_chunks_json(chunks: List[Dict[str, Any]], output_path: Optional[str] = None) -> None:
    """Write chunks to JSON file."""
    logger = get_logger(__name__)
    
    if output_path is None:
        output_path = "data/processed_phase_3/chunks.json"
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add statistics
    output_data = {
        "chunks": chunks,
        "statistics": {
            "total_chunks": len(chunks),
            "documents_represented": len(set(chunk["document_uid"] for chunk in chunks)),
            "source_types": {
                "article": len([c for c in chunks if c["source_type"] == "article"]),
                "note": len([c for c in chunks if c["source_type"] == "note"]),
                "clause": len([c for c in chunks if c["source_type"] == "clause"])
            },
            "avg_chunk_chars": sum(c["char_count"] for c in chunks) // len(chunks) if chunks else 0,
            "avg_chunk_tokens": sum(c["token_count"] for c in chunks) // len(chunks) if chunks else 0
        }
    }
    
    logger.info(f"Writing {len(chunks)} chunks to {output_path}")
    logger.info(f"نوشتن {len(chunks)} تکه در فایل {output_path}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info("Chunks written successfully")
    logger.info("تکه‌ها با موفقیت نوشته شدند")


def main():
    """Main entry point for CLI execution."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting chunker process")
        logger.info("شروع فرآیند تکه‌بند")
        
        # Load configuration
        config = load_chunking_config()
        
        # Create chunks from database
        chunks = create_chunks_from_database(config=config)
        
        if not chunks:
            logger.warning("No chunks created")
            return
        
        # Write to output file
        write_chunks_json(chunks)
        
        logger.info("Chunking process completed successfully")
        logger.info("فرآیند تکه‌بندی با موفقیت تکمیل شد")
        
        print(f"[SUCCESS] Created {len(chunks)} chunks")
        print(f"[SUCCESS] {len(chunks)} تکه ایجاد شد")
        
    except Exception as e:
        logger.error(f"Chunking process failed: {e}")
        logger.error(f"فرآیند تکه‌بندی ناموفق بود: {e}")
        print(f"[ERROR] Chunking failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()