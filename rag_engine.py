#!/usr/bin/env python3
"""
Legal Persian RAG Engine
موتور RAG حقوقی فارسی

A comprehensive Persian legal document retrieval and answer generation system.
سیستم جامع بازیابی اسناد حقوقی و تولید پاسخ به زبان فارسی
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np

# Optional imports for vector backends
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

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Data class for retrieved chunks with metadata."""
    chunk_id: str
    content: str
    similarity_score: float
    document_title: str
    document_type: str
    article_number: Optional[str] = None
    note_label: Optional[str] = None
    clause_label: Optional[str] = None
    section: Optional[str] = None
    approval_authority: Optional[str] = None
    chunk_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class RAGEngineError(Exception):
    """Custom exception for RAG engine errors."""
    pass


class LegalRAGEngine:
    """
    موتور RAG حقوقی فارسی
    Legal Persian RAG Engine
    
    A comprehensive system for retrieval-augmented generation on Persian legal documents.
    """
    
    def __init__(self, config_path: str = "rag_config.json"):
        """
        Initialize the Legal RAG Engine.
        
        Args:
            config_path: Path to RAG configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize logger
        self.logger = logging.getLogger('rag_engine')
        
        # Initialize components
        self.vector_backend = None
        self.chunks = []
        self.db_connection = None
        self.embedding_model = None
        self.prompt_templates = {}
        self.llm_client = None
        
        # Load all components
        self._initialize_components()
        
        self.logger.info("Legal RAG Engine initialized successfully")
        self.logger.info("موتور RAG حقوقی با موفقیت راه‌اندازی شد")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load RAG configuration from JSON file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            # Create default config if it doesn't exist
            default_config = self._create_default_config()
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Created default config file: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Config loaded from: {config_file}")
            return config
        except Exception as e:
            raise RAGEngineError(f"Failed to load config: {e}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default RAG configuration."""
        return {
            "vector_store": {
                "type": "faiss",
                "index_path": "data/processed_phase_3/vector_db/faiss/faiss.index",
                "embeddings_path": "data/processed_phase_3/embeddings.npy"
            },
            "chunks_file": "data/processed_phase_3/chunks.json",
            "database_path": "data/db/legal_assistant.db",
            "prompt_templates_path": "phase_4_llm_rag/prompt_templates.json",
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b-instruct",
                "backup_model": "mistral:7b",
                "temperature": 0.2,
                "max_tokens": 1800,
                "timeout_s": 60
            }
        }
    
    def _initialize_components(self):
        """Initialize all RAG components."""
        self.logger.info("Initializing RAG components...")
        
        # Load vector database
        self._load_vector_backend()
        
        # Load chunks
        self._load_chunks()
        
        # Connect to SQLite database
        self._connect_database()
        
        # Load embedding model
        self._load_embedding_model()
        
        # Load prompt templates
        self._load_prompt_templates()
        
        # Initialize LLM client
        self._initialize_llm_client()
    
    def _load_vector_backend(self):
        """Load vector database backend (FAISS or Chroma)."""
        vector_store_config = self.config.get("vector_store", {})
        vector_store_type = vector_store_config.get("type", "faiss").lower()
        
        if vector_store_type == "faiss":
            self._load_faiss_backend()
            self.logger.info("Vector backend loaded: FAISS")
        elif vector_store_type == "chroma":
            self._load_chroma_backend()
            self.logger.info("Vector backend loaded: Chroma")
        else:
            raise RAGEngineError(f"Unsupported vector store type: {vector_store_type}")
    
    def _load_faiss_backend(self):
        """Load FAISS vector database."""
        if not HAS_FAISS:
            raise RAGEngineError("FAISS library not available")
        
        vector_store_config = self.config.get("vector_store", {})
        index_path = vector_store_config.get("index_path")
        embeddings_path = vector_store_config.get("embeddings_path")
        
        if not index_path:
            raise RAGEngineError("vector_store.index_path is required for FAISS backend")
        
        index_path = Path(index_path)
        if not index_path.exists():
            raise RAGEngineError(f"FAISS index file not found: {index_path}")
        
        # Load the persisted FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load embeddings if path is provided
        embeddings = None
        if embeddings_path:
            embeddings_path = Path(embeddings_path)
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
            else:
                self.logger.warning(f"Embeddings file not found: {embeddings_path}")
        
        # Load metadata
        metadata_file = index_path.parent / "embeddings_meta.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            self.logger.warning(f"Metadata file not found: {metadata_file}")
        
        self.vector_backend = {
            'type': 'faiss',
            'index': index,
            'embeddings': embeddings,
            'metadata': metadata,
            'chunk_uid_order': metadata.get('chunk_uid_order', [])
        }
    
    def _load_chroma_backend(self):
        """Load ChromaDB vector database."""
        if not HAS_CHROMA:
            raise RAGEngineError("ChromaDB library not available")
        
        vector_store_config = self.config.get("vector_store", {})
        db_path = vector_store_config.get("db_path", "data/processed_phase_3/vector_db/chroma")
        collection_name = vector_store_config.get("collection_name", "legal_chunks")
        
        if not db_path:
            raise RAGEngineError("vector_store.db_path is required for ChromaDB backend")
        
        db_path = Path(db_path)
        
        # Create ChromaDB client
        try:
            client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
            
            # Get or create collection
            try:
                collection = client.get_collection(collection_name)
            except:
                # If collection doesn't exist, we'll need to create it from our data
                collection = client.create_collection(collection_name)
                # TODO: Populate collection with embeddings and chunks
            
            self.vector_backend = {
                'type': 'chroma',
                'client': client,
                'collection': collection
            }
        except Exception as e:
            raise RAGEngineError(f"Failed to initialize ChromaDB: {e}")
    
    def _load_chunks(self):
        """Load text chunks from JSON file."""
        chunks_file_path = self.config.get("chunks_file")
        if not chunks_file_path:
            raise RAGEngineError("chunks_file is required in config")
        
        chunks_file = Path(chunks_file_path)
        if not chunks_file.exists():
            raise RAGEngineError(f"Chunks file not found: {chunks_file}")
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                self.chunks = data.get('chunks', [])
            elif isinstance(data, list):
                self.chunks = data
            else:
                raise ValueError("Invalid chunks file format")
            
            self.logger.info(f"Loaded {len(self.chunks)} chunks")
            
        except Exception as e:
            raise RAGEngineError(f"Failed to load chunks: {e}")
    
    def _connect_database(self):
        """Connect to SQLite database for structural filters."""
        # Use the new database_path config key
        db_path_str = self.config.get("database_path")
        if not db_path_str:
            raise RAGEngineError("database_path is required in config")
        
        db_path = Path(db_path_str)
        if not db_path.exists():
            raise RAGEngineError(f"Database not found: {db_path}")
        
        try:
            self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
            self.db_connection.row_factory = sqlite3.Row
            self.logger.info("Database connection established")
        except Exception as e:
            raise RAGEngineError(f"Database connection failed: {e}")
    
    def _load_embedding_model(self):
        """Load sentence embedding model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.warning("Sentence Transformers not available - search will be limited")
            return
        
        try:
            # Get model name from metadata or use default
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Default
            
            # Try to get from vector backend metadata if available
            if self.vector_backend and 'metadata' in self.vector_backend:
                metadata_model = self.vector_backend['metadata'].get('model_name')
                if metadata_model:
                    model_name = metadata_model
            
            # Remove 'sentence-transformers/' prefix if present for loading
            if model_name.startswith('sentence-transformers/'):
                model_name = model_name.replace('sentence-transformers/', '')
            
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Embedding model loaded: {model_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _load_prompt_templates(self):
        """Load prompt templates from JSON file."""
        templates_file_path = self.config.get("prompt_templates_path")
        if not templates_file_path:
            raise RAGEngineError("prompt_templates_path is required in config")
        
        templates_file = Path(templates_file_path)
        if not templates_file.exists():
            raise RAGEngineError(f"Prompt templates file not found: {templates_file}")
        
        try:
            with open(templates_file, 'r', encoding='utf-8') as f:
                self.prompt_templates = json.load(f)
            self.logger.info("Prompt templates loaded")
        except Exception as e:
            raise RAGEngineError(f"Failed to load prompt templates: {e}")
    
    def _create_default_templates(self) -> Dict[str, str]:
        """Create default prompt templates."""
        return {
            "default": """شما یک دستیار حقوقی هستید که به سوالات حقوقی بر اساس اسناد قانونی پاسخ می‌دهید.

سوال: {question}

اسناد مرتبط:
{retrieved_chunks}

لطفاً بر اساس اسناد ارائه شده، پاسخی جامع و دقیق به زبان فارسی ارائه دهید. در پاسخ خود حتماً به مواد، تبصره‌ها و بندهای مرتبط اشاره کنید.

پاسخ:""",
            
            "detailed": """شما یک مشاور حقوقی متخصص هستید. به سوال زیر بر اساس مجموعه اسناد حقوقی ارائه شده پاسخ دهید.

سوال کاربر: {question}

اسناد و مواد حقوقی مرتبط:
{retrieved_chunks}

راهنمای پاسخ:
1. پاسخ خود را به زبان فارسی و با جزئیات کامل ارائه دهید
2. به طور صریح به شماره مواد، تبصره‌ها و بندهای مرتبط اشاره کنید
3. در صورت وجود ابهام یا تفسیرهای مختلف، آن‌ها را ذکر کنید
4. منابع و مراجع قانونی را در پایان پاسخ فهرست کنید

پاسخ تخصصی:""",
            
            "brief": """سوال: {question}

مواد قانونی مرتبط:
{retrieved_chunks}

پاسخ مختصر و دقیق به زبان فارسی با ذکر شماره مواد مرتبط:"""
        }
    
    def _initialize_llm_client(self):
        """Initialize LLM client based on provider in config."""
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "ollama").lower()
        
        if provider == "ollama":
            self._initialize_ollama_client()
        else:
            self.logger.warning(f"Unsupported LLM provider: {provider}. Defaulting to Ollama.")
            self._initialize_ollama_client()
    
    def _initialize_ollama_client(self):
        """Initialize Ollama client."""
        if not HAS_REQUESTS:
            self.logger.warning("Requests library not available - LLM generation disabled")
            return
        
        llm_config = self.config.get("llm", {})
        
        # Use new config keys
        self.llm_client = {
            'type': 'ollama',
            'base_url': llm_config.get("base_url", "http://localhost:11434"),
            'model': llm_config.get("model", "qwen2.5:7b-instruct"),
            'timeout': llm_config.get("timeout_s", 60),
            'max_tokens': llm_config.get("max_tokens", 1800),
            'temperature': llm_config.get("temperature", 0.2)
        }
        
        # Test connection
        try:
            response = requests.get(
                f"{self.llm_client['base_url']}/api/version",
                timeout=5
            )
            if response.status_code == 200:
                self.logger.info("Ollama client initialized successfully")
            else:
                self.logger.warning("Ollama server not responding properly")
        except Exception as e:
            self.logger.warning(f"Ollama connection test failed: {e}")
    
    def _initialize_api_client(self):
        """Initialize API client."""
        # Placeholder for other API providers (OpenAI, etc.)
        self.llm_client = {
            'type': 'api',
            'base_url': self.config["llm"].get("base_url"),
            'api_key': self.config["llm"].get("api_key"),
            'model': self.config["llm"]["model"]
        }
        self.logger.info("API client configured")
    
    def retrieve(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a question.
        
        Args:
            question: User question in Persian
            top_k: Number of chunks to retrieve
            filters: Optional database filters (document_type, section, article_range, etc.)
            
        Returns:
            List of RetrievedChunk objects with content and metadata
        """
        if not question.strip():
            raise RAGEngineError("Empty question provided")
        
        # Validate top_k
        max_k = 20  # Default max for safety
        top_k = min(max(1, top_k), max_k)
        
        self.logger.info(f"Retrieving {top_k} chunks for question: {question[:50]}...")
        
        # Get vector search results
        vector_results = self._vector_search(question, top_k * 2)  # Get more for filtering
        
        # Apply database filters if provided
        if filters:
            vector_results = self._apply_database_filters(vector_results, filters)
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        similarity_threshold = 0.4  # Override with reasonable threshold for testing
        
        for result in vector_results[:top_k]:
            # self.logger.info(f"Processing result: score={result['similarity_score']}, threshold={similarity_threshold}")
            if result["similarity_score"] < similarity_threshold:
                # self.logger.info(f"Skipping result due to low similarity: {result['similarity_score']} < {similarity_threshold}")
                continue
                
            # Get chunk metadata from database
            chunk_metadata = self._get_chunk_metadata(result["chunk_id"])
            
            retrieved_chunk = RetrievedChunk(
                chunk_id=result["chunk_id"],
                content=result["content"],
                similarity_score=result["similarity_score"],
                document_title=chunk_metadata.get("document_title", "نامشخص"),
                document_type=chunk_metadata.get("document_type", "نامشخص"),
                article_number=chunk_metadata.get("article_number"),
                note_label=chunk_metadata.get("note_label"),
                clause_label=chunk_metadata.get("clause_label"),
                section=chunk_metadata.get("section"),
                approval_authority=chunk_metadata.get("approval_authority"),
                chunk_metadata=chunk_metadata
            )
            
            retrieved_chunks.append(retrieved_chunk)
        
        self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        return retrieved_chunks
    
    def _vector_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        if not self.embedding_model:
            raise RAGEngineError("Embedding model not available")
        
        # Encode question
        try:
            question_embedding = self.embedding_model.encode([question])[0]
        except Exception as e:
            raise RAGEngineError(f"Failed to encode question: {e}")
        
        # Search based on backend type
        if self.vector_backend['type'] == 'faiss':
            return self._search_faiss(question_embedding, top_k)
        elif self.vector_backend['type'] == 'chroma':
            return self._search_chroma(question_embedding, top_k)
        else:
            raise RAGEngineError("No vector backend available")
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        index = self.vector_backend['index']
        chunk_uid_order = self.vector_backend['chunk_uid_order']
        
        # Prepare query
        query_embedding = query_embedding.astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
        
        # Debug: log similarity scores (can be removed in production)
        # self.logger.info(f"FAISS search scores: {scores[0][:5]}")
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(chunk_uid_order):  # Invalid result
                continue
            
            chunk_id = chunk_uid_order[idx]
            
            # Find chunk content
            chunk_content = None
            for chunk in self.chunks:
                if chunk.get('uid', chunk.get('chunk_uid', '')) == chunk_id:
                    chunk_content = chunk.get('content', chunk.get('text', ''))
                    break
            
            if chunk_content:
                results.append({
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "similarity_score": float(score)
                })
        
        return results
    
    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search using ChromaDB."""
        collection = self.vector_backend['collection']
        
        # Search
        search_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        results = []
        if search_results['ids'] and search_results['ids'][0]:
            for chunk_id, document, distance in zip(
                search_results['ids'][0],
                search_results['documents'][0],
                search_results['distances'][0]
            ):
                results.append({
                    "chunk_id": chunk_id,
                    "content": document,
                    "similarity_score": 1.0 - distance  # Convert distance to similarity
                })
        
        return results
    
    def _apply_database_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply database filters to search results."""
        if not self.db_connection or not filters:
            return results
        
        # Build filter query
        where_clauses = []
        params = []
        
        if "document_type" in filters:
            where_clauses.append("d.document_type = ?")
            params.append(filters["document_type"])
        
        if "section" in filters:
            where_clauses.append("d.section = ?")
            params.append(filters["section"])
        
        if "approval_authority" in filters:
            where_clauses.append("d.approval_authority = ?")
            params.append(filters["approval_authority"])
        
        if "article_range" in filters:
            start, end = filters["article_range"]
            where_clauses.append("CAST(a.article_number AS INTEGER) BETWEEN ? AND ?")
            params.extend([start, end])
        
        if not where_clauses:
            return results
        
        # Query database for allowed chunk IDs
        query = f"""
        SELECT DISTINCT
            -- We'll need to match chunk UIDs, but this is a simplified approach
            d.document_uid,
            a.article_number,
            n.note_label
        FROM documents d
        LEFT JOIN chapters c ON d.id = c.document_id
        LEFT JOIN articles a ON c.id = a.chapter_id
        LEFT JOIN notes n ON a.id = n.article_id
        WHERE {' AND '.join(where_clauses)}
        """
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            allowed_items = cursor.fetchall()
            
            # For now, we'll do a simple content-based filtering
            # In a production system, you'd want proper chunk-to-database mapping
            allowed_content_keywords = set()
            for row in allowed_items:
                if row["article_number"]:
                    allowed_content_keywords.add(f"ماده {row['article_number']}")
                if row["note_label"]:
                    allowed_content_keywords.add(f"تبصره {row['note_label']}")
            
            # Filter results based on content matching
            if allowed_content_keywords:
                filtered_results = []
                for result in results:
                    content = result["content"].lower()
                    if any(keyword.lower() in content for keyword in allowed_content_keywords):
                        filtered_results.append(result)
                return filtered_results
            
        except Exception as e:
            self.logger.warning(f"Database filtering failed: {e}")
        
        return results
    
    def _get_chunk_metadata(self, chunk_id: str) -> Dict[str, Any]:
        """Get chunk metadata from database."""
        if not self.db_connection:
            return {}
        
        # Find the chunk in our loaded chunks
        chunk_data = None
        for chunk in self.chunks:
            if chunk.get('uid', chunk.get('chunk_uid', '')) == chunk_id:
                chunk_data = chunk
                break
        
        if not chunk_data:
            return {}
        
        # Extract metadata from chunk
        metadata = {
            "document_title": chunk_data.get("document_title", "نامشخص"),
            "document_type": chunk_data.get("document_type", "نامشخص"),
            "article_number": chunk_data.get("article_number"),
            "note_label": chunk_data.get("note_label"),
            "clause_label": chunk_data.get("clause_label"),
            "section": chunk_data.get("section"),
            "approval_authority": chunk_data.get("approval_authority")
        }
        
        return metadata
    
    def build_prompt(self, question: str, retrieved_chunks: List[RetrievedChunk], template_name: str = "default") -> str:
        """
        Build prompt from template with retrieved chunks.
        
        Args:
            question: User question
            retrieved_chunks: List of retrieved chunks
            template_name: Template to use from prompt_templates.json
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self.prompt_templates:
            template_name = "default"
        
        template = self.prompt_templates[template_name]
        
        # Format retrieved chunks
        chunks_text = self._format_chunks_for_prompt(retrieved_chunks)
        
        # Build prompt
        prompt = template.format(
            question=question,
            retrieved_chunks=chunks_text
        )
        
        # Trim if too long (default max length)
        max_length = 8000  # Default max content length
        if len(prompt) > max_length:
            # Trim from the chunks section
            available_space = max_length - len(template.format(question=question, retrieved_chunks=""))
            if available_space > 0:
                chunks_text = chunks_text[:available_space-100] + "...\n[محتوا کوتاه شده]"
                prompt = template.format(question=question, retrieved_chunks=chunks_text)
        
        return prompt
    
    def _format_chunks_for_prompt(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks for prompt inclusion."""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            # Build citation
            citation_parts = []
            if chunk.document_title and chunk.document_title != "نامشخص":
                citation_parts.append(f"سند: {chunk.document_title}")
            if chunk.article_number:
                citation_parts.append(f"ماده {chunk.article_number}")
            if chunk.note_label:
                citation_parts.append(f"تبصره {chunk.note_label}")
            if chunk.clause_label:
                citation_parts.append(f"بند {chunk.clause_label}")
            
            citation = " - ".join(citation_parts) if citation_parts else f"قطعه {i}"
            
            formatted_chunk = f"""[{i}] {citation}
{chunk.content.strip()}
"""
            formatted_chunks.append(formatted_chunk)
        
        return "\n".join(formatted_chunks)
    
    def generate_answer(self, prompt: str) -> Dict[str, Any]:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Dict with answer, citations, and metadata
        """
        if not self.llm_client:
            return {
                "answer": "سیستم تولید پاسخ در دسترس نیست. لطفاً تنظیمات LLM را بررسی کنید.",
                "citations": [],
                "error": "LLM client not available"
            }
        
        start_time = time.time()
        
        try:
            if self.llm_client['type'] == 'ollama':
                result = self._generate_with_ollama(prompt)
            elif self.llm_client['type'] == 'api':
                result = self._generate_with_api(prompt)
            else:
                raise RAGEngineError("Unsupported LLM client type")
            
            # Extract citations from answer
            citations = self._extract_citations(result["answer"])
            
            result.update({
                "citations": citations,
                "generation_time": time.time() - start_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return {
                "answer": f"خطا در تولید پاسخ: {str(e)}",
                "citations": [],
                "error": str(e),
                "generation_time": time.time() - start_time
            }
    
    def _generate_with_ollama(self, prompt: str) -> Dict[str, Any]:
        """Generate answer using Ollama."""
        if not HAS_REQUESTS:
            raise RAGEngineError("Requests library not available")
        
        url = f"{self.llm_client['base_url']}/api/generate"
        
        payload = {
            "model": self.llm_client['model'],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.llm_client.get("temperature", 0.2),
                "num_predict": self.llm_client.get("max_tokens", 1800)
            }
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.llm_client['timeout']
        )
        
        if response.status_code != 200:
            raise RAGEngineError(f"Ollama API error: {response.status_code}")
        
        result = response.json()
        
        return {
            "answer": result.get("response", "").strip(),
            "model": self.llm_client['model'],
            "provider": "ollama"
        }
    
    def _generate_with_api(self, prompt: str) -> Dict[str, Any]:
        """Generate answer using API provider."""
        # Placeholder for other API implementations
        raise RAGEngineError("API provider not implemented")
    
    def _extract_citations(self, answer: str) -> List[str]:
        """Extract citations from the generated answer."""
        citations = []
        
        # Look for common Persian legal reference patterns
        import re
        
        patterns = [
            r'ماده\s+(\d+)',  # Article numbers
            r'تبصره\s+(\d+)',  # Note numbers
            r'بند\s+(\w+)',   # Clause labels
            r'فصل\s+(\w+)',   # Chapter labels
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                if pattern.startswith('ماده'):
                    citations.append(f"ماده {match}")
                elif pattern.startswith('تبصره'):
                    citations.append(f"تبصره {match}")
                elif pattern.startswith('بند'):
                    citations.append(f"بند {match}")
                elif pattern.startswith('فصل'):
                    citations.append(f"فصل {match}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        return unique_citations
    
    def answer(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve → build_prompt → generate_answer.
        
        Args:
            question: User question in Persian
            **kwargs: Additional parameters (top_k, filters, template_name, etc.)
            
        Returns:
            Dict with answer, citations, retrieved_chunks, and metadata
        """
        if not question or not question.strip():
            return {
                "answer": "لطفاً سوال خود را وارد کنید.",
                "citations": [],
                "retrieved_chunks": [],
                "error": "Empty question"
            }
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant chunks
            top_k = kwargs.get('top_k', 5)  # Default top_k
            filters = kwargs.get('filters')
            
            retrieved_chunks = self.retrieve(question, top_k=top_k, filters=filters)
            
            if not retrieved_chunks:
                return {
                    "answer": "متأسفانه هیچ سند مرتبط با سوال شما یافت نشد. لطفاً سوال خود را بازنویسی کنید.",
                    "citations": [],
                    "retrieved_chunks": [],
                    "warning": "No relevant chunks found"
                }
            
            # Step 2: Build prompt
            template_name = kwargs.get('template_name', 'default')
            prompt = self.build_prompt(question, retrieved_chunks, template_name)
            
            # Step 3: Generate answer
            generation_result = self.generate_answer(prompt)
            
            # Combine results
            result = {
                "question": question,
                "answer": generation_result["answer"],
                "citations": generation_result["citations"],
                "retrieved_chunks": [chunk.to_dict() for chunk in retrieved_chunks],
                "metadata": {
                    "total_time": time.time() - start_time,
                    "generation_time": generation_result.get("generation_time", 0),
                    "chunks_count": len(retrieved_chunks),
                    "template_used": template_name,
                    "model": generation_result.get("model"),
                    "provider": generation_result.get("provider"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
            if "error" in generation_result:
                result["error"] = generation_result["error"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"RAG pipeline failed: {e}")
            return {
                "question": question,
                "answer": f"خطا در پردازش سوال: {str(e)}",
                "citations": [],
                "retrieved_chunks": [],
                "error": str(e),
                "metadata": {
                    "total_time": time.time() - start_time,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics and health info."""
        stats = {
            "chunks_loaded": len(self.chunks),
            "vector_backend": self.vector_backend['type'] if self.vector_backend else None,
            "database_connected": self.db_connection is not None,
            "embedding_model_loaded": self.embedding_model is not None,
            "llm_client_available": self.llm_client is not None,
            "prompt_templates_count": len(self.prompt_templates),
            "config_file": self.config_path,
        }
        
        if self.vector_backend and self.vector_backend['type'] == 'faiss':
            stats["faiss_index_size"] = self.vector_backend['index'].ntotal
        
        return stats
    
    def close(self):
        """Close database connections and cleanup resources."""
        if self.db_connection:
            self.db_connection.close()
            self.logger.info("Database connection closed")


def main():
    """Demo function to test the RAG engine."""
    try:
        # Initialize RAG engine
        engine = LegalRAGEngine()
        
        # Print stats
        stats = engine.get_stats()
        print("RAG Engine Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test questions
        test_questions = [
            "مجازات سرقت چیست؟",
            "تعریف قانونی ازدواج چه است؟",
            "شرایط دریافت حقوق بازنشستگی چیست؟"
        ]
        
        for question in test_questions:
            print(f"\n{'='*50}")
            print(f"سوال: {question}")
            print(f"{'='*50}")
            
            result = engine.answer(question)
            
            print(f"پاسخ: {result['answer']}")
            print(f"منابع: {', '.join(result['citations'])}")
            print(f"تعداد قطعات بازیابی شده: {result['metadata']['chunks_count']}")
            print(f"زمان کل: {result['metadata']['total_time']:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'engine' in locals():
            engine.close()


if __name__ == "__main__":
    main()