import json
import sqlite3
import argparse
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import re

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from .api_connections import get_llm_client
except ImportError:
    # Fallback for direct execution or when relative imports fail
    from api_connections import get_llm_client


# Configure logging for Persian messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_engine.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_faiss_index(index_path: str, embeddings_path: str = None):
    """Load FAISS index and embeddings if available."""
    if not faiss:
        raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    
    index = faiss.read_index(index_path)
    embeddings = None
    
    if embeddings_path and os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
    
    return index, embeddings


def load_chroma_collection(db_path: str, collection_name: str = "legal_docs"):
    """Load Chroma collection from persistent database."""
    if not chromadb:
        raise ImportError("ChromaDB not installed. Install with: pip install chromadb")
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    return collection


def validate_citations(citations: List[Dict], db_path: str) -> List[Dict]:
    """Validate that cited articles/notes exist in the database."""
    if not citations:
        return []
    
    validated_citations = []
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            for citation in citations:
                document_uid = citation.get('document_uid')
                article_number = citation.get('article_number')
                note_label = citation.get('note_label')
                
                # Check if the citation exists in the database
                if note_label:
                    cursor.execute("""
                        SELECT COUNT(*) FROM chunks 
                        WHERE document_uid = ? AND article_number = ? AND note_label = ?
                    """, (document_uid, article_number, note_label))
                else:
                    cursor.execute("""
                        SELECT COUNT(*) FROM chunks 
                        WHERE document_uid = ? AND article_number = ?
                    """, (document_uid, article_number))
                
                if cursor.fetchone()[0] > 0:
                    validated_citations.append(citation)
                else:
                    logger.warning(f"Invalid citation: {citation}")
    
    except sqlite3.Error as e:
        logger.error(f"Database error during citation validation: {e}")
    
    return validated_citations


class LegalRAGEngine:
    """Legal document RAG (Retrieval-Augmented Generation) engine."""
    
    def __init__(self, config_path: str = "Rag_config.json"):
        """Initialize the RAG engine with configuration."""
        # Handle relative paths based on module location
        if not os.path.isabs(config_path):
            module_dir = os.path.dirname(__file__)
            self.config_path = os.path.join(module_dir, config_path)
        else:
            self.config_path = config_path
        self.config = self._load_config()
        
        # Validate vector_store configuration
        if not self.config.get('vector_store'):
            raise ValueError(
                "تنظیمات vector_store موجود نیست.\n"
                "Vector store configuration is missing.\n"
                "لطفاً بخش vector_store را با type، index_path و embeddings_path در فایل تنظیمات اضافه کنید.\n"
                "Please add vector_store section with type, index_path, and embeddings_path to the config file."
            )
        
        self.chunks = self._load_chunks()
        self.embeddings_meta = self._load_embeddings_meta()
        self.embedding_model = self._load_embedding_model()
        self.vector_store = self._connect_vector_store()
        db_path = self.config.get("database_path", "legal_assistant.db")
        self.db_path = os.path.abspath(db_path) if not os.path.isabs(db_path) else db_path
        self.prompt_templates = self._load_prompt_templates()
        self.llm_client = self._prepare_llm_client()
        
        # Log LLM client info in Persian if available
        if self.llm_client and hasattr(self.llm_client, 'model') and hasattr(self.llm_client, 'base_url'):
            logger.info(f"مدل LLM فعال: {self.llm_client.model} | آدرس Ollama: {self.llm_client.base_url}")
        
        logger.info("LegalRAGEngine initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
    
    def _load_chunks(self) -> List[Dict]:
        """Load chunks from chunks.json file."""
        chunks_path = self.config.get("chunks_file", "chunks.json")
        if not os.path.isabs(chunks_path):
            chunks_path = os.path.abspath(chunks_path)
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different chunk file formats
            if isinstance(data, dict) and 'chunks' in data:
                chunks = data['chunks']
            elif isinstance(data, list):
                chunks = data
            else:
                chunks = []
            
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
            return chunks
        except FileNotFoundError:
            logger.warning(f"Chunks file not found: {chunks_path}. Starting with empty chunks.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chunks file: {e}")
            raise
    
    def _load_embeddings_meta(self) -> Dict[str, Any]:
        """Load embeddings metadata from embeddings_meta.json file."""
        embeddings_path = self.config.get('vector_store', {}).get('embeddings_path', '')
        if embeddings_path:
            # Infer metadata path from embeddings path
            embeddings_dir = os.path.dirname(embeddings_path)
            meta_path = os.path.join(embeddings_dir, 'embeddings_meta.json')
        else:
            meta_path = 'data/processed_phase_3/embeddings_meta.json'
        
        if not os.path.isabs(meta_path):
            meta_path = os.path.abspath(meta_path)
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            logger.info(f"Loaded embeddings metadata from {meta_path}")
            return meta
        except FileNotFoundError:
            logger.warning(f"Embeddings metadata file not found: {meta_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in embeddings metadata file: {e}")
            return {}
    
    def _load_embedding_model(self) -> Optional[object]:
        """Load the embedding model used for encoding queries."""
        if not SentenceTransformer:
            logger.warning("sentence-transformers not available, falling back to keyword search")
            return None
        
        try:
            # Get model name from metadata or use default
            model_name = self.embeddings_meta.get('model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            # Remove 'sentence-transformers/' prefix if present for loading
            if model_name.startswith('sentence-transformers/'):
                model_name = model_name.replace('sentence-transformers/', '')
            
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}. Falling back to keyword search.")
            return None
    
    def _connect_vector_store(self):
        """Connect to vector store (FAISS or Chroma) based on configuration."""
        vector_store_type = self.config.get('vector_store', {}).get('type', 'faiss').lower()
        
        if vector_store_type == 'faiss':
            index_path = self.config.get('vector_store', {}).get('index_path', 'faiss_index.bin')
            embeddings_path = self.config.get('vector_store', {}).get('embeddings_path')
            try:
                index, embeddings = load_faiss_index(index_path, embeddings_path)
                logger.info("Connected to FAISS vector store")
                return {'type': 'faiss', 'index': index, 'embeddings': embeddings}
            except (FileNotFoundError, ImportError) as e:
                logger.warning(f"Failed to load FAISS: {e}")
                return None
        
        elif vector_store_type == 'chroma':
            db_path = self.config.get('vector_store', {}).get('db_path', 'chroma_db')
            collection_name = self.config.get('vector_store', {}).get('collection_name', 'legal_docs')
            try:
                collection = load_chroma_collection(db_path, collection_name)
                logger.info("Connected to Chroma vector store")
                return {'type': 'chroma', 'collection': collection}
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to load Chroma: {e}")
                return None
        
        else:
            logger.error(f"Unsupported vector store type: {vector_store_type}")
            return None
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from JSON file."""
        templates_path = self.config.get("prompt_templates_path", "prompt_templates.json")
        if not os.path.isabs(templates_path):
            templates_path = os.path.abspath(templates_path)
        try:
            with open(templates_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            logger.info(f"Loaded prompt templates from {templates_path}")
            return templates
        except FileNotFoundError:
            logger.error(f"Prompt templates file not found: {templates_path}")
            # Return default templates
            return {
                "default": "سؤال:\n{question}\n\nمتون بازیابی‌شده:\n{retrieved_text}\n\nلطفاً به فارسی پاسخ دهید و شمارهٔ ماده/تبصره و نام قانون را ذکر کنید.",
                "compare": "هدف: مقایسه یا تضاد بین دو متن حقوقی.\nسؤال:\n{question}\n\nمتون بازیابی‌شده:\n{retrieved_text}\n\nلطفاً شباهت‌ها و تفاوت‌ها را توضیح دهید و به مواد یا تبصره‌های مرتبط اشاره کنید.",
                "draft": "هدف: تهیهٔ پیش‌نویس متن حقوقی.\nشرح درخواست:\n{question}\n\nمنابع مرتبط:\n{retrieved_text}\n\nپیش‌نویس پیشنهادی را بر اساس چارچوب‌های موجود و با ذکر استنادات ارائه دهید."
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in templates file: {e}")
            raise
    
    def _prepare_llm_client(self):
        """Prepare LLM client via api_connections.py or direct Ollama call."""
        try:
            client = get_llm_client(self.config)
            logger.info("LLM client prepared successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to prepare LLM client: {e}")
            return None
    
    def retrieve(self, question: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search vector index for most relevant chunks with optional metadata filtering."""
        if not self.vector_store:
            logger.warning("Vector store not available, using fallback keyword search")
            # Get allowed document UIDs if filters are provided
            allowed_uids = None
            if filters:
                allowed_uids = self._apply_metadata_filters(filters)
                if not allowed_uids:
                    logger.info("No documents match the provided filters")
                    return []
            return self._fallback_keyword_search(question, top_k, allowed_uids)
        
        # Get allowed document UIDs if filters are provided
        allowed_uids = None
        if filters:
            allowed_uids = self._apply_metadata_filters(filters)
            if not allowed_uids:
                logger.info("No documents match the provided filters")
                return []
        
        # Perform vector search based on store type
        if self.vector_store['type'] == 'faiss':
            return self._search_faiss(question, top_k, allowed_uids)
        elif self.vector_store['type'] == 'chroma':
            return self._search_chroma(question, top_k, allowed_uids)
        else:
            # No vector store available, use fallback
            return self._fallback_keyword_search(question, top_k, allowed_uids)
    
    def _apply_metadata_filters(self, filters: Dict) -> Optional[List[str]]:
        """Apply metadata filters by querying SQLite DB to get allowed document UIDs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic WHERE clause based on filters
                where_conditions = []
                params = []
                
                for key, value in filters.items():
                    if key == "document_type":
                        where_conditions.append("document_type = ?")
                        params.append(value)
                    elif key == "section":
                        where_conditions.append("section = ?")
                        params.append(value)
                    elif key == "document_uid":
                        where_conditions.append("document_uid = ?")
                        params.append(value)
                
                if where_conditions:
                    query = f"SELECT DISTINCT document_uid FROM chunks WHERE {' AND '.join(where_conditions)}"
                    cursor.execute(query, params)
                    result = cursor.fetchall()
                    return [row[0] for row in result]
                
        except sqlite3.Error as e:
            logger.error(f"Database error during filtering: {e}")
        
        return None
    
    def _search_faiss(self, question: str, top_k: int, allowed_uids: Optional[List[str]]) -> List[Dict]:
        """Search using FAISS index."""
        try:
            # Check if we have the embedding model
            if not self.embedding_model:
                logger.warning("Embedding model not available, falling back to keyword search")
                return self._fallback_keyword_search(question, top_k, allowed_uids)
            
            # Encode the question
            question_vector = self.embedding_model.encode([question], convert_to_numpy=True)
            question_vector = question_vector.astype(np.float32)
            
            # Search the FAISS index
            index = self.vector_store['index']
            distances, indices = index.search(question_vector, top_k * 2)  # Get more results for filtering
            
            # Get chunk UID order from metadata
            chunk_uid_order = self.embeddings_meta.get('chunk_uid_order', [])
            if not chunk_uid_order:
                logger.warning("No chunk_uid_order found in embeddings metadata")
                return self._fallback_keyword_search(question, top_k, allowed_uids)
            
            # Map indices to chunk UIDs and filter
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= len(chunk_uid_order):
                    continue
                
                chunk_uid = chunk_uid_order[idx]
                
                # Find the chunk with this UID
                chunk = next((c for c in self.chunks if c.get('uid') == chunk_uid or c.get('chunk_uid') == chunk_uid), None)
                if not chunk:
                    continue
                
                # Apply allowed_uids filter if provided
                if allowed_uids and chunk.get('document_uid') not in allowed_uids:
                    continue
                
                # Add similarity score
                chunk_with_score = chunk.copy()
                chunk_with_score['similarity_score'] = float(1.0 / (1.0 + distance))  # Convert distance to similarity
                results.append(chunk_with_score)
                
                if len(results) >= top_k:
                    break
            
            logger.info(f"FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            logger.info("Falling back to keyword search")
            return self._fallback_keyword_search(question, top_k, allowed_uids)
    
    def _search_chroma(self, question: str, top_k: int, allowed_uids: Optional[List[str]]) -> List[Dict]:
        """Search using Chroma collection."""
        try:
            collection = self.vector_store['collection']
            
            # Build where filter for Chroma
            where_filter = None
            if allowed_uids:
                where_filter = {"document_uid": {"$in": allowed_uids}}
            
            results = collection.query(
                query_texts=[question],
                n_results=top_k,
                where=where_filter
            )
            
            # Convert Chroma results to our format
            chunks = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    chunks.append({
                        'text': doc,
                        'document_uid': metadata.get('document_uid'),
                        'article_number': metadata.get('article_number'),
                        'note_label': metadata.get('note_label'),
                        'score': results['distances'][0][i] if results['distances'] else 0
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error during Chroma search: {e}")
            return []
    
    def _fallback_keyword_search(self, question: str, top_k: int, allowed_uids: Optional[List[str]]) -> List[Dict]:
        """Fallback keyword-based search when vector store is unavailable."""
        logger.info("Using fallback keyword search")
        
        # Convert question to lowercase for case-insensitive matching
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        # Score chunks based on keyword overlap
        scored_chunks = []
        for chunk in self.chunks:
            # Skip invalid chunks
            if not isinstance(chunk, dict):
                continue
                
            # Apply allowed_uids filter if provided
            if allowed_uids and chunk.get('document_uid') not in allowed_uids:
                continue
            
            # Get chunk text
            text = chunk.get('text', '').lower()
            text_words = set(text.split())
            
            # Calculate simple keyword overlap score
            overlap = len(question_words.intersection(text_words))
            if overlap > 0:
                # Normalize by question length
                score = overlap / len(question_words)
                chunk_with_score = chunk.copy()
                chunk_with_score['keyword_score'] = score
                scored_chunks.append(chunk_with_score)
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x['keyword_score'], reverse=True)
        result = scored_chunks[:top_k]
        
        logger.info(f"Keyword search returned {len(result)} results")
        return result
    
    def build_prompt(self, question: str, retrieved_chunks: List[Dict], template_name: str = "default") -> str:
        """Build prompt using template and retrieved chunks."""
        template = self.prompt_templates.get(template_name, self.prompt_templates["default"])
        
        # Concatenate retrieved chunks with consistent citations
        retrieved_text = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get('text', '')
            document_uid = chunk.get('document_uid', 'نامشخص')
            document_title = chunk.get('document_title', chunk.get('title', ''))
            article_number = chunk.get('article_number', '')
            note_label = chunk.get('note_label', '')
            
            # Build consistent citation format: [n] document_title (document_uid) - ماده number note_label
            citation = f"[{i}] "
            if document_title:
                citation += f"{document_title} ({document_uid})"
            else:
                citation += document_uid
            
            if article_number:
                citation += f" - ماده {article_number}"
            if note_label:
                citation += f" - {note_label}"
            
            retrieved_text += f"{citation}:\n{text}\n\n"
        
        # Fill template placeholders
        prompt = template.format(
            question=question,
            retrieved_text=retrieved_text.strip()
        )
        
        return prompt
    
    def generate_answer(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with prompt and return answer with citations."""
        if not self.llm_client:
            return {
                "answer": "خطا: اتصال به مدل زبانی برقرار نشد.",
                "citations": []
            }
        
        try:
            # Call LLM (implementation depends on api_connections.py)
            llm_config = self.config.get('llm', {})
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=llm_config.get('max_tokens', 4000),
                temperature=llm_config.get('temperature', 0.3),
                timeout_s=llm_config.get('timeout_s', 60)
            )
            
            # Extract answer from response (OllamaClient.generate returns a string)
            if isinstance(response, dict):
                answer_text = response.get('text', 'پاسخی دریافت نشد.')
            else:
                answer_text = str(response) if response else 'پاسخی دریافت نشد.'
            
            # Parse citations from answer (simple implementation)
            citations = self._extract_citations_from_answer(answer_text)
            
            # Validate citations
            validated_citations = validate_citations(citations, self.db_path)
            
            return {
                "answer": answer_text,
                "citations": validated_citations
            }
            
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            return {
                "answer": f"خطا در تولید پاسخ: {str(e)}",
                "citations": []
            }
    
    def _extract_citations_from_answer(self, answer: str) -> List[Dict]:
        """Extract citations from generated answer using inline citation markers."""
        citations = []
        
        # Extract inline citation markers like [1] document_title (document_uid) - ماده 12 - note
        # or [2] document_uid - ماده 5
        # Updated pattern to stop at punctuation that ends the citation
        citation_pattern = r'\[(\d+)\]\s*([^\[\n,،.。؟!]+?)(?=[,،.。؟!]|\[|$|\n)'
        matches = re.findall(citation_pattern, answer, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            citation_num, citation_text = match
            citation_text = citation_text.strip()
            
            # Parse the citation text to extract components
            document_uid = 'نامشخص'
            document_title = ''
            article_number = ''
            note_label = ''
            
            # Try to extract document info: title (uid) or just uid
            doc_match = re.search(r'([^\(]+)\(([^\)]+)\)', citation_text)
            if doc_match:
                document_title = doc_match.group(1).strip()
                document_uid = doc_match.group(2).strip()
            else:
                # Look for document_uid at the beginning
                parts = citation_text.split(' - ', 1)
                if parts:
                    document_uid = parts[0].strip()
            
            # Extract article number (support both ASCII and Persian numerals)
            article_match = re.search(r'ماده\s*([\d۰-۹]+)', citation_text)
            if article_match:
                article_number = article_match.group(1)
            
            # Extract note label (everything after the last -)
            note_parts = citation_text.split(' - ')
            if len(note_parts) > 2:  # More than doc - ماده
                potential_note = note_parts[-1].strip()
                if not re.match(r'ماده\s*\d+', potential_note):  # Not an article reference
                    # Extract only the note identifier, not the full descriptive text
                    note_match = re.match(r'(تبصره\s*\d+|بند\s*\d+|ذیل\s*\d+)', potential_note)
                    if note_match:
                        note_label = note_match.group(1)
                    else:
                        # If no specific pattern, take the whole thing but limit length
                        note_label = potential_note[:50]  # Limit to 50 characters
            
            citations.append({
                'document_uid': document_uid,
                'document_title': document_title,
                'article_number': article_number,
                'note_label': note_label if note_label else None
            })
        
        # Also look for standalone legal references
        legal_patterns = [
            (r'ماده\s*(\d+)', 'article'),
            (r'تبصره\s*(\d+)', 'note'),
            (r'بند\s*(\d+)', 'clause')
        ]
        
        for pattern, ref_type in legal_patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                # Only add if not already captured in inline citations
                existing = any(c.get('article_number') == match for c in citations)
                if not existing:
                    citations.append({
                        'document_uid': 'نامشخص',
                        'document_title': '',
                        'article_number': match if ref_type == 'article' else '',
                        'note_label': f'{ref_type} {match}' if ref_type != 'article' else None
                    })
        
        return citations
    
    def answer(self, question: str, top_k: int = 5, template_name: str = "default", 
               filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Orchestrate retrieve → build_prompt → generate_answer pipeline."""
        logger.info(f"Processing question: {question[:50]}...")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question, top_k, filters)
        
        if not retrieved_chunks:
            return {
                "answer": "متأسفانه هیچ سند مرتبطی یافت نشد.",
                "citations": [],
                "retrieved_chunks": 0
            }
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, retrieved_chunks, template_name)
        
        # Step 3: Generate answer
        result = self.generate_answer(prompt)
        result["retrieved_chunks"] = len(retrieved_chunks)
        
        logger.info(f"Generated answer with {len(result['citations'])} citations")
        return result


if __name__ == "__main__":
    # Simple CLI for testing
    parser = argparse.ArgumentParser(description="Test Legal RAG Engine")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--template", default="default", help="Prompt template to use")
    default_config = os.path.join(os.path.dirname(__file__), "Rag_config.json")
    parser.add_argument("--config", default=default_config, help="Config file path")
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG engine
        engine = LegalRAGEngine(config_path=args.config)
        
        # Get answer
        result = engine.answer(
            question=args.question,
            top_k=args.top_k,
            template_name=args.template
        )
        
        # Print results
        print("=" * 50)
        print("سؤال:", args.question)
        print("=" * 50)
        print("پاسخ:")
        print(result["answer"])
        print("\n" + "=" * 50)
        
        if result["citations"]:
            print("منابع:")
            for i, citation in enumerate(result["citations"], 1):
                print(f"{i}. {citation}")
        else:
            print("هیچ منبعی یافت نشد.")
        
        print(f"\nتعداد اسناد بازیابی‌شده: {result.get('retrieved_chunks', 0)}")
        
    except Exception as e:
        print(f"خطا: {e}")
        logger.error(f"CLI error: {e}")