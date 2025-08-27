#!/usr/bin/env python3
"""
ارزیابی کیفیت سیستم RAG
RAG Quality Evaluation System

این ماژول ارزیابی کیفیت بردارها و عملکرد جستجو را انجام می‌دهد
This module performs embedding quality assessment and retrieval performance evaluation
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from collections import defaultdict, Counter

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

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_eval.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QualityEvaluator:
    """
    کلاس ارزیابی کیفیت سیستم RAG
    RAG System Quality Evaluator Class
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        راه‌اندازی ارزیاب کیفیت
        Initialize quality evaluator
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.rag_config = config.get('rag', {})
        
        # Set random seed for deterministic sampling
        self.random_seed = config.get('random_seed', 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Backend configuration
        self.backend = self.rag_config.get('index_backend', 'faiss').lower()
        
        # Model configuration
        self.model_name = self.rag_config.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        
        # Paths
        self.embeddings_path = Path("data/processed_phase_3/embeddings.npy")
        self.meta_path = Path("data/processed_phase_3/embeddings_meta.json")
        self.chunks_path = Path("data/processed_phase_3/chunks.json")
        self.vector_db_dir = Path("data/processed_phase_3/vector_db")
        
        # Output paths
        self.output_dir = Path(".")
        self.embedding_report_path = self.output_dir / "embedding_report.json"
        self.retrieval_report_path = self.output_dir / "retrieval_sanity.json"
        
        # Data storage
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.chunks: Optional[List[Dict[str, Any]]] = None
        self.model: Optional[SentenceTransformer] = None
        
        # Persian legal keywords for sanity testing
        self.persian_keywords = [
            "هیئت علمی",
            "تبصره", 
            "آیین‌نامه",
            "بودجه",
            "قانون",
            "ماده",
            "فصل",
            "بخش",
            "کمیسیون",
            "شورا",
            "مجلس",
            "وزارت",
            "اداره",
            "سازمان",
            "دانشگاه"
        ]
    
    def _load_data(self) -> None:
        """
        بارگذاری داده‌های مورد نیاز
        Load required data
        """
        logger.info("در حال بارگذاری داده‌های مورد نیاز...")
        logger.info("Loading required data...")
        
        # Load embeddings
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"فایل بردارها یافت نشد: {self.embeddings_path}")
        
        self.embeddings = np.load(self.embeddings_path)
        logger.info(f"بردارها بارگذاری شد: شکل {self.embeddings.shape}")
        
        # Load metadata
        if not self.meta_path.exists():
            raise FileNotFoundError(f"فایل متادیتای بردارها یافت نشد: {self.meta_path}")
        
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load chunks
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"فایل قطعات یافت نشد: {self.chunks_path}")
        
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        if isinstance(chunks_data, dict):
            self.chunks = chunks_data.get('chunks', [])
        elif isinstance(chunks_data, list):
            self.chunks = chunks_data
        else:
            raise ValueError("فرمت نامعتبر برای فایل chunks.json")
        
        logger.info(f"قطعات بارگذاری شد: {len(self.chunks)} قطعه")
    
    def _load_model(self) -> SentenceTransformer:
        """
        بارگذاری مدل embedding برای تست‌های جستجو
        Load embedding model for retrieval tests
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("کتابخانه sentence-transformers در دسترس نیست")
        
        try:
            logger.info(f"در حال بارگذاری مدل: {self.model_name}")
            # Try to load from local cache first
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            model = SentenceTransformer(self.model_name, local_files_only=True)
            logger.info("مدل با موفقیت از کش محلی بارگذاری شد")
            return model
        except Exception as e:
            logger.warning(f"خطا در بارگذاری مدل از کش محلی: {str(e)}")
            try:
                # Try without offline mode
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                model = SentenceTransformer(self.model_name)
                logger.info("مدل با موفقیت از اینترنت بارگذاری شد")
                return model
            except Exception as e2:
                logger.error(f"خطا در بارگذاری مدل: {str(e2)}")
                logger.warning("ادامه بدون تست‌های جستجو...")
                return None
    
    def analyze_embeddings(self) -> Dict[str, Any]:
        """
        تجزیه و تحلیل کیفیت بردارها
        Analyze embedding quality
        """
        logger.info("شروع تجزیه و تحلیل کیفیت بردارها...")
        logger.info("Starting embedding quality analysis...")
        
        report = {}
        
        # Basic dimensions and count
        report['vector_dimension'] = int(self.embeddings.shape[1])
        report['vector_count'] = int(self.embeddings.shape[0])
        
        # Calculate L2 norms
        l2_norms = np.linalg.norm(self.embeddings, axis=1)
        report['l2_norms'] = {
            'mean': float(np.mean(l2_norms)),
            'variance': float(np.var(l2_norms)),
            'std': float(np.std(l2_norms)),
            'min': float(np.min(l2_norms)),
            'max': float(np.max(l2_norms))
        }
        
        logger.info(f"L2 نرم‌ها - میانگین: {report['l2_norms']['mean']:.4f}, واریانس: {report['l2_norms']['variance']:.4f}")
        
        # Find potential duplicates using cosine similarity
        logger.info("جستجوی قطعات مشابه با کسینوس > 0.98...")
        duplicates = self._find_duplicate_candidates(threshold=0.98)
        report['duplicate_candidates'] = duplicates
        
        # Analyze chunk lengths
        logger.info("تجزیه و تحلیل توزیع طول قطعات...")
        length_distribution = self._analyze_chunk_lengths()
        report['chunk_length_distribution'] = length_distribution
        
        # Additional quality metrics
        report['embedding_stats'] = {
            'mean_magnitude': float(np.mean(l2_norms)),
            'embedding_density': float(np.mean(np.abs(self.embeddings))),
            'zero_vectors': int(np.sum(l2_norms < 1e-8)),
            'model_name': self.metadata.get('model_name', 'unknown')
        }
        
        report['analysis_metadata'] = {
            'analysis_time_utc': datetime.now(timezone.utc).isoformat(),
            'random_seed': self.random_seed,
            'chunks_md5': self.metadata.get('chunks_md5', ''),
            'embedding_creation_time': self.metadata.get('created_utc', '')
        }
        
        logger.info("تجزیه و تحلیل بردارها کامل شد")
        logger.info("Embedding analysis completed")
        
        return report
    
    def _find_duplicate_candidates(self, threshold: float = 0.98) -> List[Dict[str, Any]]:
        """
        یافتن نامزدهای مشابه با کسینوس بالا
        Find duplicate candidates with high cosine similarity
        """
        duplicates = []
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity matrix (sample-based for efficiency)
        sample_size = min(500, len(self.embeddings))  # Sample for performance
        sample_indices = random.sample(range(len(self.embeddings)), sample_size)
        
        logger.info(f"بررسی {sample_size} نمونه برای یافتن قطعات مشابه...")
        
        for i, idx_i in enumerate(sample_indices):
            if i % 100 == 0:
                logger.info(f"پردازش نمونه {i}/{sample_size}")
            
            # Compare with all other embeddings
            similarities = np.dot(normalized_embeddings[idx_i], normalized_embeddings.T)
            high_sim_indices = np.where(similarities > threshold)[0]
            
            for idx_j in high_sim_indices:
                if idx_i != idx_j:  # Skip self-similarity
                    chunk_uid_i = self.metadata['chunk_uid_order'][idx_i] if idx_i < len(self.metadata['chunk_uid_order']) else f"idx_{idx_i}"
                    chunk_uid_j = self.metadata['chunk_uid_order'][idx_j] if idx_j < len(self.metadata['chunk_uid_order']) else f"idx_{idx_j}"
                    
                    duplicates.append({
                        'chunk_uid_1': chunk_uid_i,
                        'chunk_uid_2': chunk_uid_j,
                        'cosine_similarity': float(similarities[idx_j]),
                        'embedding_indices': [int(idx_i), int(idx_j)]
                    })
        
        # Remove duplicates and sort by similarity
        seen = set()
        unique_duplicates = []
        for dup in duplicates:
            # Create a unique key
            key = tuple(sorted([dup['chunk_uid_1'], dup['chunk_uid_2']]))
            if key not in seen:
                seen.add(key)
                unique_duplicates.append(dup)
        
        unique_duplicates.sort(key=lambda x: x['cosine_similarity'], reverse=True)
        
        logger.info(f"یافت شد {len(unique_duplicates)} جفت مشابه با کسینوس > {threshold}")
        
        return unique_duplicates[:20]  # Return top 20
    
    def _analyze_chunk_lengths(self) -> Dict[str, Any]:
        """
        تجزیه و تحلیل توزیع طول قطعات
        Analyze chunk length distribution
        """
        lengths = []
        
        for chunk in self.chunks:
            content = chunk.get('content', chunk.get('text', chunk.get('normalized_text', '')))
            char_count = len(content.strip())
            lengths.append(char_count)
        
        lengths = np.array(lengths)
        
        return {
            'mean_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'std_length': float(np.std(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'percentiles': {
                '25th': float(np.percentile(lengths, 25)),
                '50th': float(np.percentile(lengths, 50)),
                '75th': float(np.percentile(lengths, 75)),
                '90th': float(np.percentile(lengths, 90)),
                '95th': float(np.percentile(lengths, 95))
            },
            'total_chunks': len(lengths),
            'empty_chunks': int(np.sum(lengths == 0))
        }
    
    def perform_retrieval_sanity_test(self) -> Dict[str, Any]:
        """
        انجام تست‌های عقلی جستجو
        Perform retrieval sanity tests
        """
        logger.info("شروع تست‌های عقلی جستجو...")
        logger.info("Starting retrieval sanity tests...")
        
        # Load model for query encoding
        if not self.model:
            self.model = self._load_model()
        
        # If model loading failed, return a placeholder report
        if not self.model:
            logger.warning("مدل بارگذاری نشد، تست‌های جستجو قابل انجام نیست")
            logger.warning("Model not loaded, retrieval tests cannot be performed")
            return {
                'test_queries': [],
                'backend_info': {
                    'type': self.backend,
                    'model_name': self.model_name,
                    'total_vectors': self.embeddings.shape[0] if self.embeddings is not None else 0,
                    'status': 'model_loading_failed'
                },
                'test_metadata': {
                    'test_time_utc': datetime.now(timezone.utc).isoformat(),
                    'random_seed': self.random_seed,
                    'keywords_tested': 0,
                    'error': 'Model loading failed - network or cache issues'
                }
            }
        
        # Prepare backend for search
        try:
            search_backend = self._prepare_search_backend()
        except Exception as e:
            logger.error(f"خطا در آماده‌سازی پشتیبان جستجو: {str(e)}")
            return {
                'test_queries': [],
                'backend_info': {
                    'type': self.backend,
                    'model_name': self.model_name,
                    'total_vectors': self.embeddings.shape[0] if self.embeddings is not None else 0,
                    'status': 'backend_preparation_failed'
                },
                'test_metadata': {
                    'test_time_utc': datetime.now(timezone.utc).isoformat(),
                    'random_seed': self.random_seed,
                    'keywords_tested': 0,
                    'error': f'Backend preparation failed: {str(e)}'
                }
            }
        
        results = {
            'test_queries': [],
            'backend_info': {
                'type': self.backend,
                'model_name': self.model_name,
                'total_vectors': self.embeddings.shape[0]
            },
            'test_metadata': {
                'test_time_utc': datetime.now(timezone.utc).isoformat(),
                'random_seed': self.random_seed,
                'keywords_tested': len(self.persian_keywords)
            }
        }
        
        # Test each keyword
        for i, keyword in enumerate(self.persian_keywords[:10]):  # Test first 10 keywords
            logger.info(f"تست کلیدواژه: {keyword} ({i+1}/{min(10, len(self.persian_keywords))})")
            
            try:
                # Encode query
                query_embedding = self.model.encode([keyword], convert_to_numpy=True)[0]
                
                # Perform search
                top_results = self._search_with_backend(search_backend, query_embedding, k=5)
                
                # Analyze results
                query_result = {
                    'query': keyword,
                    'top_results': top_results,
                    'analysis': self._analyze_retrieval_results(keyword, top_results)
                }
                
                results['test_queries'].append(query_result)
                
            except Exception as e:
                logger.error(f"خطا در تست کلیدواژه '{keyword}': {str(e)}")
                results['test_queries'].append({
                    'query': keyword,
                    'error': str(e),
                    'top_results': [],
                    'analysis': {'error': True}
                })
        
        logger.info("تست‌های عقلی جستجو کامل شد")
        logger.info("Retrieval sanity tests completed")
        
        return results
    
    def _prepare_search_backend(self):
        """
        آماده‌سازی پشتیبان جستجو
        Prepare search backend
        """
        if self.backend == 'faiss':
            return self._prepare_faiss_backend()
        elif self.backend == 'chroma':
            return self._prepare_chroma_backend()
        else:
            raise ValueError(f"پشتیبان پشتیبانی نشده: {self.backend}")
    
    def _prepare_faiss_backend(self):
        """آماده‌سازی FAISS"""
        if not HAS_FAISS:
            raise ImportError("کتابخانه faiss در دسترس نیست")
        
        faiss_dir = self.vector_db_dir / "faiss"
        index_path = faiss_dir / "faiss.index"
        mapping_path = faiss_dir / "mapping.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"فایل فهرست FAISS یافت نشد: {index_path}")
        
        # Load index
        index = faiss.read_index(str(index_path))
        
        # Load mapping
        mapping = {}
        if mapping_path.exists():
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        
        return {'index': index, 'mapping': mapping, 'type': 'faiss'}
    
    def _prepare_chroma_backend(self):
        """آماده‌سازی Chroma"""
        if not HAS_CHROMA:
            raise ImportError("کتابخانه chromadb در دسترس نیست")
        
        chroma_dir = self.vector_db_dir / "chroma"
        
        if not chroma_dir.exists():
            raise FileNotFoundError(f"پوشه Chroma یافت نشد: {chroma_dir}")
        
        # Create client
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )
        
        # Get collection
        collection = client.get_collection("legal_chunks")
        
        return {'client': client, 'collection': collection, 'type': 'chroma'}
    
    def _search_with_backend(self, backend, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        جستجو با پشتیبان
        Search using backend
        """
        if backend['type'] == 'faiss':
            return self._search_faiss(backend, query_embedding, k)
        elif backend['type'] == 'chroma':
            return self._search_chroma(backend, query_embedding, k)
        else:
            raise ValueError(f"نوع پشتیبان نامعتبر: {backend['type']}")
    
    def _search_faiss(self, backend, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """جستجو در FAISS"""
        index = backend['index']
        mapping = backend['mapping']
        
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            chunk_uid = mapping.get(str(idx), f"idx_{idx}")
            
            # Find chunk data
            chunk_data = None
            for chunk in self.chunks:
                if chunk.get('uid', chunk.get('chunk_uid', '')) == chunk_uid:
                    chunk_data = chunk
                    break
            
            result = {
                'rank': i + 1,
                'chunk_uid': chunk_uid,
                'similarity_score': float(score),
                'chunk_data': chunk_data
            }
            results.append(result)
        
        return results
    
    def _search_chroma(self, backend, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """جستجو در Chroma"""
        collection = backend['collection']
        
        # Search
        search_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        results = []
        if search_results['ids'] and search_results['ids'][0]:
            for i, (chunk_id, document, metadata, distance) in enumerate(zip(
                search_results['ids'][0],
                search_results['documents'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            )):
                result = {
                    'rank': i + 1,
                    'chunk_uid': chunk_id,
                    'similarity_score': 1.0 - distance,  # Convert distance to similarity
                    'document_content': document,
                    'metadata': metadata
                }
                results.append(result)
        
        return results
    
    def _analyze_retrieval_results(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        تجزیه و تحلیل نتایج جستجو
        Analyze retrieval results
        """
        analysis = {
            'query_found_in_results': 0,
            'avg_similarity': 0.0,
            'result_count': len(results),
            'has_relevant_results': False
        }
        
        if not results:
            return analysis
        
        # Calculate average similarity
        similarities = [r.get('similarity_score', 0) for r in results]
        analysis['avg_similarity'] = float(np.mean(similarities)) if similarities else 0.0
        
        # Check if query appears in results
        query_lower = query.lower()
        for result in results:
            content = ""
            if 'chunk_data' in result and result['chunk_data']:
                content = result['chunk_data'].get('content', 
                    result['chunk_data'].get('text', 
                        result['chunk_data'].get('normalized_text', '')
                    )
                )
            elif 'document_content' in result:
                content = result['document_content']
            
            if query_lower in content.lower():
                analysis['query_found_in_results'] += 1
        
        # Determine if results are relevant (similarity > 0.3 or query found)
        analysis['has_relevant_results'] = (
            analysis['avg_similarity'] > 0.3 or 
            analysis['query_found_in_results'] > 0
        )
        
        return analysis
    
    def generate_persian_summary(self, embedding_report: Dict[str, Any], retrieval_report: Dict[str, Any]) -> str:
        """
        تولید خلاصه فارسی
        Generate Persian summary
        """
        # Embedding analysis summary
        duplicate_count = len(embedding_report.get('duplicate_candidates', []))
        total_vectors = embedding_report.get('vector_count', 0)
        duplicate_percentage = (duplicate_count / total_vectors * 100) if total_vectors > 0 else 0
        
        # Retrieval analysis summary
        successful_queries = 0
        total_queries = len(retrieval_report.get('test_queries', []))
        retrieval_status = retrieval_report.get('backend_info', {}).get('status', 'unknown')
        
        if retrieval_status == 'model_loading_failed':
            success_rate = -1  # Indicate test failed
        elif total_queries > 0:
            for query_result in retrieval_report.get('test_queries', []):
                if query_result.get('analysis', {}).get('has_relevant_results', False):
                    successful_queries += 1
            success_rate = (successful_queries / total_queries * 100)
        else:
            success_rate = 0
        
        summary = f"""نتایج سنجش کیفیت سیستم RAG:

📊 آمار بردارها:
- تعداد کل بردارها: {total_vectors:,}
- ابعاد بردار: {embedding_report.get('vector_dimension', 0)}
- میانگین نرم L2: {embedding_report.get('l2_norms', {}).get('mean', 0):.4f}

🔍 تشخیص تکراری:
- {duplicate_percentage:.1f}٪ قطعات مشابه بسیار بالا شناسایی شد
- {duplicate_count} جفت مشابه با کسینوس > ۰.۹۸

📏 توزیع طول قطعات:
- میانگین طول: {embedding_report.get('chunk_length_distribution', {}).get('mean_length', 0):.0f} کاراکتر
- حداکثر طول: {embedding_report.get('chunk_length_distribution', {}).get('max_length', 0):,} کاراکتر

🔎 عملکرد جستجو:
{f'- نرخ موفقیت: {success_rate:.1f}٪ ({successful_queries}/{total_queries} پرس‌وجو)' if success_rate >= 0 else '- ⚠️ تست‌های جستجو قابل انجام نبود (مشکل شبکه/مدل)'}
- پشتیبان مورد استفاده: {retrieval_report.get('backend_info', {}).get('type', 'نامشخص')}

⚠️ نتیجه‌گیری:
{
    '✅ کیفیت بردارها قابل قبول' if duplicate_percentage < 5 else '⚠️ تعداد زیاد قطعات مشابه'
}{
    ' | جستجو نیاز به بررسی' if success_rate == -1 else 
    ' | جستجو مناسب' if success_rate > 60 else 
    ' | جستجو نیاز به بهبود' if success_rate >= 0 else ''
}

📅 زمان ارزیابی: {datetime.now().strftime('%Y/%m/%d - %H:%M')}
🔢 بذر تصادفی: {self.random_seed}
"""
        
        return summary
    
    def run_quality_evaluation(self) -> None:
        """
        اجرای کامل ارزیابی کیفیت
        Run complete quality evaluation
        """
        try:
            logger.info("شروع ارزیابی کیفیت سیستم RAG...")
            logger.info("Starting RAG system quality evaluation...")
            
            # Load data
            self._load_data()
            
            # Analyze embeddings
            logger.info("مرحله ۱: تجزیه و تحلیل کیفیت بردارها")
            embedding_report = self.analyze_embeddings()
            
            # Save embedding report
            with open(self.embedding_report_path, 'w', encoding='utf-8') as f:
                json.dump(embedding_report, f, indent=2, ensure_ascii=False)
            logger.info(f"گزارش بردارها ذخیره شد: {self.embedding_report_path}")
            
            # Perform retrieval tests
            logger.info("مرحله ۲: تست‌های عقلی جستجو")
            retrieval_report = self.perform_retrieval_sanity_test()
            
            # Save retrieval report
            with open(self.retrieval_report_path, 'w', encoding='utf-8') as f:
                json.dump(retrieval_report, f, indent=2, ensure_ascii=False)
            logger.info(f"گزارش جستجو ذخیره شد: {self.retrieval_report_path}")
            
            # Generate Persian summary
            logger.info("مرحله ۳: تولید خلاصه فارسی")
            summary = self.generate_persian_summary(embedding_report, retrieval_report)
            
            # Print summary
            print("\n" + "="*60)
            try:
                print(summary)
            except UnicodeEncodeError:
                # Fallback for console encoding issues
                print(summary.encode('utf-8', errors='replace').decode('utf-8'))
            print("="*60)
            
            logger.info("ارزیابی کیفیت با موفقیت کامل شد!")
            logger.info("Quality evaluation completed successfully!")
            
        except Exception as e:
            logger.error(f"خطا در ارزیابی کیفیت: {str(e)}")
            logger.error(f"Error in quality evaluation: {str(e)}")
            raise
    
    def run(self) -> None:
        """
        اجرای کامل فرآیند ارزیابی
        Run the complete evaluation process
        """
        self.run_quality_evaluation()


def load_config() -> Dict[str, Any]:
    """
    بارگذاری تنظیمات از فایل کانفیگ
    Load configuration from config file
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
    
    return {
        "rag": {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "index_backend": "faiss"
        },
        "random_seed": 42
    }


def main():
    """
    تابع اصلی برای اجرای مستقل
    Main function for standalone execution
    """
    try:
        config = load_config()
        evaluator = QualityEvaluator(config)
        evaluator.run()
        
    except Exception as e:
        logger.error(f"خطای کلی: {str(e)}")
        logger.error(f"General error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())