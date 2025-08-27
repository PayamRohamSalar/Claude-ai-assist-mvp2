#!/usr/bin/env python3
"""
Display Quality Evaluation Results
"""

import json
from pathlib import Path


def main():
    # Read embedding report
    embedding_path = Path("embedding_report.json")
    retrieval_path = Path("retrieval_sanity.json")
    
    if not embedding_path.exists():
        print("❌ embedding_report.json not found")
        return
        
    if not retrieval_path.exists():
        print("❌ retrieval_sanity.json not found")
        return
    
    with open(embedding_path, 'r', encoding='utf-8') as f:
        embedding_report = json.load(f)
    
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        retrieval_report = json.load(f)
    
    print("="*80)
    print("RAG SYSTEM QUALITY EVALUATION RESULTS")
    print("="*80)
    
    # Embedding Analysis Results
    print("\nEMBEDDING ANALYSIS:")
    print(f"   - Vector count: {embedding_report['vector_count']:,}")
    print(f"   - Vector dimensions: {embedding_report['vector_dimension']}")
    print(f"   - L2 norm mean: {embedding_report['l2_norms']['mean']:.4f}")
    print(f"   - L2 norm variance: {embedding_report['l2_norms']['variance']:.4f}")
    print(f"   - Zero vectors: {embedding_report['embedding_stats']['zero_vectors']}")
    print(f"   - Model used: {embedding_report['embedding_stats']['model_name']}")
    
    # Duplicate Analysis
    duplicates = embedding_report['duplicate_candidates']
    duplicate_percentage = (len(duplicates) / embedding_report['vector_count']) * 100
    print(f"\nDUPLICATE DETECTION:")
    print(f"   - High similarity pairs (cosine > 0.98): {len(duplicates)}")
    print(f"   - Duplicate percentage: {duplicate_percentage:.2f}%")
    
    if duplicates:
        print("   - Top duplicate pairs:")
        for i, dup in enumerate(duplicates[:3], 1):
            print(f"     {i}. Similarity: {dup['cosine_similarity']:.4f}")
    
    # Chunk Length Analysis
    length_dist = embedding_report['chunk_length_distribution']
    print(f"\nCHUNK LENGTH DISTRIBUTION:")
    print(f"   - Mean length: {length_dist['mean_length']:.0f} chars")
    print(f"   - Median length: {length_dist['median_length']:.0f} chars")
    print(f"   - Min length: {length_dist['min_length']} chars")
    print(f"   - Max length: {length_dist['max_length']:,} chars")
    print(f"   - Empty chunks: {length_dist['empty_chunks']}")
    
    # Retrieval Testing Results
    print(f"\nRETRIEVAL SANITY TESTS:")
    backend_info = retrieval_report['backend_info']
    test_queries = retrieval_report['test_queries']
    
    print(f"   - Backend: {backend_info['type'].upper()}")
    print(f"   - Total queries tested: {len(test_queries)}")
    
    if test_queries:
        successful_queries = 0
        for query in test_queries:
            analysis = query.get('analysis', {})
            if analysis.get('has_relevant_results', False):
                successful_queries += 1
        
        success_rate = (successful_queries / len(test_queries)) * 100
        print(f"   - Success rate: {success_rate:.1f}% ({successful_queries}/{len(test_queries)})")
        
        # Show sample query results
        print("\n   Sample Query Results:")
        for i, query in enumerate(test_queries[:3], 1):
            q_text = query['query']
            results = query['top_results']
            analysis = query.get('analysis', {})
            
            print(f"     {i}. Query: '{q_text}'")
            print(f"        Results: {len(results)} chunks retrieved")
            if results:
                print(f"        Top score: {results[0]['similarity_score']:.3f}")
                print(f"        Avg score: {analysis.get('avg_similarity', 0):.3f}")
            print(f"        Relevant: {'YES' if analysis.get('has_relevant_results') else 'NO'}")
    
    # Overall Assessment
    print(f"\nOVERALL ASSESSMENT:")
    
    # Embedding quality
    if duplicate_percentage < 5:
        embedding_quality = "[GOOD]"
    elif duplicate_percentage < 10:
        embedding_quality = "[ACCEPTABLE]"
    else:
        embedding_quality = "[NEEDS IMPROVEMENT]"
    
    print(f"   - Embedding quality: {embedding_quality}")
    
    # Retrieval quality
    if test_queries:
        if success_rate >= 70:
            retrieval_quality = "[GOOD]"
        elif success_rate >= 50:
            retrieval_quality = "[ACCEPTABLE]" 
        else:
            retrieval_quality = "[NEEDS IMPROVEMENT]"
        print(f"   - Retrieval quality: {retrieval_quality}")
    else:
        print(f"   - Retrieval quality: [COULD NOT TEST]")
    
    print(f"\nAnalysis completed: {embedding_report['analysis_metadata']['analysis_time_utc']}")
    print("="*80)


if __name__ == "__main__":
    main()