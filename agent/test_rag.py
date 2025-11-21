"""
Test script for RAG system with accuracy measurement
Run this to verify the RAG system and measure retrieval accuracy.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RAGSystem


# Ground truth: query -> expected keywords/topics that should appear in results
GROUND_TRUTH = {
    # General Information
    "What is your return policy?": ["return", "30-day", "policy", "refund"],
    "What are your shipping options?": ["shipping", "delivery", "standard", "express"],
    "How do I track my order?": ["track", "order", "tracking", "shipment"],
    
    # Customer Service
    "How do I contact support?": ["support", "contact", "email", "phone", "customer service"],
    "What are your customer service hours?": ["hours", "customer service", "available", "support"],
    "Can I modify or cancel my order?": ["modify", "cancel", "order", "change"],
    
    # Products
    "Do you offer product warranties?": ["warranty", "warranties", "product", "guarantee"],
    "Are your products authentic?": ["authentic", "genuine", "original", "product"],
    "Do you offer product recommendations?": ["recommendation", "suggest", "product"],
    
    # Account & Orders
    "How do I create an account?": ["account", "create", "sign up", "register"],
    "How do I reset my password?": ["password", "reset", "forgot", "change"],
    "Can I save items for later?": ["save", "wishlist", "later", "favorite"],
    
    # Payment & Billing
    "What payment methods do you accept?": ["payment", "method", "credit card", "accept"],
    "Is my payment information secure?": ["secure", "payment", "security", "safe", "encryption"],
    "Do you offer payment plans?": ["payment plan", "installment", "klarna", "afterpay"],
    
    # Technical Support
    "Do you provide technical support?": ["technical", "support", "help", "assistance"],
    "How do I access product manuals?": ["manual", "documentation", "guide", "instructions"],
}


def calculate_relevance_score(query: str, result: Dict, ground_truth_keywords: List[str]) -> float:
    """
    Calculate relevance score based on keyword matching
    
    Args:
        query: The search query
        result: Retrieved result dictionary
        ground_truth_keywords: Expected keywords for this query
        
    Returns:
        Relevance score between 0 and 1
    """
    content = result['content'].lower()
    
    # Count how many ground truth keywords appear in the result
    matches = sum(1 for keyword in ground_truth_keywords if keyword.lower() in content)
    
    # Calculate score
    if len(ground_truth_keywords) == 0:
        return 1.0
    
    return matches / len(ground_truth_keywords)


def evaluate_retrieval(query: str, results: List[Dict], ground_truth_keywords: List[str]) -> Dict:
    """
    Evaluate retrieval results for a single query
    
    Returns:
        Dictionary with precision, relevance scores, and metrics
    """
    if not results:
        return {
            'precision_at_1': 0.0,
            'precision_at_3': 0.0,
            'avg_relevance': 0.0,
            'found_keywords': [],
            'missing_keywords': ground_truth_keywords,
            'top_score': 0.0
        }
    
    # Calculate relevance for each result
    relevance_scores = [
        calculate_relevance_score(query, result, ground_truth_keywords)
        for result in results
    ]
    
    # Precision @ k (consider relevant if relevance > 0.3)
    relevant_threshold = 0.3
    precision_at_1 = 1.0 if relevance_scores[0] > relevant_threshold else 0.0
    precision_at_3 = sum(1 for score in relevance_scores[:3] if score > relevant_threshold) / min(3, len(relevance_scores))
    
    # Find which keywords were found
    all_content = ' '.join([r['content'].lower() for r in results[:3]])
    found_keywords = [kw for kw in ground_truth_keywords if kw.lower() in all_content]
    missing_keywords = [kw for kw in ground_truth_keywords if kw.lower() not in all_content]
    
    return {
        'precision_at_1': precision_at_1,
        'precision_at_3': precision_at_3,
        'avg_relevance': sum(relevance_scores[:3]) / min(3, len(relevance_scores)),
        'found_keywords': found_keywords,
        'missing_keywords': missing_keywords,
        'top_score': results[0]['score'] if results else 0.0,
        'relevance_scores': relevance_scores[:3]
    }


def test_rag_system():
    """Test the RAG system end-to-end with accuracy metrics"""
    print("=" * 70)
    print("🧪 RAG System Test with Accuracy Measurement")
    print("=" * 70)
    
    # Initialize RAG
    print("\n1️⃣  Initializing RAG system...")
    try:
        rag = RAGSystem()
        print("   ✅ RAG system initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Check if documents exist
    print("\n2️⃣  Checking for documents...")
    docs_dir = rag.documents_dir
    if not docs_dir.exists():
        print(f"   ⚠️  Documents directory not found: {docs_dir}")
        print("   💡 Create the directory and add some documents")
        return False
    
    text_files = list(docs_dir.glob("*.txt"))
    pdf_files = list(docs_dir.glob("*.pdf"))
    total_files = len(text_files) + len(pdf_files)
    
    if total_files == 0:
        print(f"   ⚠️  No documents found in {docs_dir}")
        print("   💡 Add some .txt or .pdf files to the documents directory")
        return False
    
    print(f"   ✅ Found {total_files} document(s): {len(text_files)} text, {len(pdf_files)} PDF")
    
    # Build or load index
    print("\n3️⃣  Building/loading index...")
    try:
        if rag.index is None:
            print("   📦 Building new index...")
            rag.build_index(force_rebuild=True)
        else:
            print(f"   ✅ Using existing index ({rag.index.ntotal} vectors)")
    except Exception as e:
        print(f"   ❌ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if rag.index is None or rag.index.ntotal == 0:
        print("   ❌ Index is empty")
        return False
    
    # Test queries with accuracy measurement
    print("\n4️⃣  Testing retrieval with accuracy metrics...")
    print("-" * 70)
    
    all_metrics = []
    all_passed = True
    
    for query, ground_truth_keywords in GROUND_TRUTH.items():
        print(f"\n📝 Query: \"{query}\"")
        print(f"   Expected keywords: {', '.join(ground_truth_keywords)}")
        
        try:
            results = rag.retrieve(query, top_k=3)
            
            if results:
                # Evaluate results
                metrics = evaluate_retrieval(query, results, ground_truth_keywords)
                all_metrics.append(metrics)
                
                # Display results
                print(f"   ✅ Found {len(results)} result(s)")
                print(f"   📊 Precision@1: {metrics['precision_at_1']:.2f} | Precision@3: {metrics['precision_at_3']:.2f} | Avg Relevance: {metrics['avg_relevance']:.2f}")
                
                if metrics['found_keywords']:
                    print(f"   ✓ Found keywords: {', '.join(metrics['found_keywords'])}")
                if metrics['missing_keywords']:
                    print(f"   ✗ Missing keywords: {', '.join(metrics['missing_keywords'])}")
                
                # Show top result
                top_result = results[0]
                content_preview = top_result['content'][:120].replace('\n', ' ')
                source_name = Path(top_result['source']).name
                print(f"   🥇 Top: [{source_name}] (sim: {top_result['score']:.2f}, rel: {metrics['relevance_scores'][0]:.2f})")
                print(f"      {content_preview}...")
                
            else:
                print("   ❌ No results found")
                all_passed = False
                all_metrics.append({
                    'precision_at_1': 0.0,
                    'precision_at_3': 0.0,
                    'avg_relevance': 0.0,
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    # Calculate overall metrics
    print("\n" + "=" * 70)
    print("📊 OVERALL ACCURACY METRICS")
    print("=" * 70)
    
    if all_metrics:
        avg_precision_at_1 = sum(m['precision_at_1'] for m in all_metrics) / len(all_metrics)
        avg_precision_at_3 = sum(m['precision_at_3'] for m in all_metrics) / len(all_metrics)
        avg_relevance = sum(m['avg_relevance'] for m in all_metrics) / len(all_metrics)
        
        print(f"\n📈 Average Precision@1:  {avg_precision_at_1:.2%}")
        print(f"📈 Average Precision@3:  {avg_precision_at_3:.2%}")
        print(f"📈 Average Relevance:    {avg_relevance:.2%}")
        print(f"📈 Total Queries:        {len(all_metrics)}")
        
        # Performance rating
        print("\n🎯 Performance Rating:")
        if avg_precision_at_1 >= 0.8:
            print("   ⭐⭐⭐⭐⭐ Excellent (≥80%)")
        elif avg_precision_at_1 >= 0.6:
            print("   ⭐⭐⭐⭐ Good (60-80%)")
        elif avg_precision_at_1 >= 0.4:
            print("   ⭐⭐⭐ Fair (40-60%)")
        elif avg_precision_at_1 >= 0.2:
            print("   ⭐⭐ Poor (20-40%)")
        else:
            print("   ⭐ Very Poor (<20%)")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if avg_precision_at_1 < 0.6:
            print("   • Consider adding more relevant documents")
            print("   • Try adjusting chunk_size and chunk_overlap")
            print("   • Verify document quality and coverage")
        if avg_relevance < 0.5:
            print("   • Ground truth keywords may not match document content")
            print("   • Consider using a different embedding model")
    
    print("\n" + "=" * 70)
    if all_passed and avg_precision_at_1 >= 0.6:
        print("✅ RAG System Test: PASSED")
        print("\n💡 The RAG system is ready to use!")
    else:
        print("⚠️  RAG System Test: Issues found or low accuracy")
        print("   Please review the metrics above.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)