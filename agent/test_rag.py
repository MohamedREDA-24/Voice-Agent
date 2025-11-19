"""
Test script for RAG system
Run this to verify the RAG system is working correctly.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RAGSystem


def test_rag_system():
    """Test the RAG system end-to-end"""
    print("=" * 70)
    print("🧪 RAG System Test")
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
            rag.build_index(force_rebuild=False)
        else:
            print(f"   ✅ Using existing index ({rag.index.ntotal} vectors)")
            # Optionally rebuild
            rebuild = input("   🔄 Rebuild index? (y/N): ").strip().lower()
            if rebuild == 'y':
                rag.build_index(force_rebuild=True)
    except Exception as e:
        print(f"   ❌ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if rag.index is None or rag.index.ntotal == 0:
        print("   ❌ Index is empty")
        return False
    
    # Test queries
    print("\n4️⃣  Testing retrieval...")
    test_queries = [
        "What is the return policy?",
        "How do I contact support?",
        "What are the shipping options?",
        "Do you offer warranties?",
        "How do I track my order?",
    ]
    
    all_passed = True
    for query in test_queries:
        print(f"\n   📝 Query: \"{query}\"")
        try:
            results = rag.retrieve(query, top_k=3)
            
            if results:
                print(f"   ✅ Found {len(results)} result(s)")
                for i, result in enumerate(results[:2], 1):  # Show top 2
                    score = result['score']
                    content_preview = result['content'][:150].replace('\n', ' ')
                    source_name = Path(result['source']).name
                    print(f"      {i}. [{source_name}] (distance: {score:.2f})")
                    print(f"         {content_preview}...")
            else:
                print("   ⚠️  No results found")
                all_passed = False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    # Test context formatting
    print("\n5️⃣  Testing context formatting...")
    try:
        query = "What is your return policy?"
        context = rag.retrieve_context(query, top_k=2)
        if context:
            print("   ✅ Context retrieved:")
            print(f"   {context[:300]}...")
        else:
            print("   ⚠️  No context retrieved")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ RAG System Test: PASSED")
        print("\n💡 The RAG system is ready to use!")
        print("   You can now integrate it with the voice agent.")
    else:
        print("⚠️  RAG System Test: Some issues found")
        print("   Please review the errors above.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)

