"""
RAG (Retrieval-Augmented Generation) System
Handles document loading, embedding generation, and vector-based retrieval.
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
try:
    # Try new LangChain import paths (v0.1+)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback to old import path
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
except ImportError:
    # Fallback for older versions
    from langchain.document_loaders import TextLoader, PyPDFLoader

try:
    from langchain.schema import Document
except ImportError:
    # Newer versions use langchain_core
    try:
        from langchain_core.documents import Document
    except ImportError:
        # Last resort - define a simple Document class
        from typing import TypedDict
        class Document(TypedDict):
            page_content: str
            metadata: dict

# Load environment variables
script_dir = Path(__file__).parent
project_root = script_dir.parent
for env_path in [script_dir / ".env", project_root / ".env.local", project_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break


class RAGSystem:
    """RAG system for knowledge base retrieval"""
    
    def __init__(
        self,
        knowledge_base_dir: Optional[Path] = None,
        embedding_model: str = "google",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize RAG system
        
        Args:
            knowledge_base_dir: Path to knowledge base directory (default: project_root/knowledge_base)
            embedding_model: "google" for Google embeddings API or "local" for sentence-transformers
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.project_root = project_root
        self.knowledge_base_dir = knowledge_base_dir or (project_root / "knowledge_base")
        self.documents_dir = self.knowledge_base_dir / "documents"
        self.embeddings_dir = self.knowledge_base_dir / "embeddings" / "faiss_index"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding client
        self.embedding_client = None
        if embedding_model == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.embedding_client = genai.Client(api_key=api_key)
            else:
                print("⚠️  GOOGLE_API_KEY not found, falling back to local embeddings")
                self.embedding_model = "local"
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.metadata: List[Dict] = []
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata if available"""
        index_path = self.embeddings_dir / "index.faiss"
        metadata_path = self.embeddings_dir / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✅ Loaded existing FAISS index with {len(self.metadata)} chunks")
            except Exception as e:
                print(f"⚠️  Error loading index: {e}")
                self.index = None
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        if self.index is None:
            return
        
        index_path = self.embeddings_dir / "index.faiss"
        metadata_path = self.embeddings_dir / "metadata.pkl"
        
        try:
            faiss.write_index(self.index, str(index_path))
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"💾 Saved FAISS index with {len(self.metadata)} chunks")
        except Exception as e:
            print(f"❌ Error saving index: {e}")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.embedding_model == "google" and self.embedding_client:
            try:
                # Try Google's embedding API (various possible formats)
                # Method 1: Try direct API call
                try:
                    result = self.embedding_client.models.embed_content(
                        model="text-embedding-004",
                        content=text
                    )
                    if hasattr(result, 'embedding'):
                        embedding = np.array(result.embedding, dtype=np.float32)
                        return embedding
                except:
                    pass
                
                # Method 2: Try alternative API format
                try:
                    # Some versions use different API structure
                    embedding_service = self.embedding_client.models
                    result = embedding_service.embed_content(
                        model="text-embedding-004",
                        content=text
                    )
                    embedding = np.array(result.embedding, dtype=np.float32)
                    return embedding
                except:
                    pass
                
                # If both fail, fall through to local
                print("⚠️  Google embedding API format not recognized, using local embeddings")
                self.embedding_model = "local"
            except Exception as e:
                print(f"⚠️  Google embedding failed: {e}, trying local")
                self.embedding_model = "local"
        
        # Use local embeddings (sentence-transformers) - more reliable
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_local_model'):
                print("📦 Loading sentence-transformers model (first time may take a moment)...")
                self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = self._local_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the documents directory"""
        if not self.documents_dir.exists():
            print(f"⚠️  Documents directory not found: {self.documents_dir}")
            return []
        
        all_documents = []
        
        # Load text files
        text_files = list(self.documents_dir.glob("*.txt"))
        for text_file in text_files:
            try:
                loader = TextLoader(str(text_file), encoding='utf-8')
                docs = loader.load()
                all_documents.extend(docs)
                print(f"📄 Loaded: {text_file.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"⚠️  Error loading {text_file.name}: {e}")
        
        # Load PDF files
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                all_documents.extend(docs)
                print(f"📄 Loaded: {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"⚠️  Error loading {pdf_file.name}: {e}")
        
        return all_documents
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build FAISS index from documents
        
        Args:
            force_rebuild: If True, rebuild index even if it exists
        """
        if self.index is not None and not force_rebuild:
            print("✅ Index already exists. Use force_rebuild=True to rebuild.")
            return
        
        print("🔨 Building RAG index...")
        
        # Load documents
        raw_documents = self.load_documents()
        if not raw_documents:
            print("❌ No documents found to index")
            return
        
        # Split documents into chunks
        print("✂️  Splitting documents into chunks...")
        self.documents = self.text_splitter.split_documents(raw_documents)
        print(f"📦 Created {len(self.documents)} chunks")
        
        if not self.documents:
            print("❌ No chunks created")
            return
        
        # Generate embeddings
        print("🧮 Generating embeddings...")
        embeddings = []
        self.metadata = []
        
        for i, doc in enumerate(self.documents):
            if i % 10 == 0:
                print(f"   Processing chunk {i+1}/{len(self.documents)}...")
            
            embedding = self._generate_embedding(doc.page_content)
            embeddings.append(embedding)
            
            # Store metadata
            self.metadata.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
                'chunk_index': i
            })
        
        # Create FAISS index
        print("🔍 Creating FAISS index...")
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Use L2 distance (Inner Product for cosine similarity)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        print(f"✅ Index built with {self.index.ntotal} vectors (dimension: {dimension})")
        
        # Save index
        self._save_index()
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with 'content', 'source', and 'score'
        """
        if self.index is None or len(self.metadata) == 0:
            print("⚠️  Index not built. Call build_index() first.")
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = {
                    'content': self.metadata[idx]['content'],
                    'source': self.metadata[idx]['source'],
                    'score': float(distances[0][i]),
                    'chunk_index': self.metadata[idx]['chunk_index']
                }
                results.append(result)
        
        return results
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve context as formatted string for LLM
        
        Args:
            query: User query
            top_k: Number of top results to include
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Context {i} from {Path(result['source']).name}]\n{result['content']}\n"
            )
        
        return "\n".join(context_parts)


def main():
    """Test the RAG system"""
    print("=" * 60)
    print("🧪 Testing RAG System")
    print("=" * 60)
    
    # Initialize RAG
    rag = RAGSystem()
    
    # Build index
    rag.build_index(force_rebuild=False)
    
    # Test retrieval
    test_queries = [
        "What is the return policy?",
        "How do I contact support?",
        "What are the shipping options?",
    ]
    
    print("\n" + "=" * 60)
    print("🔍 Testing Retrieval")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = rag.retrieve(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i} (score: {result['score']:.4f}):")
                print(f"  Source: {result['source']}")
                print(f"  Content: {result['content'][:200]}...")
        else:
            print("  No results found")


if __name__ == "__main__":
    main()

