"""Hybrid retriever combining BM25 and vector search for RAG."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Import guards for optional dependencies
try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    
    # Mock Document class for fallback
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any]):
            self.page_content = page_content
            self.metadata = metadata

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from .ingest import DocumentChunk, DocumentIngestor


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search."""
    
    def __init__(self, 
                 kb_path: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 alpha: float = 0.5):
        """Initialize the hybrid retriever.
        
        Args:
            kb_path: Path to knowledge base directory
            model_name: Name of sentence transformer model
            alpha: Weight for combining BM25 (1-alpha) and vector (alpha) scores
        """
        self.kb_path = Path(kb_path)
        self.model_name = model_name
        self.alpha = alpha
        
        # Initialize components
        self.chunks: List[DocumentChunk] = []
        self.documents: List[Document] = []
        self.bm25: Optional[Any] = None
        self.vector_index: Optional[Any] = None
        self.embeddings: Optional[np.ndarray] = None
        self.sentence_model: Optional[Any] = None
        
        # Check if index exists and load, otherwise build
        self.index_dir = self.kb_path / "kb.index"
        if self.index_dir.exists():
            self._load_index()
        else:
            self._build_index()
    
    def _build_index(self) -> None:
        """Build the retrieval index from scratch."""
        print(f"Building retrieval index for {self.kb_path}")
        
        # Ingest documents
        ingestor = DocumentIngestor()
        self.chunks = ingestor.ingest_directory(str(self.kb_path))
        
        if not self.chunks:
            print("Warning: No documents found in knowledge base")
            return
        
        # Convert to Document objects
        self.documents = []
        for chunk in self.chunks:
            doc = Document(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            self.documents.append(doc)
        
        # Build BM25 index
        if BM25_AVAILABLE:
            tokenized_docs = [doc.page_content.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            print("Warning: BM25 not available, using fallback")
        
        # Build vector index
        if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
            self.sentence_model = SentenceTransformer(self.model_name)
            
            # Generate embeddings
            texts = [doc.page_content for doc in self.documents]
            self.embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.vector_index.add(self.embeddings)
        else:
            print("Warning: Vector search not available, using BM25 only")
        
        # Save index
        self._save_index()
        print(f"Built index with {len(self.documents)} documents")
    
    def _save_index(self) -> None:
        """Save the retrieval index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file
            })
        
        with open(self.index_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save BM25 index
        if self.bm25 and BM25_AVAILABLE:
            with open(self.index_dir / "bm25.pkl", 'wb') as f:
                pickle.dump(self.bm25, f)
        
        # Save vector index and embeddings
        if self.vector_index and FAISS_AVAILABLE:
            faiss.write_index(self.vector_index, str(self.index_dir / "vector.index"))
            np.save(self.index_dir / "embeddings.npy", self.embeddings)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "num_documents": len(self.documents),
            "components": {
                "bm25": self.bm25 is not None,
                "vector": self.vector_index is not None
            }
        }
        
        with open(self.index_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_index(self) -> None:
        """Load the retrieval index from disk."""
        try:
            # Load metadata
            with open(self.index_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get("model_name", self.model_name)
            self.alpha = metadata.get("alpha", self.alpha)
            
            # Load chunks
            with open(self.index_dir / "chunks.json", 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            self.chunks = []
            for chunk_data in chunks_data:
                chunk = DocumentChunk(
                    content=chunk_data["content"],
                    metadata=chunk_data["metadata"],
                    chunk_id=chunk_data["chunk_id"],
                    source_file=chunk_data["source_file"]
                )
                self.chunks.append(chunk)
            
            # Convert to Document objects
            self.documents = []
            for chunk in self.chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata=chunk.metadata
                )
                self.documents.append(doc)
            
            # Load BM25 index
            bm25_path = self.index_dir / "bm25.pkl"
            if bm25_path.exists() and BM25_AVAILABLE:
                with open(bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
            
            # Load vector index
            vector_path = self.index_dir / "vector.index"
            embeddings_path = self.index_dir / "embeddings.npy"
            
            if (vector_path.exists() and embeddings_path.exists() and 
                FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE):
                self.vector_index = faiss.read_index(str(vector_path))
                self.embeddings = np.load(embeddings_path)
                self.sentence_model = SentenceTransformer(self.model_name)
            
            print(f"Loaded index with {len(self.documents)} documents")
            
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Rebuilding index...")
            self._build_index()
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.documents:
            return []
        
        # Get BM25 scores
        bm25_scores = None
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get vector scores
        vector_scores = None
        if self.vector_index and self.sentence_model:
            query_embedding = self.sentence_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search vector index
            scores, indices = self.vector_index.search(query_embedding, len(self.documents))
            vector_scores = scores[0]
        
        # Combine scores
        if bm25_scores is not None and vector_scores is not None:
            # Normalize scores to [0, 1]
            bm25_norm = self._normalize_scores(bm25_scores)
            vector_norm = self._normalize_scores(vector_scores)
            
            # Combine with weighted average
            combined_scores = (1 - self.alpha) * bm25_norm + self.alpha * vector_norm
        elif bm25_scores is not None:
            combined_scores = self._normalize_scores(bm25_scores)
        elif vector_scores is not None:
            combined_scores = self._normalize_scores(vector_scores)
        else:
            # Fallback: simple text matching when no search engines available
            return self._simple_text_search(query, k)
        
        # Get top k documents
        top_indices = np.argsort(combined_scores)[::-1][:k]
        top_documents = [self.documents[i] for i in top_indices]
        
        return top_documents
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.ones_like(scores) / len(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _simple_text_search(self, query: str, k: int) -> List[Document]:
        """Fallback text search when no advanced search methods available."""
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        # Score documents based on term matches
        scored_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            
            # Count exact term matches
            term_matches = sum(1 for term in query_terms if term in content_lower)
            
            # Boost score for title/metadata matches
            title_matches = 0
            if 'title' in doc.metadata:
                title_lower = doc.metadata['title'].lower()
                title_matches = sum(1 for term in query_terms if term in title_lower)
            
            # Combined score: content matches + weighted title matches
            score = term_matches + (title_matches * 2)
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add new documents to the index.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
        # Create document chunks
        new_chunks = []
        for i, (text, metadata) in enumerate(zip(documents, metadatas)):
            chunk = DocumentChunk(
                content=text,
                metadata=metadata,
                chunk_id=f"dynamic_{len(self.chunks) + i:03d}",
                source_file="dynamic"
            )
            new_chunks.append(chunk)
        
        # Add to existing chunks
        self.chunks.extend(new_chunks)
        
        # Convert to Document objects
        new_docs = []
        for chunk in new_chunks:
            doc = Document(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            new_docs.append(doc)
        
        self.documents.extend(new_docs)
        
        # Rebuild indices (for simplicity - could be optimized)
        print("Rebuilding index with new documents...")
        self._build_index()
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], k: int = 10) -> List[Document]:
        """Search documents by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            k: Maximum number of documents to return
            
        Returns:
            List of filtered documents
        """
        filtered_docs = []
        
        for doc in self.documents:
            match = True
            for key, value in metadata_filter.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= k:
                    break
        
        return filtered_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.
        
        Returns:
            Dictionary with retriever statistics
        """
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "has_bm25": self.bm25 is not None,
            "has_vector": self.vector_index is not None,
            "model_name": self.model_name,
            "alpha": self.alpha,
            "index_dir": str(self.index_dir)
        }


def test_retriever(kb_path: str) -> None:
    """Test the hybrid retriever with sample queries.
    
    Args:
        kb_path: Path to knowledge base directory
    """
    retriever = HybridRetriever(kb_path)
    
    test_queries = [
        "quality control metrics",
        "doublet detection",
        "batch correction",
        "UMAP visualization",
        "filtering cells"
    ]
    
    print(f"Retriever stats: {retriever.get_stats()}")
    print()
    
    for query in test_queries:
        print(f"Query: {query}")
        docs = retriever.retrieve(query, k=3)
        
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc.metadata.get('title', 'No title')} "
                  f"({doc.metadata.get('source', 'Unknown source')})")
            print(f"     {doc.page_content[:100]}...")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python retriever.py <kb_path>")
        sys.exit(1)
    
    kb_path = sys.argv[1]
    test_retriever(kb_path)
