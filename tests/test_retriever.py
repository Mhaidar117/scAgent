"""Tests for the RAG retriever functionality."""

import json
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from scqc_agent.agent.rag.ingest import DocumentIngestor, DocumentChunk
from scqc_agent.agent.rag.retriever import HybridRetriever


class TestDocumentIngestor:
    """Test suite for document ingestion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test documents
        self.test_md = self.temp_dir / "test.md"
        self.test_md.write_text("""
# Test Document

This is a test document for quality control.

## QC Guidelines

Quality control is important for scRNA-seq analysis.
- Minimum genes per cell: 200
- Maximum mitochondrial percentage: 20%

```python
# Example code
sc.pp.calculate_qc_metrics(adata)
```

## Conclusion

Always validate your QC thresholds.
""")
        
        self.test_txt = self.temp_dir / "guidelines.txt"
        self.test_txt.write_text("""
Standard Operating Procedure for QC

1. Load data
2. Compute metrics
3. Apply filters
4. Validate results
""")
    
    def test_ingest_file_markdown(self):
        """Test ingesting a markdown file."""
        ingestor = DocumentIngestor(chunk_size=200, chunk_overlap=50, min_chunk_size=100)
        
        chunks = ingestor.ingest_file(self.test_md)
        
        assert len(chunks) > 0
        
        # Check first chunk
        first_chunk = chunks[0]
        assert isinstance(first_chunk, DocumentChunk)
        assert "Test Document" in first_chunk.content
        assert first_chunk.metadata["filename"] == "test.md"
        assert first_chunk.metadata["file_type"] == ".md"
        assert first_chunk.metadata["title"] == "Test Document"
        assert first_chunk.source_file == str(self.test_md)
    
    def test_ingest_file_text(self):
        """Test ingesting a text file."""
        ingestor = DocumentIngestor()
        
        chunks = ingestor.ingest_file(self.test_txt)
        
        assert len(chunks) > 0
        assert "Standard Operating Procedure" in chunks[0].content
        assert chunks[0].metadata["filename"] == "guidelines.txt"
    
    def test_ingest_directory(self):
        """Test ingesting an entire directory."""
        ingestor = DocumentIngestor()
        
        chunks = ingestor.ingest_directory(str(self.temp_dir))
        
        # Should have chunks from both files
        assert len(chunks) >= 2
        
        filenames = [chunk.metadata["filename"] for chunk in chunks]
        assert "test.md" in filenames
        assert "guidelines.txt" in filenames
    
    def test_save_and_load_chunks(self):
        """Test saving and loading chunks."""
        ingestor = DocumentIngestor()
        
        chunks = ingestor.ingest_file(self.test_md)
        
        # Save chunks
        output_file = self.temp_dir / "chunks.json"
        ingestor.save_chunks(chunks, str(output_file))
        
        assert output_file.exists()
        
        # Load chunks
        loaded_chunks = ingestor.load_chunks(str(output_file))
        
        assert len(loaded_chunks) == len(chunks)
        assert loaded_chunks[0].content == chunks[0].content
        assert loaded_chunks[0].metadata == chunks[0].metadata
    
    def test_code_block_preservation(self):
        """Test that code blocks are preserved intact."""
        ingestor = DocumentIngestor(chunk_size=100, min_chunk_size=50)
        
        chunks = ingestor.ingest_file(self.test_md)
        
        # Find chunk with code block
        code_chunk = None
        for chunk in chunks:
            if "```python" in chunk.content:
                code_chunk = chunk
                break
        
        assert code_chunk is not None
        assert "```python" in code_chunk.content
        assert "sc.pp.calculate_qc_metrics" in code_chunk.content
        assert "```" in code_chunk.content  # Closing backticks
    
    def test_metadata_extraction(self):
        """Test metadata extraction from files."""
        # Create file with frontmatter
        yaml_file = self.temp_dir / "with_frontmatter.md"
        yaml_file.write_text("""---
title: "QC Guidelines"
topic: "quality_control"
species: "human"
---

# QC Guidelines

Content here.
""")
        
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest_file(yaml_file)
        
        assert len(chunks) > 0
        metadata = chunks[0].metadata
        
        assert metadata["title"] == "QC Guidelines"
        assert metadata["topic"] == "quality_control"
        assert metadata["species"] == "human"
    
    def test_topic_inference_from_filename(self):
        """Test topic inference from filename."""
        # Create files with topic-specific names
        qc_file = self.temp_dir / "qc_guidelines.md"
        qc_file.write_text("QC content")
        
        doublet_file = self.temp_dir / "doublet_detection.md"
        doublet_file.write_text("Doublet content")
        
        ingestor = DocumentIngestor()
        
        qc_chunks = ingestor.ingest_file(qc_file)
        assert qc_chunks[0].metadata["topic"] == "quality_control"
        
        doublet_chunks = ingestor.ingest_file(doublet_file)
        assert doublet_chunks[0].metadata["topic"] == "doublets"


class TestHybridRetriever:
    """Test suite for hybrid retriever."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.kb_dir = self.temp_dir / "kb"
        self.kb_dir.mkdir()
        
        # Create test knowledge base
        doc1 = self.kb_dir / "qc_guidelines.md"
        doc1.write_text("""
# Quality Control Guidelines

Quality control metrics are essential for scRNA-seq analysis.
Typical thresholds:
- Minimum genes per cell: 200-500
- Maximum mitochondrial percentage: 15-25%
""")
        
        doc2 = self.kb_dir / "doublet_detection.md"
        doc2.write_text("""
# Doublet Detection

Doublets are droplets containing multiple cells.
Use Scrublet with expected rate 0.06 for 10X data.
""")
        
        doc3 = self.kb_dir / "batch_correction.md"
        doc3.write_text("""
# Batch Correction with scVI

scVI provides state-of-the-art batch correction.
Use 20-30 latent dimensions for most datasets.
""")
    
    @patch('scqc_agent.agent.rag.retriever.FAISS_AVAILABLE', False)
    @patch('scqc_agent.agent.rag.retriever.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    @patch('scqc_agent.agent.rag.retriever.BM25_AVAILABLE', False)
    def test_retriever_initialization_no_deps(self):
        """Test retriever initialization when dependencies are not available."""
        retriever = HybridRetriever(str(self.kb_dir))
        
        # Should still work but with limited functionality
        assert retriever.documents is not None
        assert len(retriever.documents) > 0
    
    @patch('scqc_agent.agent.rag.retriever.BM25_AVAILABLE', True)
    @patch('scqc_agent.agent.rag.retriever.FAISS_AVAILABLE', False)
    @patch('scqc_agent.agent.rag.retriever.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_retriever_bm25_only(self):
        """Test retriever with BM25 only."""
        with patch('scqc_agent.agent.rag.retriever.BM25Okapi') as mock_bm25:
            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            mock_bm25_instance.get_scores.return_value = np.array([0.5, 0.8, 0.3])
            
            retriever = HybridRetriever(str(self.kb_dir))
            
            # Mock the normalization to avoid issues
            retriever._normalize_scores = Mock(return_value=np.array([0.3, 0.8, 0.1]))
            
            results = retriever.retrieve("quality control", k=2)
            
            assert len(results) <= 2
            assert all(hasattr(doc, 'page_content') for doc in results)
    
    def test_retriever_fallback_search(self):
        """Test retriever fallback when no search methods available."""
        retriever = HybridRetriever(str(self.kb_dir))
        
        # Force no search methods
        retriever.bm25 = None
        retriever.vector_index = None
        
        results = retriever.retrieve("quality control", k=2)
        
        # Should return first k documents as fallback
        assert len(results) == 2
        assert all(hasattr(doc, 'page_content') for doc in results)
    
    def test_normalize_scores(self):
        """Test score normalization."""
        retriever = HybridRetriever(str(self.kb_dir))
        
        # Test normal case
        scores = np.array([1.0, 5.0, 3.0, 2.0])
        normalized = retriever._normalize_scores(scores)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        
        # Test edge case - all same scores
        same_scores = np.array([2.0, 2.0, 2.0])
        normalized_same = retriever._normalize_scores(same_scores)
        
        assert np.all(normalized_same == normalized_same[0])  # All should be equal
    
    def test_search_by_metadata(self):
        """Test metadata-based search."""
        retriever = HybridRetriever(str(self.kb_dir))
        
        # Search by filename
        results = retriever.search_by_metadata({"filename": "qc_guidelines.md"}, k=1)
        
        assert len(results) == 1
        assert "Quality Control Guidelines" in results[0].page_content
        
        # Search by inferred topic
        results = retriever.search_by_metadata({"topic": "doublets"}, k=1)
        
        assert len(results) >= 1
        # Should find the doublet detection document
        found_doublet_doc = any("Doublet Detection" in doc.page_content for doc in results)
        assert found_doublet_doc
    
    def test_add_documents(self):
        """Test adding new documents to the index."""
        retriever = HybridRetriever(str(self.kb_dir))
        initial_count = len(retriever.documents)
        
        # Add new documents
        new_docs = ["New QC content about filtering"]
        new_metadata = [{"source": "dynamic", "topic": "qc"}]
        
        # Mock the rebuild to avoid dependency issues
        with patch.object(retriever, '_build_index'):
            retriever.add_documents(new_docs, new_metadata)
        
        assert len(retriever.documents) == initial_count + 1
        assert len(retriever.chunks) == initial_count + 1
    
    def test_get_stats(self):
        """Test retriever statistics."""
        retriever = HybridRetriever(str(self.kb_dir))
        
        stats = retriever.get_stats()
        
        assert "num_documents" in stats
        assert "num_chunks" in stats
        assert "has_bm25" in stats
        assert "has_vector" in stats
        assert "model_name" in stats
        assert "alpha" in stats
        assert "index_dir" in stats
        
        assert stats["num_documents"] > 0
        assert stats["num_chunks"] > 0
    
    def test_index_persistence(self):
        """Test that index is saved and loaded correctly."""
        # First retriever - builds index
        retriever1 = HybridRetriever(str(self.kb_dir))
        initial_docs = len(retriever1.documents)
        
        # Index directory should be created
        index_dir = self.kb_dir / "kb.index"
        assert index_dir.exists()
        assert (index_dir / "chunks.json").exists()
        assert (index_dir / "metadata.json").exists()
        
        # Second retriever - loads existing index
        retriever2 = HybridRetriever(str(self.kb_dir))
        
        assert len(retriever2.documents) == initial_docs
        assert len(retriever2.chunks) == len(retriever1.chunks)
    
    def test_empty_kb_handling(self):
        """Test handling of empty knowledge base."""
        empty_kb = self.temp_dir / "empty_kb"
        empty_kb.mkdir()
        
        retriever = HybridRetriever(str(empty_kb))
        
        assert len(retriever.documents) == 0
        assert len(retriever.chunks) == 0
        
        # Should handle search gracefully
        results = retriever.retrieve("test query", k=5)
        assert len(results) == 0


def test_integration_ingest_and_retrieve():
    """Integration test for ingestion and retrieval."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_dir = Path(temp_dir) / "test_kb"
        kb_dir.mkdir()
        
        # Create test documents
        qc_doc = kb_dir / "qc_guide.md"
        qc_doc.write_text("""
# QC Guide

Filter cells with less than 200 genes.
Remove cells with more than 20% mitochondrial genes.
""")
        
        doublet_doc = kb_dir / "doublets.md"
        doublet_doc.write_text("""
# Doublet Guide

Use Scrublet for doublet detection.
Expected doublet rate is typically 6% for 10X data.
""")
        
        # Create retriever (will ingest documents)
        retriever = HybridRetriever(str(kb_dir))
        
        # Test retrieval
        qc_results = retriever.retrieve("filter cells genes", k=1)
        assert len(qc_results) >= 1
        
        doublet_results = retriever.retrieve("doublet detection scrublet", k=1)
        assert len(doublet_results) >= 1
        
        # Should be able to find relevant content
        qc_content = qc_results[0].page_content
        assert "200 genes" in qc_content or "mitochondrial" in qc_content
        
        doublet_content = doublet_results[0].page_content
        assert "Scrublet" in doublet_content or "doublet" in doublet_content.lower()


if __name__ == "__main__":
    pytest.main([__file__])
