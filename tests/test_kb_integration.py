"""Test knowledge base integration and end-to-end functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path

from scqc_agent.agent.rag.ingest import DocumentIngestor
from scqc_agent.agent.rag.retriever import HybridRetriever


class TestKBIntegration:
    """Test that the KB starter pack integrates properly."""
    
    @pytest.fixture
    def kb_path(self) -> str:
        """Return path to existing knowledge base."""
        kb_dir = Path(__file__).parent.parent / "kb"
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        return str(kb_dir)
    
    def test_kb_ingestion_works(self, kb_path: str):
        """Test that KB ingestion works with the starter pack."""
        
        # Test document ingestion
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest_directory(kb_path)
        
        assert len(chunks) > 0, "Should ingest documents from KB"
        
        # Check that we have expected document types
        sources = [chunk.source_file for chunk in chunks]
        file_names = [Path(source).name for source in sources]
        
        # Should find key documents
        expected_docs = ['qc_guidelines.md', 'doublet_detection.md', 'batch_integration.md']
        found_docs = []
        for expected in expected_docs:
            if any(expected in name for name in file_names):
                found_docs.append(expected)
        
        assert len(found_docs) > 0, f"Should find expected documents. Found files: {file_names}"
        
        print(f"Ingested {len(chunks)} chunks from {len(set(sources))} documents")
        print(f"Found expected documents: {found_docs}")
        
        # Test metadata extraction
        qc_chunks = [chunk for chunk in chunks if 'qc' in chunk.source_file.lower()]
        if qc_chunks:
            qc_chunk = qc_chunks[0]
            assert 'title' in qc_chunk.metadata or 'source' in qc_chunk.metadata, "Should have metadata"
            print(f"Sample QC chunk metadata: {qc_chunk.metadata}")
    
    def test_kb_retriever_builds_index(self, kb_path: str):
        """Test that retriever can build index from KB."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy KB to temp directory to avoid modifying original
            temp_kb = Path(temp_dir) / "test_kb"
            shutil.copytree(kb_path, temp_kb)
            
            # Remove any existing index
            index_dir = temp_kb / "kb.index"
            if index_dir.exists():
                shutil.rmtree(index_dir)
            
            # Build retriever (should create index)
            retriever = HybridRetriever(str(temp_kb))
            
            # Check that index was created
            assert index_dir.exists(), "Should create kb.index directory"
            
            # Check retriever stats
            stats = retriever.get_stats()
            assert stats["num_documents"] > 0, "Should have loaded documents"
            assert stats["num_chunks"] > 0, "Should have chunks"
            
            print(f"Built index with stats: {stats}")
    
    def test_kb_provides_relevant_citations(self, kb_path: str):
        """Test that KB provides relevant citations for common queries."""
        
        retriever = HybridRetriever(kb_path)
        
        test_cases = [
            {
                "query": "mitochondrial gene percentage threshold",
                "expected_terms": ["mito", "mitochondrial", "percentage", "threshold"],
                "description": "mitochondrial QC query"
            },
            {
                "query": "doublet detection scrublet",
                "expected_terms": ["doublet", "scrublet"],
                "description": "doublet detection query"
            },
            {
                "query": "batch correction integration",
                "expected_terms": ["batch", "correction", "integration"],
                "description": "batch integration query"
            }
        ]
        
        for case in test_cases:
            docs = retriever.retrieve(case["query"], k=3)
            assert len(docs) > 0, f"Should return documents for {case['description']}"
            
            # Check that at least one document contains relevant terms
            relevant_docs = []
            for doc in docs:
                content_lower = doc.page_content.lower()
                if any(term in content_lower for term in case["expected_terms"]):
                    relevant_docs.append(doc)
            
            assert len(relevant_docs) > 0, f"Should find relevant documents for {case['description']}. Query: {case['query']}"
            
            print(f"✓ {case['description']}: Found {len(relevant_docs)} relevant documents")
    
    def test_kb_citations_have_proper_format(self, kb_path: str):
        """Test that citations from KB have proper format for agent use."""
        
        retriever = HybridRetriever(kb_path)
        docs = retriever.retrieve("quality control metrics", k=3)
        
        for doc in docs:
            # Check document structure
            assert hasattr(doc, 'page_content'), "Document should have page_content"
            assert hasattr(doc, 'metadata'), "Document should have metadata"
            assert len(doc.page_content.strip()) > 0, "Content should not be empty"
            
            # Check metadata format
            metadata = doc.metadata
            assert isinstance(metadata, dict), "Metadata should be a dictionary"
            
            # Should have source information
            has_source_info = any(key in metadata for key in ['source', 'title', 'source_file'])
            assert has_source_info, f"Should have source information in metadata: {metadata}"
            
            # Content should be reasonable length for citations
            content_length = len(doc.page_content)
            assert 50 <= content_length <= 2000, f"Content length should be reasonable: {content_length} chars"
        
        print(f"✓ Verified citation format for {len(docs)} documents")
    
    def test_kb_performance_acceptable(self, kb_path: str):
        """Test that KB operations complete in reasonable time."""
        import time
        
        # Test retriever initialization time
        start_time = time.time()
        retriever = HybridRetriever(kb_path)
        init_time = time.time() - start_time
        
        # Should initialize reasonably quickly
        assert init_time < 30.0, f"Retriever initialization too slow: {init_time:.2f}s"
        
        # Test query time
        start_time = time.time()
        docs = retriever.retrieve("quality control thresholds", k=5)
        query_time = time.time() - start_time
        
        assert query_time < 5.0, f"Query too slow: {query_time:.2f}s"
        assert len(docs) > 0, "Should return results"
        
        print(f"✓ Performance: init={init_time:.2f}s, query={query_time:.2f}s")


class TestKBContent:
    """Test the content quality of the KB starter pack."""
    
    @pytest.fixture
    def kb_path(self) -> str:
        """Return path to knowledge base."""
        kb_dir = Path(__file__).parent.parent / "kb"
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        return str(kb_dir)
    
    def test_kb_has_essential_content(self, kb_path: str):
        """Test that KB contains essential QC knowledge."""
        
        kb_dir = Path(kb_path)
        
        # Check for key files
        essential_files = [
            "qc_guidelines.md",
            "doublet_detection.md",
            "batch_integration.md",
            "graph_analysis.md"
        ]
        
        existing_files = []
        for file_name in essential_files:
            file_path = None
            # Look for file in any subdirectory
            for candidate in kb_dir.rglob(file_name):
                if candidate.is_file():
                    file_path = candidate
                    break
            
            if file_path and file_path.exists():
                existing_files.append(file_name)
                
                # Check file is not empty
                content = file_path.read_text()
                assert len(content.strip()) > 0, f"{file_name} should not be empty"
                
                # Check for expected content patterns
                content_lower = content.lower()
                if "qc" in file_name:
                    assert any(term in content_lower for term in ["quality", "mitochondrial", "threshold"]), f"{file_name} should contain QC content"
                elif "doublet" in file_name:
                    assert any(term in content_lower for term in ["doublet", "scrublet"]), f"{file_name} should contain doublet content"
                elif "batch" in file_name:
                    assert any(term in content_lower for term in ["batch", "integration", "correction"]), f"{file_name} should contain batch content"
        
        assert len(existing_files) >= 2, f"Should have at least 2 essential files. Found: {existing_files}"
        print(f"✓ Found essential KB files: {existing_files}")
    
    def test_kb_metadata_format(self, kb_path: str):
        """Test that KB documents have proper metadata format."""
        
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest_directory(kb_path)
        
        # Check metadata format for a sample of documents
        metadata_samples = []
        for chunk in chunks[:5]:  # Check first 5 chunks
            metadata = chunk.metadata
            metadata_samples.append(metadata)
            
            # Should have basic metadata
            assert isinstance(metadata, dict), "Metadata should be a dictionary"
            
            # Should have source information
            has_source = any(key in metadata for key in ['source', 'source_file', 'title'])
            assert has_source, f"Should have source info in metadata: {metadata}"
        
        print(f"✓ Verified metadata format for {len(metadata_samples)} chunks")
        for i, meta in enumerate(metadata_samples):
            print(f"  Sample {i+1}: {dict(list(meta.items())[:3])}...")  # Show first 3 keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
