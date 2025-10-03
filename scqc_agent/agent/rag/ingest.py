"""Document ingestion for knowledge base construction."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

# Import guards for optional dependencies
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str


class DocumentIngestor:
    """Ingests and chunks documents for the knowledge base."""
    
    def __init__(self, 
                 chunk_size: int = 800, 
                 chunk_overlap: int = 100, 
                 min_chunk_size: int = 300):
        """Initialize the document ingestor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum size of chunks to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            )
        else:
            self.text_splitter = None
    
    def ingest_directory(self, kb_path: str) -> List[DocumentChunk]:
        """Ingest all documents from a knowledge base directory.
        
        Args:
            kb_path: Path to knowledge base directory
            
        Returns:
            List of document chunks
        """
        kb_dir = Path(kb_path)
        if not kb_dir.exists():
            raise ValueError(f"Knowledge base directory not found: {kb_path}")
        
        chunks = []
        
        # Process all text files
        for file_path in kb_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".md", ".txt", ".yaml", ".yml"]:
                try:
                    file_chunks = self.ingest_file(file_path)
                    chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Warning: Could not ingest {file_path}: {e}")
        
        return chunks
    
    def ingest_file(self, file_path: Path) -> List[DocumentChunk]:
        """Ingest a single file and return chunks.
        
        Args:
            file_path: Path to the file to ingest
            
        Returns:
            List of document chunks from this file
        """
        content = self._read_file(file_path)
        if not content.strip():
            return []
        
        # Extract metadata
        metadata = self._extract_metadata(file_path, content)
        
        # Split into chunks
        if self.text_splitter and LANGCHAIN_AVAILABLE:
            # Use LangChain text splitter
            docs = self.text_splitter.create_documents([content], [metadata])
            chunks = []
            for i, doc in enumerate(docs):
                if len(doc.page_content) >= self.min_chunk_size:
                    chunk = DocumentChunk(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        chunk_id=f"{file_path.stem}_{i:03d}",
                        source_file=str(file_path)
                    )
                    chunks.append(chunk)
        else:
            # Fallback chunking
            chunks = self._fallback_chunk_text(content, metadata, file_path)
        
        return chunks
    
    def _read_file(self, file_path: Path) -> str:
        """Read file content with encoding detection."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from file path and content."""
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix,
            "file_size": len(content)
        }
        
        # Extract frontmatter if present
        if content.startswith("---"):
            frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
            if frontmatter_match and YAML_AVAILABLE:
                try:
                    frontmatter = yaml.safe_load(frontmatter_match.group(1))
                    if isinstance(frontmatter, dict):
                        metadata.update(frontmatter)
                except yaml.YAMLError:
                    pass
        
        # Extract title from first heading
        title_match = re.search(r"^#+ (.+)$", content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Extract document type hints from filename
        if "qc" in file_path.name.lower():
            metadata["topic"] = "quality_control"
        elif "scar" in file_path.name.lower():
            metadata["topic"] = "denoising"
        elif "scvi" in file_path.name.lower():
            metadata["topic"] = "integration"
        elif "doublet" in file_path.name.lower():
            metadata["topic"] = "doublets"
        elif "graph" in file_path.name.lower():
            metadata["topic"] = "graph_analysis"
        
        return metadata
    
    def _fallback_chunk_text(self, content: str, metadata: Dict[str, Any], 
                           file_path: Path) -> List[DocumentChunk]:
        """Fallback text chunking when LangChain is not available."""
        chunks = []
        
        # Preserve code blocks intact
        code_blocks = []
        code_pattern = r"```[\s\S]*?```"
        
        def replace_code_block(match):
            code_blocks.append(match.group(0))
            return f"<<CODE_BLOCK_{len(code_blocks)-1}>>"
        
        content_with_placeholders = re.sub(code_pattern, replace_code_block, content)
        
        # Split by paragraphs first
        paragraphs = content_with_placeholders.split("\n\n")
        
        current_chunk = ""
        chunk_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = self._restore_code_blocks(current_chunk, code_blocks)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata=metadata.copy(),
                        chunk_id=f"{file_path.stem}_{chunk_count:03d}",
                        source_file=str(file_path)
                    )
                    chunks.append(chunk)
                    chunk_count += 1
                
                # Start new chunk with overlap
                if len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunk_text = self._restore_code_blocks(current_chunk, code_blocks)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    chunk_id=f"{file_path.stem}_{chunk_count:03d}",
                    source_file=str(file_path)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restore code block placeholders with actual code blocks."""
        for i, code_block in enumerate(code_blocks):
            placeholder = f"<<CODE_BLOCK_{i}>>"
            text = text.replace(placeholder, code_block)
        return text
    
    def save_chunks(self, chunks: List[DocumentChunk], output_path: str) -> None:
        """Save chunks to a JSON file.
        
        Args:
            chunks: List of document chunks to save
            output_path: Path to save the chunks JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    def load_chunks(self, input_path: str) -> List[DocumentChunk]:
        """Load chunks from a JSON file.
        
        Args:
            input_path: Path to the chunks JSON file
            
        Returns:
            List of document chunks
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for chunk_data in chunks_data:
            chunk = DocumentChunk(
                content=chunk_data["content"],
                metadata=chunk_data["metadata"],
                chunk_id=chunk_data["chunk_id"],
                source_file=chunk_data["source_file"]
            )
            chunks.append(chunk)
        
        return chunks


def ingest_knowledge_base(kb_path: str, output_dir: str) -> None:
    """Convenience function to ingest a knowledge base directory.
    
    Args:
        kb_path: Path to knowledge base directory
        output_dir: Directory to save processed chunks
    """
    ingestor = DocumentIngestor()
    chunks = ingestor.ingest_directory(kb_path)
    
    output_path = Path(output_dir) / "chunks.json"
    ingestor.save_chunks(chunks, str(output_path))
    
    print(f"Ingested {len(chunks)} chunks from {kb_path}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python ingest.py <kb_path> <output_dir>")
        sys.exit(1)
    
    kb_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    ingest_knowledge_base(kb_path, output_dir)
