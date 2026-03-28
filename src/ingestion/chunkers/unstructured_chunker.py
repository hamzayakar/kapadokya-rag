from typing import List, Dict, Any
from langfuse.decorators import observe
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from src.ingestion.chunkers.base_chunker import BaseChunker

class UnstructuredChunker(BaseChunker):
    def __init__(self):
        super().__init__()

    @observe(as_type="span", name="unstructured_layout_chunking")
    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        # Analyze the PDF layout (Tables, lists, titles)
        elements = partition_pdf(filename=file_path, strategy="fast")
        
        # Group elements into chunks by title and logical structure
        chunks = chunk_by_title(elements)
        
        formatted_chunks = []
        base_meta = self.get_file_metadata(file_path)
        
        for i, chunk in enumerate(chunks):
            meta = base_meta.copy()
            meta["strategy"] = "unstructured_layout"
            meta["chunk_index"] = i
            
            formatted_chunks.append({
                "text": str(chunk),
                "metadata": meta
            })
            
        return formatted_chunks