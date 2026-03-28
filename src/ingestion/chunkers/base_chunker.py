import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    def __init__(self):
        """Common initialization for all chunkers."""
        pass

    @abstractmethod
    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Takes a document and returns a list of processed chunks.
        Expected output format: 
        [{"text": "...", "metadata": {"source": "...", "chunk_index": 0, ...}}, ...]
        """
        pass

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extracts basic metadata like filename from the given path."""
        filename = os.path.basename(file_path)
        return {"source": filename}