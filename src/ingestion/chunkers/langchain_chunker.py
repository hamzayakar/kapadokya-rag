from typing import List, Dict, Any
from langfuse.decorators import observe
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.ingestion.chunkers.base_chunker import BaseChunker
from src.config.settings import settings

class LangchainChunker(BaseChunker):
    def __init__(self, strategy: str = "naive", chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__()
        self.strategy = strategy
        
        # 1. Naive Strategy (Simple character-based chunking)
        if self.strategy == "naive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        # 2. Semantic Strategy (Semantic chunking)
        elif self.strategy == "semantic":
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=settings.gemini_api_key
            )
            self.splitter = SemanticChunker(embeddings)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @observe(as_type="span", name="langchain_chunking")
    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Combine all page contents into a single string for chunking
        full_text = "\n".join([page.page_content for page in pages])
        
        # Split the full text into chunks using the langchain strategy
        docs = self.splitter.create_documents([full_text])
        
        formatted_chunks = []
        base_meta = self.get_file_metadata(file_path)
        
        for i, doc in enumerate(docs):
            meta = base_meta.copy()
            meta["strategy"] = f"langchain_{self.strategy}"
            meta["chunk_index"] = i
            
            formatted_chunks.append({
                "text": doc.page_content,
                "metadata": meta
            })
            
        return formatted_chunks