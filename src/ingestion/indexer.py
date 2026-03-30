import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langfuse.decorators import observe

from src.config.settings import settings

class Indexer:
    def __init__(self, collection_name: str):
        """Initializes Qdrant Client and Gemini Embeddings."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = collection_name
        
        # task_type="retrieval_document" for optimized indexing.
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            task_type="retrieval_document",
            google_api_key=settings.gemini_api_key
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the Qdrant collection if it does not exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print(f"[*] Created new Qdrant collection: {self.collection_name}")

    def has_document(self, source_filename: str) -> bool:
        """Checks if chunks from a specific document already exist in the collection."""
        count_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
            ),
            limit=1
        )
        return len(count_result) > 0

    def delete_by_source(self, source_filename: str) -> bool:
        """Deletes all vectors associated with a specific document source."""
        if not self.has_document(source_filename):
            print(f"[-] No existing vectors found for '{source_filename}' in '{self.collection_name}'.")
            return False

        print(f"[!] Deleting existing vectors for '{source_filename}' from '{self.collection_name}'...")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
            )
        )
        return True

    @observe(as_type="generation", name="embedding_and_indexing")
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Embeds text chunks and upserts them into Qdrant."""
        if not chunks:
            print("[-] No chunks provided for indexing.")
            return

        texts = [chunk["text"] for chunk in chunks]
        
        print(f"[*] Generating embeddings for {len(texts)} chunks...")
        vectors = self.embeddings.embed_documents(texts)
        
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            payload = {"text": chunk["text"], **chunk["metadata"]}
            
            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )
        
        print(f"[*] Upserting vectors into collection '{self.collection_name}'...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("[+] Indexing complete.")