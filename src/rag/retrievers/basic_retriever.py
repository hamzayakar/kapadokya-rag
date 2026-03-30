from typing import List, Dict, Any
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langfuse.decorators import observe

from src.config.settings import settings

class BasicRetriever:
    def __init__(self, collection_name: str):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = collection_name
        
        # task_type="retrieval_query" for optimized searching.
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            task_type="retrieval_query",
            google_api_key=settings.gemini_api_key
        )

    @observe(as_type="span", name="qdrant_semantic_search")
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Embeds the user query and retrieves the top_k most similar chunks."""
        query_vector = self.embeddings.embed_query(query)
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Format the output for the agent
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source", "Unknown"),
                "strategy": hit.payload.get("strategy", "Unknown")
            })
            
        return results