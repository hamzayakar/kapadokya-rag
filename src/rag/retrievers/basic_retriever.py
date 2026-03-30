from typing import List, Dict, Any
from qdrant_client import QdrantClient
import google.generativeai as genai
from langfuse.decorators import observe

from src.config.settings import settings

genai.configure(api_key=settings.gemini_api_key)

class BasicRetriever:
    def __init__(self, collection_name: str):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = collection_name
        self.embedding_model = "models/text-embedding-004"

    @observe(as_type="span", name="qdrant_semantic_search")
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Embeds the user query natively and retrieves the top_k most similar chunks."""
        
        # NATIVE SDK USAGE
        embedding_response = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        query_vector = embedding_response['embedding']
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.45
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