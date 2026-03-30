from qdrant_client import QdrantClient
import google.generativeai as genai
from src.config.settings import settings

genai.configure(api_key=settings.gemini_api_key)

def fetch_reference_context(collection_name: str, reference_name: str) -> str:
    """
    TOOL FOR AI AGENT:
    Use this tool when the current context cites another law, directive, or article.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    try:
        embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=reference_name,
            task_type="retrieval_query"
        )
        query_vector = embedding_response['embedding']
        
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3,
            score_threshold=0.45
        )
        
        if not search_result:
            return f"No additional context could be found in the database for reference: {reference_name}."
        
        context = "\n\n".join([hit.payload.get("text", "") for hit in search_result])
        return f"--- CONTEXT FOR REFERENCED DOCUMENT ({reference_name}) ---\n{context}"
        
    except Exception as e:
        return f"Tool execution failed: {str(e)}"