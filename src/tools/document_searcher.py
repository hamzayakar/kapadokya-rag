from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import google.generativeai as genai
from src.config.settings import settings

genai.configure(api_key=settings.gemini_api_key)

def search_within_document(collection_name: str, query: str, source_filename: str) -> str:
    """
    TOOL FOR AI AGENT:
    Use this tool to perform a semantic search STRICTLY WITHIN a specific document.
    Call this if a chunk is cut off or lacks context, to find the rest of the information in the same PDF.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    try:
        embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_vector = embedding_response['embedding']
        
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
            ),
            limit=3,
            score_threshold=0.45
        )
        
        if not search_result:
            return f"No relevant context found in {source_filename} for query: {query}"
        
        context = "\n\n".join([hit.payload.get("text", "") for hit in search_result])
        return f"--- SEARCH RESULTS IN {source_filename} ---\n{context}"
        
    except Exception as e:
        return f"Tool execution failed: {str(e)}"