from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
import google.generativeai as genai
from src.config.settings import settings

genai.configure(api_key=settings.gemini_api_key)

def fetch_reference_context(collection_name: str, target_filename: str, query: str) -> str:
    """
    TOOL FOR AI AGENT:
    Use this tool when the current context explicitly cites an external law or internal directive.
    You MUST provide the exact target_filename (from the 'references' array) and what you want to search inside it.
    
    Args:
        collection_name: The Qdrant collection to search in (e.g., "gemini_mega").
        target_filename: The exact file name of the cited document (e.g., "2547_sayili_kanun.pdf").
        query: The specific question or concept you are looking for inside that referenced document.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    try:
        # 1. Convert the agent's semantic query into a vector
        embedding_response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_vector = embedding_response['embedding']
        
        # 2. Perform Hybrid Search: Semantic search strictly bounded by the filename metadata
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="source", 
                        match=MatchText(text=target_filename) # Filter strictly by target filename
                    )
                ]
            ),
            limit=3, # Fetch the top 3 most relevant chunks from that specific document
            score_threshold=0.45
        )
        
        if not search_result:
            return f"No relevant context found for '{query}' inside the reference '{target_filename}'."
        
        context = "\n\n".join([hit.payload.get("text", "") for hit in search_result])
        return f"--- CONTEXT FOUND FOR '{query}' IN REFERENCED DOCUMENT ({target_filename}) ---\n{context}"
        
    except Exception as e:
        return f"Tool execution failed: {str(e)}"