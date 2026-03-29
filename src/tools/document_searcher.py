from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config.settings import settings

def search_within_document(collection_name: str, query: str, source_filename: str) -> str:
    """
    TOOL FOR AI AGENT:
    Use this tool to perform a semantic search STRICTLY WITHIN a specific document.
    Call this if a chunk is cut off or lacks context, to find the rest of the information in the same PDF.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=settings.gemini_api_key
    )
    
    try:
        query_vector = embeddings.embed_query(query)
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
            ),
            limit=3
        )
        
        if not search_result:
            return f"No relevant context found in {source_filename} for query: {query}"
        
        context = "\n\n".join([hit.payload.get("text", "") for hit in search_result])
        return f"--- SEARCH RESULTS IN {source_filename} ---\n{context}"
        
    except Exception as e:
        return f"Tool execution failed: {str(e)}"