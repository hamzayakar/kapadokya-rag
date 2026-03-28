from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config.settings import settings
from langfuse.decorators import observe

def fetch_document_context(collection_name: str, source_filename: str) -> str:
    """
    TOOL FOR AI AGENT:
    Use this tool ONLY if the initially retrieved chunks are insufficient to answer the user's query, 
    but you know which document (source_filename) contains the potential answer.
    This fetches ALL chunks (the entire document context) belonging to that specific source file.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    # Retrieve all chunks matching the exact source filename
    scroll_result, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
        ),
        limit=100, # Assuming max 100 chunks per document
        with_payload=True
    )
    
    if not scroll_result:
        return f"No additional context found for document: {source_filename}"
        
    full_context = "\n\n".join([point.payload.get("text", "") for point in scroll_result])
    return f"--- EXTENDED CONTEXT FOR {source_filename} ---\n{full_context}"