from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.config.settings import settings

def fetch_hierarchical_context(collection_name: str, parent_summary: str, source_filename: str) -> str:
    """
    TOOL FOR AI AGENT:
    Use this tool to fetch ALL sub-clauses (the entire family of chunks) belonging to a specific parent article.
    ONLY use this if the current chunk's metadata contains a 'parent_summary'.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    try:
        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source_filename)),
                    FieldCondition(key="parent_summary", match=MatchValue(value=parent_summary))
                ]
            ),
            limit=50,
            with_payload=True
        )
        
        if not scroll_result:
            return f"No hierarchical context found for parent: '{parent_summary}' in {source_filename}."
            
        context = "\n\n".join([point.payload.get("text", "") for point in scroll_result])
        return f"--- HIERARCHICAL CONTEXT FOR '{parent_summary}' ---\n{context}"
        
    except Exception as e:
        return f"Tool execution failed: {str(e)}"