from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config.settings import settings

def fetch_reference_context(collection_name: str, reference_name: str) -> str:
    """
    TOOL FOR AI AGENT:
    Use this tool when the current context cites another law, directive, or article 
    (e.g., "2547 Sayılı Kanun", "Öğrenci İşleri Yönetmeliği") and you need to know 
    what that referenced document says to provide a complete answer.
    """
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=settings.gemini_api_key
    )
    
    try:
        # Perform a semantic search specifically for the referenced law/directive
        query_vector = embeddings.embed_query(reference_name)
        
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3 # Fetch the top 3 most relevant chunks about this reference
        )
        
        if not search_result:
            return f"No additional context could be found in the database for reference: {reference_name}."
        
        # Combine the retrieved texts
        context = "\n\n".join([hit.payload.get("text", "") for hit in search_result])
        return f"--- CONTEXT FOR REFERENCED DOCUMENT ({reference_name}) ---\n{context}"
        
    except Exception as e:
        return f"Tool execution failed. Error: {e}"