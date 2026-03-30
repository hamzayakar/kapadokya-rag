import json
import logging
import google.generativeai as genai
from google.generativeai.types import content_types
from langfuse.decorators import observe, langfuse_context

from src.config.settings import settings
from src.rag.retrievers.basic_retriever import BasicRetriever
from src.rag.langfuse_client import lf_manager
from src.tools.document_searcher import search_within_document
from src.tools.hierarchy_fetcher import fetch_hierarchical_context
from src.tools.reference_fetcher import fetch_reference_context

logger = logging.getLogger(__name__)
genai.configure(api_key=settings.gemini_api_key)

class KapadokyaAgent:
    def __init__(self, collection_name: str, strategy: str):
        self.collection_name = collection_name
        self.strategy = strategy
        self.retriever = BasicRetriever(collection_name=collection_name)
        
        router_config = lf_manager.get_prompt_and_config("kapadokya_router")
        agent_config = lf_manager.get_prompt_and_config("kapadokya_agent")
        
        self.router_model_name = router_config["model_name"]
        self.agent_model_name = agent_config["model_name"]

        # Tool wrappers
        def document_search_wrapper(query: str, source_filename: str) -> str:
            return search_within_document(self.collection_name, query, source_filename)
            
        def hierarchy_fetch_wrapper(parent_summary: str, source_filename: str) -> str:
            return fetch_hierarchical_context(self.collection_name, parent_summary, source_filename)
            
        def reference_fetch_wrapper(reference_name: str) -> str:
            return fetch_reference_context(self.collection_name, reference_name)

        # Dynamic tool allocation based on strategy
        assigned_tools = [document_search_wrapper, reference_fetch_wrapper]
        if self.strategy == "parent_child":
            assigned_tools.append(hierarchy_fetch_wrapper)

        self.router_model = genai.GenerativeModel(
            model_name=self.router_model_name,
            system_instruction=router_config["system_prompt"],
            generation_config=genai.GenerationConfig(
                temperature=router_config["temperature"],
                response_mime_type="application/json" 
            )
        )
        
        self.agent_model = genai.GenerativeModel(
            model_name=self.agent_model_name,
            system_instruction=agent_config["system_prompt"],
            tools=assigned_tools,
            generation_config=genai.GenerationConfig(
                temperature=agent_config["temperature"]
            )
        )

    def _format_history_for_gemini(self, gradio_history: list) -> list:
        gemini_history = []
        for msg in gradio_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append(
                content_types.ContentDict(role=role, parts=[msg["content"]])
            )
        return gemini_history

    @observe(as_type="generation", name="router_decision")
    def route_query(self, user_query: str, formatted_history: list) -> dict:
        chat = self.router_model.start_chat(history=formatted_history)
        prompt = (
            f"Analyze this user query: '{user_query}'. "
            "Respond ONLY in JSON with two keys: 'intent' (strictly either 'chat' or 'rag'), "
            "and 'response' (if intent is 'chat', put your conversational answer here. If 'rag', leave empty)."
        )
        response = chat.send_message(prompt)
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            langfuse_context.update_current_observation(
                usage={
                    "input": usage.prompt_token_count, 
                    "output": usage.candidates_token_count, 
                    "total": usage.total_token_count
                },
                model=self.router_model_name
            )
            
        # Fail-fast: Do not catch exception, let it crash if router fails
        return json.loads(response.text)

    @observe(as_type="generation", name="agent_execution")
    def ask(self, user_query: str, gradio_history: list = None) -> str:
        # EXPLICITLY SET INPUT AND TAGS FOR LANGFUSE UI
        langfuse_context.update_current_observation(input=user_query)
        langfuse_context.update_current_trace(
            tags=[self.strategy],
            metadata={"collection": self.collection_name}
        )

        if gradio_history is None:
            gradio_history = []
            
        formatted_history = self._format_history_for_gemini(gradio_history)
        
        route_decision = self.route_query(user_query, formatted_history)
        
        # Fail-fast strictly checking "intent"
        if route_decision["intent"] == "chat":
            logger.info("Router: Handled as casual chat.")
            # EXPLICITLY SET OUTPUT FOR CHAT INTENT
            langfuse_context.update_current_observation(output=route_decision["response"])
            return route_decision["response"]
            
        logger.info("Router: Proceeding with Vector Retrieval...")
        initial_chunks = self.retriever.retrieve(query=user_query, top_k=5)
        
        context_str = "--- RETRIEVED CONTEXT ---\n"
        for chunk in initial_chunks:
            context_str += f"[Source: {chunk['source']} | Strategy: {chunk['strategy']}]\n{chunk['text']}\n\n"
            
        full_prompt = f"USER QUERY: {user_query}\n\n{context_str}"
        
        chat = self.agent_model.start_chat(history=formatted_history)
        logger.info("Agent: Generating response (and calling tools if necessary)...")
        response = chat.send_message(full_prompt)
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            langfuse_context.update_current_observation(
                usage={
                    "input": usage.prompt_token_count, 
                    "output": usage.candidates_token_count, 
                    "total": usage.total_token_count
                },
                model=self.agent_model_name
            )
            
        # EXPLICITLY SET OUTPUT FOR RAG INTENT
        langfuse_context.update_current_observation(output=response.text)
        return response.text