import json
import logging
import google.generativeai as genai
from google.generativeai.types import content_types
from langfuse.decorators import observe, langfuse_context

from src.config.settings import settings
from src.rag.retrievers.basic_retriever import BasicRetriever
from src.rag.langfuse_client import lf_manager
from src.tools.parent_fetcher import fetch_document_context

logger = logging.getLogger(__name__)
genai.configure(api_key=settings.gemini_api_key)

class KapadokyaAgent:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.retriever = BasicRetriever(collection_name=collection_name)
        
        # Fetch prompts and configs from Langfuse
        router_config = lf_manager.get_prompt_and_config("kapadokya_router")
        agent_config = lf_manager.get_prompt_and_config("kapadokya_agent")
        
        self.router_model_name = router_config.get("model_name", "gemini-1.5-flash")
        self.agent_model_name = agent_config.get("model_name", "gemini-2.5-pro")

        # Initialize tools
        def tool_wrapper(source_filename: str) -> str:
            return fetch_document_context(self.collection_name, source_filename)
            
        # 1. ROUTER MODEL (Fast, cheap, JSON output)
        self.router_model = genai.GenerativeModel(
            model_name=self.router_model_name,
            system_instruction=router_config["system_prompt"],
            generation_config=genai.GenerationConfig(
                temperature=router_config.get("temperature", 0.0),
                response_mime_type="application/json" 
            )
        )
        
        # 2. AGENT MODEL (Pro, tool-capable, text output)
        self.agent_model = genai.GenerativeModel(
            model_name=self.agent_model_name,
            system_instruction=agent_config["system_prompt"],
            tools=[tool_wrapper],
            generation_config=genai.GenerationConfig(
                temperature=agent_config.get("temperature", 0.1)
            )
        )

    def _format_history_for_gemini(self, gradio_history: list) -> list:
        """Converts Gradio's history format to Gemini's native Content objects."""
        gemini_history = []
        for msg in gradio_history:
            # Gradio role matches Gemini role (user/model)
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append(
                content_types.ContentDict(role=role, parts=[msg["content"]])
            )
        return gemini_history

    @observe(as_type="generation", name="router_decision")
    def route_query(self, user_query: str, formatted_history: list) -> dict:
        """Determines if the query is general chat or requires RAG."""
        chat = self.router_model.start_chat(history=formatted_history)
        
        # Prompting the router specifically for JSON structure
        prompt = (
            f"Analyze this user query: '{user_query}'. "
            "Respond ONLY in JSON with two keys: 'intent' (strictly either 'chat' or 'rag'), "
            "and 'response' (if intent is 'chat', put your conversational answer here. If 'rag', leave empty)."
        )
        
        response = chat.send_message(prompt)
        
        # Cost tracking with Langfuse Bug Fix
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
            
        try:
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Router JSON parse error: {e}. Defaulting to RAG.")
            return {"intent": "rag", "response": ""}

    @observe(as_type="generation", name="agent_execution")
    def ask(self, user_query: str, gradio_history: list = None) -> str:
        """Main entry point for user queries."""
        if gradio_history is None:
            gradio_history = []
            
        formatted_history = self._format_history_for_gemini(gradio_history)
        
        # 1. Routing Decision
        route_decision = self.route_query(user_query, formatted_history)
        
        if route_decision.get("intent") == "chat":
            logger.info("Router: Handled as casual chat.")
            return route_decision.get("response", "Merhaba! Size nasıl yardımcı olabilirim?")
            
        # 2. RAG Flow Starts
        logger.info("Router: Proceeding with Vector Retrieval...")
        initial_chunks = self.retriever.retrieve(query=user_query, top_k=5)
        
        context_str = "--- RETRIEVED CONTEXT ---\n"
        for chunk in initial_chunks:
            context_str += f"[Source: {chunk['source']} | Strategy: {chunk['strategy']}]\n{chunk['text']}\n\n"
            
        full_prompt = f"USER QUERY: {user_query}\n\n{context_str}"
        
        # 3. Agent Execution with Tools and History
        chat = self.agent_model.start_chat(history=formatted_history)
        logger.info("Agent: Generating response (and calling tools if necessary)...")
        response = chat.send_message(full_prompt)
        
        # Cost tracking with Langfuse Bug Fix
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
            
        return response.text