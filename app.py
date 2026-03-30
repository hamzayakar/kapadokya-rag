import gradio as gr
import hashlib
from src.rag.agent import KapadokyaAgent

# We do NOT initialize agents globally to prevent startup crashes if Qdrant collections are missing.
agent_configs = {
    "Gemini Mega-Chunking": {"collection": "kapadokya_gemini_mega", "strategy": "mega"},
    "Gemini Parent-Child": {"collection": "kapadokya_gemini_parent_child", "strategy": "parent_child"},
    "Baseline: Langchain Naive": {"collection": "kapadokya_baseline_naive", "strategy": "naive"},
    "Baseline: Langchain Semantic": {"collection": "kapadokya_baseline_semantic", "strategy": "semantic"},
    "Baseline: Unstructured": {"collection": "kapadokya_baseline_unstructured", "strategy": "unstructured"}
}

# A cache dictionary to hold the initialized agents in memory.
loaded_agents = {}

def create_chat_fn(strategy_name, config):
    def chat_interface(user_message, history, request: gr.Request):
        try:
            # 1. Base session hash from browser
            browser_session = request.session_hash if request else "local"
            
            # 2. Identify the FIRST message of the current active chat cycle
            # If history is empty (new chat or cleared), the current message is the first.
            first_msg = history[0]["content"] if len(history) > 0 else user_message
            
            # 3. Create a unique Session ID: Browser + Tab Name + First Message
            raw_id = f"{browser_session}_{strategy_name}_{first_msg}"
            # Hash it to keep it clean and short for Langfuse
            session_id = f"{strategy_name}_{hashlib.md5(raw_id.encode()).hexdigest()[:8]}"

            # LAZY INITIALIZATION
            if strategy_name not in loaded_agents:
                print(f"[{strategy_name}] Triggered for the first time. Connecting to Qdrant...")
                loaded_agents[strategy_name] = KapadokyaAgent(
                    collection_name=config["collection"], 
                    strategy=config["strategy"]
                )
            
            # Route query with the bulletproof session_id
            return loaded_agents[strategy_name].ask(
                user_query=user_message, 
                gradio_history=history,
                session_id=session_id
            )
        except Exception as e:
            return f"System error occurred. Please check logs: {str(e)}"
    return chat_interface

with gr.Blocks(theme="soft", title="Kapadokya RAG Benchmark") as demo:
    gr.Markdown("# Kapadokya University RAG Benchmark & Testing Arena")
    gr.Markdown(
        "Compare different document chunking strategies side-by-side. "
        "Each tab is connected to a different Qdrant collection and maintains its own chat history. "
        "Use the trash can icon to clear the chat and start a new Langfuse session."
    )

    for strategy_name, config in agent_configs.items():
        with gr.Tab(strategy_name):
            gr.ChatInterface(
                fn=create_chat_fn(strategy_name, config),
                type="messages",
                examples=[
                    "Öğrenci disiplin suçlarında hangi kanun maddeleri uygulanır?",
                    "Çift anadal başvurusu için not ortalaması (GPA) kaç olmalıdır?",
                    "Merhaba, nasılsın?"
                ]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)