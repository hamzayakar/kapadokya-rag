import gradio as gr
from src.rag.agent import KapadokyaAgent

# We do NOT initialize agents globally to prevent startup crashes if Qdrant collections are missing.
# Instead, we store their configurations in a dictionary.
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
    def chat_interface(user_message, history):
        try:
            # LAZY INITIALIZATION: Initialize the agent only when the user asks the first question.
            # This prevents Hugging Face deployment timeouts and gracefully handles missing collections.
            if strategy_name not in loaded_agents:
                print(f"[{strategy_name}] Triggered for the first time. Connecting to Qdrant and Langfuse...")
                loaded_agents[strategy_name] = KapadokyaAgent(
                    collection_name=config["collection"], 
                    strategy=config["strategy"]
                )
            
            # The agent is ready in memory, route the query.
            return loaded_agents[strategy_name].ask(user_query=user_message, gradio_history=history)
        except Exception as e:
            # Graceful degradation: Show the error in the chat UI instead of crashing the whole app.
            return f"System error occurred. Please check logs: {str(e)}"
    return chat_interface

with gr.Blocks(theme="soft", title="Kapadokya RAG Benchmark") as demo:
    gr.Markdown("# Kapadokya University RAG Benchmark & Testing Arena")
    gr.Markdown(
        "Compare different document chunking strategies side-by-side. "
        "Each tab is connected to a different Qdrant collection and maintains its own chat history. "
        "Use the trash can icon to clear the chat."
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