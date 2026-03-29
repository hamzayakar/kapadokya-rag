import gradio as gr
from src.rag.agent import KapadokyaAgent

agents = {
    "Gemini Mega-Chunking": KapadokyaAgent(collection_name="kapadokya_gemini_mega", strategy="mega"),
    "Gemini Parent-Child": KapadokyaAgent(collection_name="kapadokya_gemini_parent_child", strategy="parent_child"),
    "Baseline: Langchain Naive": KapadokyaAgent(collection_name="kapadokya_baseline_naive", strategy="naive"),
    "Baseline: Langchain Semantic": KapadokyaAgent(collection_name="kapadokya_baseline_semantic", strategy="semantic"),
    "Baseline: Unstructured": KapadokyaAgent(collection_name="kapadokya_baseline_unstructured", strategy="unstructured")
}

def create_chat_fn(agent_instance):
    def chat_interface(user_message, history):
        try:
            return agent_instance.ask(user_query=user_message, gradio_history=history)
        except Exception as e:
            return f"System error occurred. Please check logs: {str(e)}"
    return chat_interface

with gr.Blocks(theme="soft", title="Kapadokya RAG Benchmark") as demo:
    gr.Markdown("# Kapadokya University RAG Benchmark & Testing Arena")
    gr.Markdown(
        "Compare different document chunking strategies side-by-side. "
        "Each tab is connected to a different Qdrant collection and maintains its own chat history. "
        "Use the trash can icon to clear the chat."
    )

    for strategy_name, agent in agents.items():
        with gr.Tab(strategy_name):
            gr.ChatInterface(
                fn=create_chat_fn(agent),
                type="messages",
                examples=[
                    "Öğrenci disiplin suçlarında hangi kanun maddeleri uygulanır?",
                    "Çift anadal başvurusu için not ortalaması (GPA) kaç olmalıdır?",
                    "Merhaba, nasılsın?"
                ]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)