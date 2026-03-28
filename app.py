import gradio as gr
from src.rag.agent import KapadokyaAgent

# Initialize the agent with the default collection
agent = KapadokyaAgent(collection_name="kapadokya_gemini_mega")

def chat_interface(user_message, history):
    """
    Passes the user message and conversation history to the agent.
    Gradio 5 history format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    try:
        response = agent.ask(user_query=user_message, gradio_history=history)
        return response
    except Exception as e:
        return f"System error occurred. Please check logs: {str(e)}"

# Gradio Chatbot UI Setup
demo = gr.ChatInterface(
    fn=chat_interface,
    type="messages",
    title="Kapadokya University RAG Assistant",
    description="Ask questions about university regulations, directives, and laws powered by Agentic RAG architecture.",
    theme="soft",
    examples=[
        "Which laws apply to student disciplinary offenses?",
        "What happens if I miss the course registration deadline?",
        "Hello, how are you?"
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)