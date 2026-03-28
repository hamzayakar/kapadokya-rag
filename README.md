# Agentic RAG Pipeline for Institutional Documents

An advanced, research-oriented Retrieval-Augmented Generation (RAG) system built for Kapadokya University's official regulations, directives, and external legal references. 

## Project Overview
This project goes beyond standard semantic search by implementing a robust, cross-modal ingestion pipeline and an agentic retrieval architecture. It is designed to handle complex inter-document references and legal hierarchies found in 69 official university documents.

## Key Features
- **Multimodal Ingestion Pipeline:** Utilizing Gemini's native PDF understanding (`mime_type=pdf`) for superior document parsing compared to traditional OCR/text splitters.
- **Advanced Chunking Strategies:** Comparing Mega-Chunking vs. Tool-based Parent-Child Chunking.
- **Agentic Retrieval:** Dynamic traversal of document cross-references (e.g., fetching External Law 2587 when cited by an internal directive) via custom tool calling.
- **Cloud-Native Architecture:** Stateless Gradio UI deployed alongside a Qdrant Cloud vector database.
- **Observability:** Full LLM trace, cost, and prompt management via Langfuse.

## Directory Structure
- `data/`: Local storage for raw PDFs, processed chunks, and evaluation datasets (ignored by version control).
- `src/`: Core application logic (ingestion, RAG agents, tools).
- `research/`: Benchmarking scripts, Jupyter notebooks, and evaluation reports for academic output.

## Stack
- **LLM:** Google Gemini Series
- **Embedding:** `text-embedding-004`
- **Vector DB:** Qdrant
- **UI:** Gradio
- **Observability:** Langfuse