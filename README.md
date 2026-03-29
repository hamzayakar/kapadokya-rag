---
title: Kapadokya RAG
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: Agentic RAG System for Kapadokya University
---

# Agentic RAG Pipeline for Institutional Documents

An advanced, research-oriented Retrieval-Augmented Generation (RAG) system built for Kapadokya University's official regulations, directives, and external legal references. 

## Project Overview
This project goes beyond standard semantic search by implementing a robust, cross-modal ingestion pipeline and an agentic retrieval architecture. It is designed to handle complex inter-document references and legal hierarchies found in 69+ official university documents.

## Key Features
- **Multimodal Ingestion Pipeline:** Utilizing Gemini's native PDF understanding (`mime_type=pdf`) for superior document parsing compared to traditional OCR/text splitters.
- **Advanced Chunking Strategies:** Comparing Mega-Chunking vs. Tool-based Parent-Child Chunking.
- **Agentic Retrieval & Tools:** - `parent_fetcher`: Dynamically fetches the entirety of a specific document if the initial retrieved chunks lack context.
  - `reference_fetcher`: Automatically traverses cross-references (e.g., fetching External Law 2547 when cited by an internal directive) via custom tool calling.
- **Router-Agent Architecture:** A fail-fast, dual-model setup where a smaller model (Gemini 1.5 Flash) routes queries (Chat vs. RAG) and a larger model (Gemini 2.5 Pro) executes the RAG and tool calls.
- **Cloud-Native & Production Ready:** Stateless Gradio UI deployed alongside a Qdrant Cloud vector database, complete with collision control during data ingestion.
- **Observability:** Full LLM trace, token-based cost estimation, and dynamic prompt management via Langfuse.

## Directory Structure
- `data/`: Local storage for raw PDFs, processed chunks, and evaluation datasets (ignored by version control).
- `src/`: Core application logic (ingestion, RAG agents, tools).
- `research/`: Benchmarking scripts, Jupyter notebooks, and evaluation reports for academic output.
- `app.py`: The main Gradio web interface.

## Stack
- **LLM:** Google Gemini Series (1.5 Flash & 2.5 Pro)
- **Embedding:** `text-embedding-004`
- **Vector DB:** Qdrant Cloud
- **UI:** Gradio 5
- **Observability:** Langfuse

## Ingestion Pipeline Usage (CLI)

You can process raw PDFs into structured JSON chunks and push them to Qdrant using the built-in enterprise CLI tool.

**1. Extract (PDF to JSON)**
Extracts text using the chosen engine and strategy. Includes collision control to prevent overwriting existing JSONs.
```bash
python -m src.ingestion.cli --action extract --method gemini --strategy mega --folder kapadokya
```

**2. Push (JSON to Qdrant)**
Embeds the extracted chunks and upserts them to Qdrant. Automatically skips documents that are already indexed unless `--force` is used.
```bash
python -m src.ingestion.cli --action push --method gemini --strategy mega --folder kapadokya
```

**3. Delete (Remove from Qdrant)**
Safely deletes all vectors associated with a specific document from the specified collection.
```bash
python -m src.ingestion.cli --action delete --method gemini --strategy mega --folder kapadokya --file yonetmelik.pdf
```

## Running the Web UI

To start the chat interface locally:
```bash
python app.py
```