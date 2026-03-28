import argparse
import os
import json
from pathlib import Path

from src.ingestion.chunkers.gemini_chunker import GeminiChunker
from src.ingestion.chunkers.langchain_chunker import LangchainChunker
from src.ingestion.chunkers.unstructured_chunker import UnstructuredChunker

def save_chunks(chunks: list, output_dir: str, filename: str):
    """Chunkları JSON olarak diske kaydeder."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{Path(filename).stem}_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(chunks)} chunks to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Kapadokya RAG Ingestion Pipeline")
    parser.add_argument("--method", type=str, required=True, choices=["gemini", "langchain", "unstructured"], help="Chunking engine to use.")
    parser.add_argument("--strategy", type=str, default="mega", choices=["mega", "parent_child", "naive", "semantic"], help="Specific strategy for the chosen engine.")
    parser.add_argument("--input_dir", type=str, default="data/raw_docs/kapadokya", help="Directory containing raw PDFs.")
    
    args = parser.parse_args()

    # Choose Chunker
    if args.method == "gemini":
        chunker = GeminiChunker(strategy=args.strategy)
        out_folder = f"gemini_{args.strategy}"
    elif args.method == "langchain":
        chunker = LangchainChunker(strategy=args.strategy)
        out_folder = f"baseline_{args.strategy}"
    elif args.method == "unstructured":
        chunker = UnstructuredChunker()
        out_folder = "baseline_unstructured"

    output_dir = os.path.join("data/processed_chunks", out_folder)

    # Process PDFs
    pdf_files = [f for f in os.listdir(args.input_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs. Starting ingestion using {args.method} ({args.strategy})...")

    for pdf in pdf_files:
        file_path = os.path.join(args.input_dir, pdf)
        print(f"Processing: {pdf}")
        chunks = chunker.chunk_document(file_path)
        if chunks:
            save_chunks(chunks, output_dir, pdf)

if __name__ == "__main__":
    main()