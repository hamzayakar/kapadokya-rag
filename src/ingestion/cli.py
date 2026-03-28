import argparse
import os
import json
import logging
from pathlib import Path

from src.ingestion.chunkers.gemini_chunker import GeminiChunker
from src.ingestion.chunkers.langchain_chunker import LangchainChunker
from src.ingestion.chunkers.unstructured_chunker import UnstructuredChunker
from src.ingestion.indexer import Indexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_out_folder_name(method: str, strategy: str) -> str:
    """Returns the standardized folder name for processed chunks."""
    if method == "gemini":
        return f"gemini_{strategy}"
    elif method == "langchain":
        return f"baseline_{strategy}"
    elif method == "unstructured":
        return "baseline_unstructured"
    return "unknown_method"

def save_chunks(chunks: list, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{Path(filename).stem}_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved {len(chunks)} chunks to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Kapadokya RAG Enterprise Ingestion CLI")
    parser.add_argument("--action", type=str, required=True, choices=["extract", "push", "delete"],
                        help="'extract' (PDF to JSON), 'push' (JSON to Qdrant), or 'delete' (Remove from Qdrant)")
    parser.add_argument("--method", type=str, choices=["gemini", "langchain", "unstructured"],
                        help="Chunking engine. Required for extract/push.")
    parser.add_argument("--strategy", type=str, default="mega", choices=["mega", "parent_child", "naive", "semantic", "layout"],
                        help="Specific strategy for the chosen engine.")
    parser.add_argument("--folder", type=str, default="kapadokya", choices=["kapadokya", "external"],
                        help="Target document folder (kapadokya or external).")
    parser.add_argument("--file", type=str, required=False,
                        help="(Optional) Specific filename (e.g., yonetmelik.pdf or yonetmelik_chunks.json).")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction (overwrite JSON) or re-indexing (delete existing in Qdrant and push).")

    args = parser.parse_args()

    # --- Directory Setup ---
    base_raw_dir = os.path.join(os.getcwd(), "data", "raw_docs", args.folder)
    
    if args.method:
        out_folder_name = get_out_folder_name(args.method, args.strategy)
        processed_dir = os.path.join(os.getcwd(), "data", "processed_chunks", out_folder_name)
        # Unique Qdrant collection name for each folder + strategy combo (e.g., kapadokya_gemini_mega)
        collection_name = f"{args.folder}_{out_folder_name}"
    
    # ==========================================
    # ACTION: EXTRACT (PDF -> JSON)
    # ==========================================
    if args.action == "extract":
        if not args.method:
            logger.error("--method is required for extraction.")
            exit(1)

        # Init appropriate chunker
        if args.method == "gemini":
            chunker = GeminiChunker(strategy=args.strategy)
        elif args.method == "langchain":
            chunker = LangchainChunker(strategy=args.strategy)
        elif args.method == "unstructured":
            chunker = UnstructuredChunker()

        # Find targets
        pdfs_to_process = []
        if args.file:
            target_path = os.path.join(base_raw_dir, args.file)
            if not target_path.lower().endswith(".pdf"):
                logger.error(f"Extract requires a .pdf file. Got: {args.file}")
                exit(1)
            if os.path.exists(target_path):
                pdfs_to_process.append(target_path)
            else:
                logger.error(f"File not found: {target_path}")
                exit(1)
        else:
            if not os.path.exists(base_raw_dir):
                logger.error(f"Raw directory not found: {base_raw_dir}")
                exit(1)
            pdfs_to_process = [os.path.join(base_raw_dir, f) for f in os.listdir(base_raw_dir) if f.lower().endswith(".pdf")]

        if not pdfs_to_process:
            logger.warning("No PDF files found to process.")
            exit(0)

        for pdf_path in pdfs_to_process:
            filename = os.path.basename(pdf_path)
            json_name = f"{Path(filename).stem}_chunks.json"
            json_path = os.path.join(processed_dir, json_name)

            # Collision Control
            if not args.force and os.path.exists(json_path):
                logger.warning(f"Skipping {filename} - JSON already exists. Use --force to overwrite.")
                continue

            logger.info(f"Extracting: {filename} using {args.method} ({args.strategy})")
            chunks = chunker.chunk_document(pdf_path)
            if chunks:
                save_chunks(chunks, processed_dir, filename)

    # ==========================================
    # ACTION: PUSH (JSON -> QDRANT)
    # ==========================================
    elif args.action == "push":
        if not args.method:
            logger.error("--method is required for push to determine collection.")
            exit(1)

        indexer = Indexer(collection_name=collection_name)
        jsons_to_process = []

        if args.file:
            target_file = args.file
            if target_file.lower().endswith(".pdf"):
                target_file = f"{Path(target_file).stem}_chunks.json"
            elif not target_file.lower().endswith(".json"):
                target_file = f"{target_file}_chunks.json"

            target_path = os.path.join(processed_dir, target_file)
            if os.path.exists(target_path):
                jsons_to_process.append(target_path)
            else:
                logger.error(f"File not found: {target_path}")
                exit(1)
        else:
            if not os.path.exists(processed_dir):
                logger.error(f"Processed directory not found: {processed_dir}")
                exit(1)
            jsons_to_process = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.lower().endswith(".json")]

        for json_path in jsons_to_process:
            filename_json = os.path.basename(json_path)
            original_pdf_name = filename_json.replace("_chunks.json", ".pdf")

            # Qdrant Collision Control
            if indexer.has_document(original_pdf_name):
                if not args.force:
                    logger.warning(f"Skipping {original_pdf_name} - Vectors already in Qdrant. Use --force to re-index.")
                    continue
                else:
                    logger.info(f"Force flag detected. Deleting old vectors for {original_pdf_name}...")
                    indexer.delete_by_source(original_pdf_name)

            logger.info(f"Pushing {filename_json} to collection: {collection_name}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                indexer.index_chunks(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in {filename_json}. Skipping.")

    # ==========================================
    # ACTION: DELETE (QDRANT -> REMOVE VECTORS)
    # ==========================================
    elif args.action == "delete":
        if not args.method or not args.file:
            logger.error("Delete action requires --method (to find collection) and --file (to specify document).")
            exit(1)

        indexer = Indexer(collection_name=collection_name)
        
        target_source = args.file
        if target_source.lower().endswith(".json"):
            target_source = target_source.replace("_chunks.json", ".pdf")
        elif not target_source.lower().endswith(".pdf"):
            target_source = f"{target_source}.pdf"

        logger.info(f"Attempting to delete all chunks for '{target_source}' from '{collection_name}'...")
        success = indexer.delete_by_source(target_source)
        if success:
            logger.info("Deletion successful.")

if __name__ == "__main__":
    main()