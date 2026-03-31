import os
import json
import logging
import google.generativeai as genai
from typing_extensions import TypedDict
from dotenv import load_dotenv

from src.config.settings import settings

load_dotenv()
genai.configure(api_key=settings.gemini_api_key)
logger = logging.getLogger(__name__)

# Guaranteed schema for Gemini's response to ensure we get a consistent mapping format
class CitationMapping(TypedDict):
    mapping: dict[str, str]

class CitationLinker:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Collect all valid PDF filenames from the raw_docs folders to use as the authoritative list for matching
        self.valid_filenames = []
        raw_base = os.path.join(os.getcwd(), "data", "raw_docs")
        for folder in ["kapadokya", "external"]:
            folder_path = os.path.join(raw_base, folder)
            if os.path.exists(folder_path):
                self.valid_filenames.extend(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
                )

    def resolve_citations(self, chunk_file_path: str):
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # 1. Collect all unique fuzzy citations from the "raw_references" fields across all chunks
        unique_raw_refs = set()
        for chunk in chunks:
            raw_refs = chunk.get("metadata", {}).get("raw_references", [])
            unique_raw_refs.update(raw_refs)

        if not unique_raw_refs:
            logger.info(f"No citations found in {os.path.basename(chunk_file_path)}.")
            return # Atıf yoksa işlemi atla

        # 2. Assign the fuzzy citations and valid filenames to the Gemini model for strict matching
        prompt = f"""
        You are an expert strict citation resolver.
        
        I have a list of fuzzy citation names extracted from a document:
        {list(unique_raw_refs)}
        
        I have a strictly defined list of actual PDF filenames currently in my database:
        {self.valid_filenames}
        
        Your task:
        Create a mapping for each fuzzy citation to the EXACT matching actual filename.
        If a fuzzy citation does not clearly match any of the actual filenames, map it to an empty string "".
        
        Return exactly a JSON object matching the requested schema.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CitationMapping,
                    temperature=0.0 # Sıfır halüsinasyon, kesin eşleşme
                )
            )
            
            mapping_result = json.loads(response.text).get("mapping", {})
            
            # 3. Write the mapping results back into each chunk's metadata under a new "references" key, replacing fuzzy citations with exact matches
            mapped_count = 0
            for chunk in chunks:
                raw_refs = chunk.get("metadata", {}).get("raw_references", [])
                actual_refs = []
                for fuzzy_ref in raw_refs:
                    exact_match = mapping_result.get(fuzzy_ref, "")
                    if exact_match:
                        actual_refs.append(exact_match)
                        mapped_count += 1
                
                # Create the "references" key with the list of exact matches (or empty if no match)
                chunk["metadata"]["references"] = actual_refs

            # 4. Overwrite the original chunk file with the updated metadata containing the resolved citations
            with open(chunk_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
                
            logger.info(f"Successfully linked {mapped_count} citations in {os.path.basename(chunk_file_path)}")

        except Exception as e:
            logger.error(f"Failed to resolve citations for {chunk_file_path}: {e}")