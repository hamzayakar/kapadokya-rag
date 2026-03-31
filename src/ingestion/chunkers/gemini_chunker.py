import json
import yaml
import google.generativeai as genai
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langfuse.decorators import observe, langfuse_context

from src.ingestion.chunkers.base_chunker import BaseChunker
from src.config.settings import settings

genai.configure(api_key=settings.gemini_api_key)

# --- Define Enforced JSON Schemas ---
class MegaChunkSchema(TypedDict):
    section_title: str
    text: str
    raw_references: list[str]

class ParentChildChunkSchema(TypedDict):
    parent_summary: str
    child_chunks: list[str]
    raw_references: list[str]

class GeminiChunker(BaseChunker):
    def __init__(self, strategy: str = "mega"):
        super().__init__()
        self.strategy = strategy
        self.model_name = "gemini-2.5-pro"
        self.model = genai.GenerativeModel(self.model_name)
        
        with open("src/config/ingestion_prompts.yaml", "r", encoding="utf-8") as f:
            ingestion_prompts = yaml.safe_load(f)
            
        base_prompt = ingestion_prompts["gemini_base"]
        focus_prompt = ingestion_prompts[f"{strategy}_focus"]
        self.system_prompt = f"{base_prompt}\n\n{focus_prompt}"

    @observe(as_type="generation")
    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        langfuse_context.update_current_observation(name=f"gemini_{self.strategy}_chunking")
        uploaded_file = genai.upload_file(path=file_path, mime_type="application/pdf")
        
        # Select the target schema based on the strategy to guarantee determinism
        target_schema = list[MegaChunkSchema] if self.strategy == "mega" else list[ParentChildChunkSchema]
        
        try:
            response = self.model.generate_content(
                [uploaded_file, self.system_prompt],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=target_schema,
                    temperature=0.1 # Low temperature for factual extraction
                )
            )
            
            if response.usage_metadata:
                langfuse_context.update_current_observation(
                    usage={
                        "input": response.usage_metadata.prompt_token_count,
                        "output": response.usage_metadata.candidates_token_count
                    },
                    model=self.model_name
                )
                
            raw_chunks = json.loads(response.text)
            formatted_chunks = []
            base_meta = self.get_file_metadata(file_path)
            
            for i, chunk in enumerate(raw_chunks):
                meta = base_meta.copy()
                meta["strategy"] = f"gemini_{self.strategy}"
                meta["chunk_index"] = i
                # Save as raw_references for the intermediate Linker phase
                meta["raw_references"] = chunk.get("raw_references", [])
                
                if self.strategy == "mega":
                    meta["section_title"] = chunk.get("section_title", "")
                    formatted_chunks.append({"text": chunk.get("text", ""), "metadata": meta})
                    
                elif self.strategy == "parent_child":
                    meta["parent_summary"] = chunk.get("parent_summary", "")
                    for j, child_text in enumerate(chunk.get("child_chunks", [])):
                        child_meta = meta.copy()
                        child_meta["child_index"] = j
                        formatted_chunks.append({"text": child_text, "metadata": child_meta})
                        
            return formatted_chunks
            
        except Exception as e:
            raise RuntimeError(f"Gemini API or JSON parsing error failed: {e}")
        finally:
            uploaded_file.delete()