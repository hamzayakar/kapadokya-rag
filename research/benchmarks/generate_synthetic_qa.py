import json
import os
import yaml
import google.generativeai as genai
from typing_extensions import TypedDict
from dotenv import load_dotenv

from src.tools.reference_fetcher import fetch_reference_context
from src.config.settings import settings

load_dotenv()
genai.configure(api_key=settings.gemini_api_key)

# Define the expected output schema for the synthetic questions to prevent parsing errors
class QAPair(TypedDict):
    question: str
    ground_truth: str
    difficulty: str
    reasoning: str

def generate_synthetic_dataset(chunk_directory: str, output_file: str):
    """
    Reads processed chunks, allows the Agent to use tools for multi-hop reasoning,
    and generates a synthetic QA dataset.
    """
    # 1. Load prompts
    with open("src/config/benchmark_prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    
    # 2. Dynamically determine the collection name from the directory path
    # Example: 'data/processed_chunks/gemini_mega' -> 'kapadokya_gemini_mega'
    strategy_folder = os.path.basename(os.path.normpath(chunk_directory))
    collection_name = f"kapadokya_{strategy_folder}"
    
    # 3. FIX: Create a wrapper to inject the 'collection_name' into the tool
    # The LLM only knows target_filename and query, it doesn't know the DB architecture!
    def reference_fetch_wrapper(target_filename: str, query: str) -> str:
        return fetch_reference_context(collection_name, target_filename, query)

    # 4. Initialize the generator agent with the injected wrapper
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=prompts["synthetic_generator"],
        tools=[reference_fetch_wrapper] # Tool is now properly wrapped!
    )

    qa_dataset = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Walk through the directory and process JSON chunks
    for filename in os.listdir(chunk_directory):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(chunk_directory, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        for chunk in chunks:
            # We only generate questions for chunks that have enough substance
            if len(chunk.get("text", "")) < 100:
                continue

            print(f"Generating QA for chunk in {chunk['metadata'].get('source')}...")
            
            prompt_payload = f"Chunk Text: {chunk['text']}\nMetadata: {json.dumps(chunk['metadata'])}"
            
            try:
                # Enable automatic function calling so the agent can use the tool if it sees a reference
                response = model.generate_content(
                    prompt_payload,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=list[QAPair],
                        temperature=0.7 # A bit of creativity for generating diverse, realistic questions
                    ),
                    tool_config={"function_calling_config": {"mode": "AUTO"}}
                )
                
                generated_pairs = json.loads(response.text)
                
                # Attach source metadata to the generated questions for traceability
                for pair in generated_pairs:
                    pair["source_chunk_metadata"] = chunk["metadata"]
                    qa_dataset.append(pair)
                    
            except Exception as e:
                print(f"Failed to generate QA for a chunk: {e}")

    # Save the final Ground Truth dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully generated {len(qa_dataset)} QA pairs and saved to {output_file}")

if __name__ == "__main__":
    # Assuming we generate synthetic QA based on the most logically coherent chunks (mega-chunking)
    INPUT_DIR = "data/processed_chunks/gemini_mega"
    OUTPUT_FILE = "data/evaluation_data/ground_truth.json"
    
    generate_synthetic_dataset(INPUT_DIR, OUTPUT_FILE)