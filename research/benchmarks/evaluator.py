import json
import time
import os
import pandas as pd
import yaml
import google.generativeai as genai
from typing_extensions import TypedDict
from dotenv import load_dotenv

from src.config.settings import settings
from src.rag.agent import KapadokyaAgent  # Integrating the actual Live Agent!

load_dotenv()
genai.configure(api_key=settings.gemini_api_key)

# --- Define Enforced JSON Schema for the Judge ---
class JudgeScore(TypedDict):
    score: int
    critique: str

def run_evaluation(ground_truth_path: str, output_dir: str):
    """
    Runs the LLM-as-a-Judge evaluation pipeline across all chunking strategies.
    It calls the actual KapadokyaAgent for each strategy and evaluates the response.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the LLM Judge prompt
    try:
        with open("src/config/benchmark_prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        judge_prompt = prompts["llm_judge"]
    except FileNotFoundError:
        print("Error: benchmark_prompts.yaml not found. Please ensure it exists in src/config/")
        return
        
    # 2. Initialize the Judge Model (Gemini 2.5 Pro for high reasoning)
    judge_model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=judge_prompt
    )

    # 3. Load the synthetic ground truth dataset
    try:
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            qa_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}.")
        print("Please run generate_synthetic_qa.py first.")
        return

    # 4. Define the 5 strategies (Mapping strategy name to exact Qdrant collection name)
    # Adjust the collection names here if your CLI logic names them differently
    strategies = {
        "naive": "kapadokya_baseline_naive", 
        "semantic": "kapadokya_baseline_semantic", 
        "layout": "kapadokya_baseline_unstructured", 
        "mega": "kapadokya_gemini_mega", 
        "parent_child": "kapadokya_gemini_parent_child"
    }

    results = []

    print(f"Starting evaluation for {len(qa_dataset)} questions across {len(strategies)} strategies...")

    for strategy_name, collection_name in strategies.items():
        print(f"\n==================================================")
        print(f"--- Testing Strategy: {strategy_name.upper()} ---")
        print(f"--- Collection: {collection_name} ---")
        print(f"==================================================")
        
        # Initialize the actual RAG Agent for this specific strategy
        try:
            agent = KapadokyaAgent(collection_name=collection_name, strategy=strategy_name)
        except Exception as e:
            print(f"Failed to initialize agent for {strategy_name}: {e}. Skipping...")
            continue
        
        for index, qa_pair in enumerate(qa_dataset):
            question = qa_pair["question"]
            ground_truth = qa_pair["ground_truth"]
            difficulty = qa_pair.get("difficulty", "Unknown")
            
            # Create a unique session ID for observability in Langfuse
            session_id = f"benchmark_{strategy_name}_q{index}"
            
            # --- STEP A: Ask the Live RAG System & Measure Latency ---
            start_time = time.time()
            try:
                # Call the actual agent we built
                actual_answer = agent.ask(user_query=question, session_id=session_id)
            except Exception as e:
                actual_answer = f"ERROR: Agent execution failed - {str(e)}"
            
            latency = round(time.time() - start_time, 2)
            
            # --- STEP B: Prepare the payload for the Judge ---
            evaluation_payload = f"""
            Question: {question}
            Ground Truth: {ground_truth}
            Actual Answer from RAG: {actual_answer}
            
            Evaluate the 'Actual Answer' based on the 'Ground Truth'.
            """
            
            # --- STEP C: Call the Judge (with enforced schema) ---
            try:
                response = judge_model.generate_content(
                    evaluation_payload,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=list[JudgeScore], # Strictly enforce the TypedDict array
                        temperature=0.0 # Zero temperature for strict, deterministic judging
                    )
                )
                
                # Parse the enforced JSON output
                judge_result = json.loads(response.text)[0] # Extract the first object from the array
                score = judge_result.get("score", 0)
                critique = judge_result.get("critique", "No critique provided.")
                judge_tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
                
            except Exception as e:
                print(f"Judge failed for question {index}: {e}")
                score = 0
                critique = f"JUDGE ERROR: {str(e)}"
                judge_tokens = 0

            # --- STEP D: Log the result ---
            results.append({
                "Strategy": strategy_name,
                "Collection": collection_name,
                "Question": question,
                "Difficulty": difficulty,
                "Ground_Truth": ground_truth,
                "RAG_Answer": actual_answer,
                "Score": score,
                "Latency_sec": latency,
                "Critique": critique,
                "Judge_Tokens": judge_tokens
            })
            
            print(f"[{strategy_name.upper()}] Q{index+1}/{len(qa_dataset)} - Score: {score}/5 | Latency: {latency}s")

    # 5. Save final results to a CSV report
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print(f"\nEvaluation complete! Detailed report saved to: {output_file}")
    
    # 6. Print a quick summary table
    summary = df.groupby('Strategy').agg(
        Average_Score=('Score', 'mean'),
        Average_Latency=('Latency_sec', 'mean')
    ).reset_index()
    
    # Sort by highest score first
    summary = summary.sort_values(by='Average_Score', ascending=False)
    
    print("\n--- FINAL BENCHMARK SUMMARY ---")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    GT_PATH = "data/evaluation_data/ground_truth.json"
    RESULTS_DIR = "research/benchmarks/results"
    
    run_evaluation(GT_PATH, RESULTS_DIR)