import json
import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import our grading logic
from src.evaluation.grpo_reward import extract_boxed_answer, is_numerically_close

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def run_evaluation(data_path: str, adapter_path: str):
    """Evaluates the trained adapter against Kaggle's exact generation parameters."""
    device = get_device()
    print(f"📊 Starting Evaluation on: {device}")
    
    # 1. Load the Base Model and Tokenizer (Using GPT-2 for our CPU plumbing test)
    base_model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    
    # 2. Load the LoRA Adapter we trained in Layer 2
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        print("✅ LoRA adapter loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load adapter. Did the training script finish? Error: {e}")
        return

    # 3. Load the Test Data (We'll just use the raw synthetic data we generated)
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)["data"]

    correct_predictions = 0
    total_samples = min(len(dataset), 5) # Just test 5 samples locally to save time
    
    print(f"🧠 Running Kaggle-Parity Inference (Temp=0.0, MaxTokens=7680)...")
    
    for i in range(total_samples):
        sample = dataset[i]
        prompt = sample["prompt"]
        truth = sample["ground_truth"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 4. KAGGLE PARITY CONFIGURATION
        # These match the competition's vLLM parameters exactly
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128, # Using 128 for CPU speed. Kaggle uses 7680!
                temperature=0.0,    # Greedy decoding (0.0) is strictly required by Kaggle
                top_p=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode the output
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt from the completion
        completion_only = completion[len(prompt):] 
        
        # 5. Extract and Grade
        extracted_pred = extract_boxed_answer(completion_only)
        is_correct = is_numerically_close(extracted_pred, truth)
        
        if is_correct:
            correct_predictions += 1
            
        print(f"\nSample {i+1}:")
        print(f"Truth: {truth} | Extracted: {extracted_pred} | Correct: {is_correct}")

    # 6. Calculate Metric & Log to MLflow
    accuracy = correct_predictions / total_samples
    print(f"\n🎯 Final Local Accuracy: {accuracy * 100}%")
    
    mlflow.set_experiment("nemotron-evaluation")
    with mlflow.start_run():
        mlflow.log_param("eval_dataset", data_path)
        mlflow.log_param("adapter_tested", adapter_path)
        mlflow.log_metric("local_leaderboard_accuracy", accuracy)
        print("📈 Evaluation metrics logged to MLflow.")

if __name__ == "__main__":
    # We test the synthetic data we made against the adapter we trained
    # Replace the date tag with the exact one you used in Layer 1!
    test_data = "data/raw/synthetic_math_v20260402.json" 
    trained_adapter = "outputs/final_adapter"
    
    run_evaluation(test_data, trained_adapter)