import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Import the reward function we wrote earlier!
from src.evaluation.grpo_reward import compute_grpo_rewards

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def test_grpo_pipeline():
    device = get_device()
    print(f"🚀 Initializing GRPO Pipeline on: {device}")
    
    # 1. Load a microscopic model for local CPU testing
    # In production on GCP, this will be "nvidia/Nemotron-3-Nano-30B"
    model_name = "gpt2" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # 2. Configure the LoRA Adapter (Strict Kaggle Rule: Rank <= 32)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["c_attn"], # gpt2 specific. For Nemotron use ["q_proj", "v_proj", "k_proj", "o_proj"]
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 3. Create a tiny dummy dataset with formatting
    # The 'prompt' is what the model sees. The 'ground_truth' is passed to our reward function.
    dummy_data = {
        "prompt": ["What is 2 + 2? Please format as \\boxed{answer}. "],
        "ground_truth": ["4"]
    }
    dataset = Dataset.from_dict(dummy_data)
    
    # 4. Set up the GRPO Training config
    training_args = GRPOConfig(
        output_dir="outputs/grpo_test",
        learning_rate=2e-5,
        logging_steps=1,
        max_steps=2, # Just run 2 steps to prove it works
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        report_to="mlflow", # Auto-logs to our MLflow server!
        use_vllm=False, # Set to True later when on GCP Blackwells
        max_prompt_length=128,
        max_completion_length=128
    )
    
    # 5. Initialize the Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_grpo_rewards], # Hooking in our custom grading logic!
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        tokenizer=tokenizer
    )
    
    print("🧠 Starting 2-step plumbing test training loop...")
    trainer.train()
    
    # 6. Save the Kaggle-compliant adapter
    print("💾 Saving LoRA adapter...")
    trainer.model.save_pretrained("outputs/final_adapter")
    print("✅ Pipeline test complete. adapter_config.json generated in outputs/final_adapter")

if __name__ == "__main__":
    # Ensure MLflow tracks this
    mlflow.set_experiment("nemotron-grpo-plumbing")
    test_grpo_pipeline()