import torch
import mlflow
import os

def get_device():
    """Dynamically routes workloads based on available hardware."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("🐢 No GPU detected. Falling back to CPU for local testing.")
        return torch.device("cpu")

def train_lora(data_version: str, rank: int, learning_rate: float):
    device = get_device()
    mlflow.set_experiment("nemotron-reasoning-lora")
    
    with mlflow.start_run():
        # Log our hyperparameters
        mlflow.log_params({
            "data_version": data_version,
            "lora_rank": rank,
            "learning_rate": learning_rate,
            "device": str(device)
        })
        
        print(f"Initializing pipeline on {device}...")
        print("Simulating adapter generation for local ops validation...")
        
        # Create dummy adapter config to satisfy Kaggle requirements
        adapter_dir = f"outputs\\adapter_rank_{rank}"
        os.makedirs(adapter_dir, exist_ok=True)
        config_path = f"{adapter_dir}\\adapter_config.json"
        
        with open(config_path, "w") as f:
            f.write(f'{{"r": {rank}, "target_modules": ["q_proj", "v_proj"]}}')
            
        # Log the artifact to MLflow
        mlflow.log_artifact(config_path, artifact_path="adapter")
        print("✅ Run logged successfully to MLflow.")

if __name__ == "__main__":
    train_lora(data_version="v1.0", rank=32, learning_rate=2e-5)