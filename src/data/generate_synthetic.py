import json
import os
import random
from datetime import datetime

def call_llm_api(topic: str) -> dict:
    """
    In production, this calls Claude 3.5 Sonnet or GPT-4o to generate a complex math problem.
    For our local plumbing test, it returns a deterministic mock payload.
    """
    # Mocking a realistic Kaggle reasoning dataset format
    problem_id = f"SYNTH-{random.randint(1000, 9999)}"
    return {
        "id": problem_id,
        "prompt": f"Solve this advanced problem regarding {topic}. Show your step-by-step reasoning and conclude with your final answer formatted as \\boxed{{answer}}.",
        "ground_truth": str(random.randint(10, 100))
    }

def build_synthetic_dataset(output_path: str, num_samples: int = 5):
    """Generates a dataset and saves it to the raw data layer."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    topics = ["linear algebra", "calculus", "probability", "combinatorics", "number theory"]
    print(f"🏭 Generating {num_samples} synthetic reasoning samples...")
    
    dataset = {"data": []}
    for i in range(num_samples):
        topic = random.choice(topics)
        sample = call_llm_api(topic)
        dataset["data"].append(sample)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"✅ Saved {num_samples} samples to {output_path}")

if __name__ == "__main__":
    # Generate a versioned file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = f"data\\raw\\synthetic_math_v{timestamp}.json"
    build_synthetic_dataset(output_file, num_samples=10)