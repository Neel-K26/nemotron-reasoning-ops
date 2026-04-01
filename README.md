# 🚀 Nemotron Reasoning Ops: Production-Grade Kaggle Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?style=for-the-badge&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-Passing-brightgreen.svg?style=for-the-badge&logo=github-actions)
![Kaggle](https://img.shields.io/badge/Kaggle-Nemotron_Challenge-20BEFF.svg?style=for-the-badge&logo=kaggle)

An end-to-end, production-grade MLOps pipeline engineered for the **NVIDIA Nemotron Model Reasoning Challenge**. 

This repository moves beyond standard Kaggle notebooks by providing a robust, scalable, and containerized architecture. It is designed to generate synthetic data, track hyperparameter experiments, apply Group Relative Policy Optimization (GRPO), and automatically validate LoRA adapters for Kaggle deployment.



---

## 🏗️ Architecture: The 6-Layer Foundation

This system is built to seamlessly transition from local CPU/AMD prototyping to production-grade training on Google Cloud Platform (GCP) G4 instances powered by **NVIDIA RTX PRO 6000 Blackwell GPUs**.

1. **Layer 1: Immutable Data Versioning (DVC + GCS)**
   * Curated public math/science QA datasets and dynamically generated Synthetic Chain-of-Thought (CoT) data are versioned immutably via DVC. Every model run is tied to an exact data hash.
2. **Layer 2: Experiment Engine (MLflow)**
   * Fully automated tracking of datasets, seeds, LoRA ranks (strictly $\le 32$), learning rates, and adapter hashes. Includes a dynamic hardware router (CPU fallback for local ops).
3. **Layer 3: Kaggle-Parity Evaluation Harness**
   * A local testing harness that perfectly mirrors Kaggle's vLLM evaluation parameters (`temperature=0.0`, `max_new_tokens=7680`). Prevents leaderboard divergence by ensuring local accuracy reflects actual submission accuracy.
4. **Layer 4: Automated CI/CD Guardrails**
   * GitHub Actions automatically parse the `adapter_config.json`, mathematically validate that the LoRA rank meets the competition limit, and construct the `submission.zip` artifact.
5. **Layer 5: Monitoring & Overfitting Detection**
   * Tracks local evaluation delta vs. public leaderboard scores to monitor model generalization.
6. **Layer 6: Production Infrastructure (Docker)**
   * A containerized environment pulling the official `nvidia/cuda:12.1.1-devel-ubuntu22.04` base image, ensuring 100% hardware and dependency parity in the cloud.

---

## 📂 Project Structure

```text
nemotron-reasoning-ops/
├── .github/workflows/      # CI/CD validation and auto-packaging pipelines
├── data/                   # Data layer (strictly tracked by DVC, ignored by Git)
│   ├── raw/                # Synthetic CoT data and raw competition datasets
│   ├── processed/          # Tokenized, filtered, and formatted datasets
│   └── synthetic/          # LLM-generated math/logic problem templates
├── src/                    # Core pipeline logic
│   ├── data/               # Scripts for synthetic generation and preprocessing
│   ├── evaluation/         # GRPO reward shaping and vLLM parity evaluation
│   ├── models/             # LoRA adapters, Unsloth/TRL loops, and inference
│   └── utils/              # Helper functions (logging, hardware routing)
├── tests/                  # Pytest unit tests for reward functions and formatting
├── notebooks/              # Exploratory Data Analysis (EDA) and prize write-ups
├── outputs/                # Local staging for adapter_config.json and weights
├── Dockerfile              # NVIDIA CUDA 12.1 production container blueprint
├── requirements.txt        # Core ML and Ops dependencies
└── config.yaml             # Global pipeline hyperparameters




Gemini said
Here is an expanded, highly detailed, and portfolio-ready README.md. I have added more comprehensive setup instructions, deeper explanations of the math/reward logic, and a structured workflow section so anyone visiting your repository knows exactly how advanced this architecture is.

Copy the entire block below and paste it directly into your README.md file!

Markdown
# 🚀 Nemotron Reasoning Ops: Production-Grade Kaggle Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?style=for-the-badge&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-Passing-brightgreen.svg?style=for-the-badge&logo=github-actions)
![Kaggle](https://img.shields.io/badge/Kaggle-Nemotron_Challenge-20BEFF.svg?style=for-the-badge&logo=kaggle)

An end-to-end, production-grade MLOps pipeline engineered for the **NVIDIA Nemotron Model Reasoning Challenge**. 

This repository moves beyond standard Kaggle notebooks by providing a robust, scalable, and containerized architecture. It is designed to generate synthetic data, track hyperparameter experiments, apply Group Relative Policy Optimization (GRPO), and automatically validate LoRA adapters for Kaggle deployment.



---

## 🏗️ Architecture: The 6-Layer Foundation

This system is built to seamlessly transition from local CPU/AMD prototyping to production-grade training on Google Cloud Platform (GCP) G4 instances powered by **NVIDIA RTX PRO 6000 Blackwell GPUs**.

1. **Layer 1: Immutable Data Versioning (DVC + GCS)**
   * Curated public math/science QA datasets and dynamically generated Synthetic Chain-of-Thought (CoT) data are versioned immutably via DVC. Every model run is tied to an exact data hash.
2. **Layer 2: Experiment Engine (MLflow)**
   * Fully automated tracking of datasets, seeds, LoRA ranks (strictly $\le 32$), learning rates, and adapter hashes. Includes a dynamic hardware router (CPU fallback for local ops).
3. **Layer 3: Kaggle-Parity Evaluation Harness**
   * A local testing harness that perfectly mirrors Kaggle's vLLM evaluation parameters (`temperature=0.0`, `max_new_tokens=7680`). Prevents leaderboard divergence by ensuring local accuracy reflects actual submission accuracy.
4. **Layer 4: Automated CI/CD Guardrails**
   * GitHub Actions automatically parse the `adapter_config.json`, mathematically validate that the LoRA rank meets the competition limit, and construct the `submission.zip` artifact.
5. **Layer 5: Monitoring & Overfitting Detection**
   * Tracks local evaluation delta vs. public leaderboard scores to monitor model generalization.
6. **Layer 6: Production Infrastructure (Docker)**
   * A containerized environment pulling the official `nvidia/cuda:12.1.1-devel-ubuntu22.04` base image, ensuring 100% hardware and dependency parity in the cloud.

---

## 📂 Project Structure

```text
nemotron-reasoning-ops/
├── .github/workflows/      # CI/CD validation and auto-packaging pipelines
├── data/                   # Data layer (strictly tracked by DVC, ignored by Git)
│   ├── raw/                # Synthetic CoT data and raw competition datasets
│   ├── processed/          # Tokenized, filtered, and formatted datasets
│   └── synthetic/          # LLM-generated math/logic problem templates
├── src/                    # Core pipeline logic
│   ├── data/               # Scripts for synthetic generation and preprocessing
│   ├── evaluation/         # GRPO reward shaping and vLLM parity evaluation
│   ├── models/             # LoRA adapters, Unsloth/TRL loops, and inference
│   └── utils/              # Helper functions (logging, hardware routing)
├── tests/                  # Pytest unit tests for reward functions and formatting
├── notebooks/              # Exploratory Data Analysis (EDA) and prize write-ups
├── outputs/                # Local staging for adapter_config.json and weights
├── Dockerfile              # NVIDIA CUDA 12.1 production container blueprint
├── requirements.txt        # Core ML and Ops dependencies
└── config.yaml             # Global pipeline hyperparameters

## 🧠 The GRPO Multi-Component Reward Function

To excel in structured reasoning, models must be penalized for hallucinations and rewarded for strict formatting. Our custom GRPO reward function (`src/evaluation/grpo_reward.py`) applies a heavily shaped reward structure based on Kaggle's evaluation metric:

* **Format Compliance (+0.1):** Model successfully wraps its final answer using the LaTeX `\boxed{answer}` command.
* **Exact/Numerical Correctness (+1.0):** The extracted string exactly matches the ground truth, OR falls within Kaggle's strict relative numerical tolerance of `1e-5`.
* **Hallucination Penalty (-0.5):** Model confidently outputs a boxed answer, but the mathematics are incorrect.
* **Missing Format Penalty (-1.0):** Model completely fails to use the `\boxed{}` syntax.
* **Padding Penalty (-0.2):** Discourages the model from generating excessive tokens (length > 2000) to farm reasoning steps without reaching a conclusion.

---

## 🛠️ Quick Start (Local Prototyping)

The codebase dynamically detects your hardware. For local validation without a GPU, it gracefully falls back to CPU computation for fast plumbing tests.

### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/Neel-K26/nemotron-reasoning-ops.git](https://github.com/Neel-K26/nemotron-reasoning-ops.git)
cd nemotron-reasoning-ops

# Activate your Conda environment
conda activate ml_env

# Install CPU dependencies (for local testing)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt


# Initialize DVC and pull the latest tracked datasets
dvc init
dvc pull


# Generate a batch of synthetic reasoning data
python -m src.data.generate_synthetic

# Run a 2-step RL loop plumbing test
python -m src.models.train

# Evaluate the generated LoRA adapter against Kaggle parameters
python -m src.evaluation.evaluate


# Launch the MLflow tracking UI
mlflow ui



---

## 🐳 Production Deployment (GCP / NVIDIA Blackwell)

To deploy the pipeline for heavy workloads on cloud GPU runners, utilize the provided Docker infrastructure to ensure CUDA 12.1 compatibility.

```bash
# Build the production image
docker build -t nemotron-ops:v1 .

# Run the container interactively with full GPU passthrough
docker run --gpus all -it --rm nemotron-ops:v1 /bin/bash

# Inside the container, execute the full training loop
python -m src.models.train
