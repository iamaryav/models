## Core ML sequence

## Neural-network architectures Pre-training

Numpy, Pandas, Scikit Learn, PyTorch, CUDA
- RL library in C/Python

### Text, sequence, and transformer models

Linear → Logistic → Softmax → NN/MLP → |bigram → makemore series (MLP w/ embeddings, BatchNorm, WaveNet-like) → RNN → LSTM → GRU → GPT → SLM → MoE → SSM → LNN → RLM → LAM   
Papers -> Chinchilla → PaLM/PaLM 2 → DeepSeek v4 → Kimi K3 → Model Card

### Vision and multimodal models

Linear → Logistic → Softmax on MNIST → MLP on CIFAR-10 → |LeNet + Conv + Pool from scratch → AlexNet → VGG/ResNet → ViT → Diffusion → SAM → VLM → MLLM → VLA

## Post-training

- Evaluation design — build capability, safety, and regression evals before training; keep a held-out test set
- Capability profiling — map strengths and failure modes across tasks, languages, reasoning, safety, and tool use to target post-training
- Data curation — quality filtering, deduplication, decontamination, and balanced instruction/task datasets
- Supervised fine-tuning (SFT) — instruction following, chat formatting, structured outputs, and domain adaptation
- Tool-use and agent training — function calling, retrieval, code execution, web/computer use, and multi-step task traces
- Human feedback and reward modeling (RLHF) — collect human-ranked responses (e.g., InstructGPT); train reliable reward models or reward graders
- Preference Optimization Algorithms — Direct Preference Optimization (DPO), ORPO, SimPO, Kahneman-Tversky Optimization (KTO), and RLOO
- RL environments and verifiers — build reset/step task environments, tool sandboxes, outcome checkers, and reward signals for agent rollouts
- Reinforcement Learning with verifiable rewards — optimize reasoning, coding, and tool-use tasks using Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) (e.g., DeepSeekMath, DeepSeek-R1)
- Constitutional AI — training models to self-critique and revise responses based on a set of rules or principles
- Reasoning post-training — process supervision, outcome verification, self-correction, and test-time compute strategies
- Safety alignment — refusal behavior, policy training, adversarial red-teaming, and safety evals
- Distillation — use a stronger model to create data and train smaller, faster specialist models
- Advanced Evaluation Benchmarks — progression tracking across MMLU-Pro → GPQA → SWE-bench → IFEval → RewardBench → Arena/MT-Bench
- Continuous evaluation and iteration — monitor reward hacking, regressions, and capability/safety trade-offs; improve data, graders, and training



## Agentic Orchestration & RAG

- Retrieval-Augmented Generation (RAG) — vector databases (e.g., Chroma, FAISS), document chunking strategies, embedding models, and hybrid search
- Advanced RAG — query routing, self-correction, re-ranking, and parent-document retrieval
- LangChain — chains, prompt templates, output parsers, memory, and the broader tool integration ecosystem
- LangGraph — stateful, multi-actor applications built on LLMs; modeling complex agent workflows as cyclic graphs
- LlamaIndex — data frameworks specifically optimized for ingesting, structuring, and accessing private/domain-specific data
- Agentic Patterns — ReAct (Reasoning and Acting) prompting, multi-agent collaboration, and Human-in-the-loop (HITL)
- MCP



ML problems

Core AI / DL

- Linear Regression
- Logistic Regression
- Softmax Regression
- Neural Networks / MLPs
- PCA, SVD & Eigendecomposition

Tree-Based

- Decision Tree
- Random Forest
- Gradient Boosting (XGBoost)

Unsupervised / Distance

- K-Means
- KNN
- t-SNE / UMAP

Classical ML

- Naive Bayes
- SVM
