# Read/Recreate Paper/blogpost

## Core ML sequence
- Linear: Linear Regression → Logistic Regression → Softmax Regression → SVM
- Probabilistic: Naive Bayes
- Distance: KNN → K-Means
- Tree: Decision Tree → Random Forest → Gradient Boosting
- Dimensionality reduction: PCA / SVD → t-SNE / UMAP → Collaborative Filtering

## Neural-network architectures Pre-training

#### Books/Papers - Implementation
- Numpy, Pandas, Scikit Learn, PyTorch, CUDA/Triton
- RL library in C/Python

### Sequence by architecture type

- Full order (paper → torch): Autograd → MLP → Activations → SGD → Bigram → MLP w/ embeddings → BatchNorm → WaveNet-like → Word2Vec → LeNet → AlexNet → VGG → ResNet → Style Transfer → RNN → LSTM → GRU → VAE → GAN → DQN → Distillation → Attention → Transformer → BLEU → BERT → GPT-2 → ViT → Diffusion → RoPE → LoRA → SLM → Llama 2 → Mistral 7B → MoE → DiT → SSM → LNN → VLM → NeoBERT → Llama 4 → RLM → LAM → SAM → VLA
- MLP: Autograd → NN / MLP → Activations → SGD → Bigram → MLP w/ embeddings → BatchNorm → WaveNet-like → Word2Vec
- RNN: RNN → LSTM → GRU → LNN
- CNN: LeNet → AlexNet → VGG / ResNet → Style Transfer
- Encoder-decoder Transformer: Attention → Transformer → BLEU
- Encoder Transformer: BERT → NeoBERT → ViT → SAM
- Decoder Transformer: GPT-2 → SLM → RoPE → Llama 2 → Mistral 7B → RLM → LAM → VLM → VLA
- MoE: MoE → Llama 4
- State space: SSM
- Generative: VAE → GAN → Diffusion → DiT
- RL: DQN
- Adaptation: LoRA → Distillation

Papers -> Chinchilla → PaLM/PaLM 2 → DeepSeek v4.1 → Kimi K3 → Model Card

## Post-training

- Evaluation: Evaluation design → Capability profiling → Continuous evaluation
- Data: Data curation
- Supervised: SFT → Tool-use and agent training
- Preference: RLHF / reward modeling → DPO → ORPO → SimPO → KTO → RLOO
- RL: RL environments and verifiers → PPO → GRPO
- Reasoning and safety: Constitutional AI → Reasoning post-training → Safety alignment
- Benchmarks: MMLU-Pro → GPQA → SWE-bench → IFEval → RewardBench → Arena/MT-Bench

## Agentic Orchestration & RAG

- RAG: RAG → Advanced RAG
- Frameworks: LangChain → LangGraph → LlamaIndex
- Agents: Agentic Patterns (ReAct, multi-agent, HITL) → MCP
