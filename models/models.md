
## Neural-network architectures Pre-training

#### Books/Papers - Implementation | Read/Recreate Paper/blogpost
- resnet -> vit 
- Papers -> Chinchilla → PaLM/PaLM 2 → DeepSeek v4.1 → Kimi K3 -> GLM → Model Card
- Numpy, Pandas, Scikit Learn, PyTorch, CUDA/Triton
- RL library in C/Python

### Sequence by architecture type

- Full order (paper → torch): Autograd → MLP → Activations → SGD → Bigram → MLP w/ embeddings → Word2Vec → BatchNorm → WaveNet-like → LeNet → AlexNet → VGG → ResNet → Style Transfer → VAE → GAN → DQN → Distillation → RNN → LSTM → GRU → Attention → Transformer → BLEU → BERT → GPT-2 → ViT → Diffusion → DiT → RoPE → LoRA → SLM → Llama 2 → Mistral 7B → NeoBERT → MoE → SSM → LNN → VLM → SAM → Llama 4 → RLM → LAM → VLA
- MLP: Autograd → NN / MLP → Activations → SGD → Bigram → MLP w/ embeddings → Word2Vec → BatchNorm → WaveNet-like
- RNN: RNN → LSTM → GRU → LNN
- CNN: LeNet → AlexNet → VGG / ResNet → Style Transfer
- Encoder-decoder Transformer: Attention → Transformer → BLEU
- Encoder Transformer: BERT → ViT → NeoBERT → SAM
- Decoder Transformer: GPT-2 → RoPE → SLM → Llama 2 → Mistral 7B → VLM → RLM → LAM → VLA
- MoE: MoE → Llama 4
- State space: SSM
- Generative: VAE → GAN → Diffusion → DiT
- RL: DQN
- Adaptation: Distillation → LoRA


## Post-training

- Evaluation: Evaluation design → Capability profiling → Continuous evaluation
- Data: Data curation
- Supervised: SFT → Tool-use and agent training
- Preference: RLHF / reward modeling → DPO → ORPO → SimPO → KTO → RLOO
- RL: RL environments and verifiers → PPO → GRPO
- Reasoning and safety: Constitutional AI → Reasoning post-training → Safety alignment
- Benchmarks: MMLU-Pro → GPQA → SWE-bench → IFEval → RewardBench → Arena/MT-Bench

## Core ML sequence
- Linear: Linear Regression → Logistic Regression → Softmax Regression → SVM
- Probabilistic: Naive Bayes
- Distance: KNN → K-Means
- Tree: Decision Tree → Random Forest → Gradient Boosting
- Dimensionality reduction: PCA / SVD → t-SNE / UMAP → Collaborative Filtering

## Agentic Orchestration & RAG

- RAG: RAG → Advanced RAG
- Frameworks: LangChain → LangGraph → LlamaIndex
- Agents: Agentic Patterns (ReAct, multi-agent, HITL) → MCP
