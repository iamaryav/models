## currently...

- TF-IDF in MLP Logistic regression
- Linear Regression | House-price prediction with California Housing | sklearns, numpy
- Scale to larger datasets: evaluation metrics, train/test split, batch training, learning-rate scheduler, dropout, BatchNorm / LayerNorm, early stopping, inference pipeline

## Neural-network architectures Pre-training

### Text, sequence, and transformer models
- Re-implement bigram -> makemore -> gpt
- RNN (Recurrent Neural Network) — including LSTM and GRU
- Transformer — self-attention, positional embeddings, encoder/decoder blocks, and training loop
- LLM (Large Language Model) — e.g., GPT, LLaMA
- SLM (Small Language Model) — efficient language models for edge or low-memory use
- MoE (Mixture of Experts) — sparse expert routing for scalable transformers
- SSM (State Space Model) — efficient long-sequence models, e.g., Mamba and RetNet
- LNN (Liquid Neural Network) — adaptive, continuous-time sequence models
- RLM (Reasoning Language Model) — multi-step reasoning, tool use, and test-time scaling
- LAM (Large Action Model) — planning and generating physical or digital action sequences

### Vision and multimodal models

- CNN (Convolutional Neural Network) — CNN → ResNet → Vision Transformer (ViT) → fine-tune on CIFAR-10/custom dataset
- SAM (Segment Anything Model) — promptable image segmentation
- VLM (Vision-Language Model) — image-text understanding, e.g., CLIP and Flamingo
- MLLM (Multimodal Large Language Model) — text with images, audio, video, or other modalities
- VLA (Vision-Language-Action Model) — convert visual and language inputs into robot or embodied-agent actions

## Post-training 

- Evaluation design — build capability, safety, and regression evals before training; keep a held-out test set
- Capability profiling — map strengths and failure modes across tasks, languages, reasoning, safety, and tool use to target post-training
- Data curation — quality filtering, deduplication, decontamination, and balanced instruction/task datasets
- Supervised fine-tuning (SFT) — instruction following, chat formatting, structured outputs, and domain adaptation
- Tool-use and agent training — function calling, retrieval, code execution, web/computer use, and multi-step task traces
- Human feedback and reward modeling (RLHF) — collect human-ranked responses; train reliable reward models or reward graders
- Direct Preference Optimization (DPO) — align style, helpfulness, and other subjective preferences using chosen/rejected pairs
- RL environments and verifiers — build reset/step task environments, tool sandboxes, outcome checkers, and reward signals for agent rollouts
- RLHF and reinforcement learning with verifiable rewards — optimize reasoning, coding, and tool-use tasks with human feedback or checkable graders; study PPO and GRPO (Group Relative Policy Optimization)
- Reasoning post-training — process supervision, outcome verification, self-correction, and test-time compute strategies
- Safety alignment — refusal behavior, policy training, adversarial red-teaming, and safety evals
- Distillation — use a stronger model to create data and train smaller, faster specialist models
- Continuous evaluation and iteration — monitor reward hacking, regressions, and capability/safety trade-offs; improve data, graders, and training

## ML problems

| Tier | Algorithm | Study focus | Suggested problem |
| --- | --- | --- | --- |
| Tier 1 | Linear & Logistic Regression | Normal equation, gradient descent, and L1/L2 regularization | Titanic survival or binary Iris classification |
| Tier 1 | Neural Networks / MLPs | Forward and backward passes; ReLU, sigmoid, tanh, softmax, and loss functions | XOR (to demonstrate non-linearity) or MNIST |
| Tier 1 | PCA, SVD & Eigendecomposition | Explained variance and dimensionality reduction | Eigenfaces or 2D visualization of Iris |
| Tier 2 | K-Means | Lloyd's algorithm, initialization, and the elbow method | Customer segmentation or image color quantization |
| Tier 2 | Decision Tree | Entropy/Gini splitting criteria and pruning | Iris or Titanic |
| Tier 2 | Random Forest | Bagging, feature randomness, and out-of-bag error | Feature importance on a tabular dataset |
| Tier 2 | Gradient Boosting (XGBoost) | Boosting intuition and gradient-based updates | Kaggle-style tabular regression |
| Tier 3 | SVM | Hinge loss, the RBF kernel trick, and the dual formulation | XOR with an RBF kernel or handwritten digits |
| Tier 3 | Naive Bayes | Conditional probability and Laplace smoothing | Spam detection |
| Tier 3 | t-SNE / UMAP | Visualization and loss-function intuition | 2D visualization of high-dimensional embeddings |
| Tier 3 | KNN | Distance metrics, neighbor selection, and voting | MNIST benchmark |
