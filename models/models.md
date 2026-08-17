## Core ML sequence
Numpy, Pandas, Scikit Learn, PyTorch, CUDA

Complete these fundamentals before moving to sequence models and transformers:

1. **Evaluation and data splits** — define the task and metric; create train, validation, and held-out test sets; prevent duplicate leakage; establish a simple baseline.
2. **Linear Regression** — California Housing in scikit-learn, then NumPy; learn the loss, gradients, and regularization.
3. **Logistic Regression** — binary classification in scikit-learn, then NumPy; implement sigmoid, binary cross-entropy, thresholding, accuracy, precision, recall, and F1.
4. **Text classification baselines** — TF-IDF + Naive Bayes and Logistic Regression for spam detection; fit all preprocessing only on training data.
5. **MLPs** — build a scikit-learn baseline, then implement a one-hidden-layer network in NumPy, then PyTorch; learn forward passes, backpropagation, ReLU, sigmoid/softmax, and loss functions.
6. **Training loops at scale** — mini-batches, epochs, shuffling, validation loss, checkpoints, and inference pipelines.
7. **Regularization and stability** — L1/L2 regularization, dropout, learning-rate scheduling, BatchNorm/LayerNorm, early stopping, and overfitting diagnostics.

For each selected from-scratch project: build the scikit-learn reference first, reimplement it in NumPy, compare logits/loss/gradients on a small fixed dataset, then use PyTorch for larger training runs.

## Neural-network architectures Pre-training

### Text, sequence, and transformer models
- Linear → Logistic → Softmax → NN/MLP → bigram → makemore series (MLP w/ embeddings, BatchNorm, WaveNet-like) → GPT
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

- Linear on pixels → Logistic on images → Softmax on MNIST → MLP on CIFAR-10 → Conv + Pool from scratch → LeNet → AlexNet → VGG → ResNet → ViT → fine-tune on CIFAR-10/custom dataset
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

## ML problems

| Category | Algorithm | Implementation target | Study focus | Recommended dataset | Dataset details |
| --- | --- | --- | --- | --- | --- |
| Core AI / DL | Linear Regression | scikit-learn, then full NumPy | Normal equation, gradient descent, and L1/L2 regularization | [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | Regression; 20,640 districts and 8 numerical features. Start with scikit-learn, then reproduce it in NumPy. |
| Core AI / DL | Logistic Regression | scikit-learn, then full NumPy | Sigmoid, binary cross-entropy, thresholding, and classification metrics | [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) | Binary classification; 569 samples and 30 numerical features. Small enough to inspect every gradient. |
| Core AI / DL | Softmax Regression | scikit-learn, then full NumPy | Multiclass Logistic Regression, numerically stable softmax, and cross-entropy | [Optical Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | 10-class digit classification; verify that class probabilities sum to one. |
| Core AI / DL | Neural Networks / MLPs | scikit-learn, then full NumPy, then PyTorch | Forward/backward passes; ReLU, sigmoid, softmax, and loss functions | XOR (create manually) → [Optical Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | Implement mini-batch SGD, momentum, Adam, and finite-difference gradient checks; then move to PyTorch. |
| Core AI / DL | PCA, SVD & Eigendecomposition | scikit-learn | Explained variance and dimensionality reduction | [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html) | Create eigenfaces and reconstruct images with different component counts. |
| Tree-Based | Decision Tree | scikit-learn, then simple NumPy | Entropy/Gini splitting criteria and pruning | [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) | Tabular binary classification with mixed categorical and numerical features; practice encoding and pruning. |
| Tree-Based | Random Forest | scikit-learn | Bagging, feature randomness, out-of-bag error, and feature importance | [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) | Reuse the Decision Tree preprocessing, then compare a single tree with a forest. |
| Tree-Based | Gradient Boosting (XGBoost) | XGBoost/scikit-learn | Boosting intuition and gradient-based updates | [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | Reuse the regression pipeline; compare boosted trees against linear regression and random forests. |
| Unsupervised / Distance | K-Means | scikit-learn | Lloyd's algorithm, initialization, and the elbow method | [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail) | Aggregate transactions into RFM features per customer, then cluster customers. |
| Unsupervised / Distance | KNN | scikit-learn | Distance metrics, neighbor selection, and voting | [Optical Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | Learn how scaling and the choice of k affect performance. |
| Unsupervised / Distance | t-SNE / UMAP | Study conceptually; use scikit-learn/UMAP | Visualization and loss-function intuition | [Optical Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | Visualize the 64-dimensional digit vectors in 2D; color points by digit class. |
| Classical ML | Naive Bayes | scikit-learn | Conditional probability and Laplace smoothing | [UCI SMS Spam Collection](https://archive-beta.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection/files) | 5,574 labelled SMS messages. Compare count vectors and TF-IDF with Logistic Regression. |
| Classical ML | SVM | scikit-learn | Hinge loss, the RBF kernel trick, and the dual formulation | [Optical Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | Multiclass handwritten digits; compare linear and RBF SVMs after feature scaling. |

## Projects

- RL library in C/Python

## Papers

- Kimi 3/ DS technical papers - K3
- DeepSeek v4
- palm/palm2
- chinchilla

----

Model Card 

----
