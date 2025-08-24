# My Journey in AI: Building Models From Scratch

Welcome to my repository! This space documents my journey as an AI Engineer, with a focus on understanding core concepts by implementing cutting-edge models from the ground up using PyTorch. The philosophy here is that true understanding comes from building. By manually coding each component, from data loaders to attention mechanisms, the goal is to demystify the inner workings of these complex architectures and cultivate a strong, first-principles-based foundation in AI.

---

## Completed Projects

Here are the models I have built so far. Each project is self-contained in its directory and includes a detailed, technical README that breaks down the code and the underlying theory.

### 1. Transformer from Scratch

* **Description**: A complete implementation of the original Transformer model from the "Attention Is All You Need" paper. This project serves as a foundational exercise, focusing on the encoder-decoder architecture to solve a real-world Neural Machine Translation (NMT) task: translating from English to Indonesian using the Helsinki-NLP/opus-100 dataset.

* **Key Features**: This implementation includes all the critical components that made the Transformer revolutionary, such as **Multi-Head Self-Attention**, which allows the model to weigh the importance of different words in a sequence simultaneously, and sinusoidal **Positional Encoding** to give the model a sense of word order. It also features distinct **Encoder/Decoder stacks** and meticulous implementation of **Layer Normalization** and residual connections for stable training.

* **Directory**: `transformers/`

* [**➡️ View Detailed README**](./transformers/README.md)

### 2. Llama 2 from Scratch

* **Description**: A from-scratch implementation of the Llama 2 decoder-only architecture. This project moves beyond the original Transformer to explore the specific optimizations and architectural refinements that make modern Large Language Models (LLMs) like Llama 2 so efficient and powerful for text generation.

* **Key Features**: The code highlights several key innovations. **RMSNorm** is used for pre-normalization, offering a computationally simpler yet effective alternative to LayerNorm. **Rotary Positional Embeddings (RoPE)** are implemented to more effectively encode relative positional information. For efficiency, the model uses **Grouped-Query Attention (GQA)**, which significantly reduces the memory and computation required during inference, and a **KV Cache** to make auto-regressive generation much faster. Finally, the **SwiGLU** activation function is used in the feed-forward layers for improved performance.

* **Directory**: `llama2-from-scratch/`

* [**➡️ View Detailed README**](./llama2-from-scratch/README.md)

---

## 🗺️ What's Next on the Roadmap?

My learning journey is ongoing. Here are the next frontiers I plan to explore and implement from scratch to deepen my understanding of the field's current trajectory.

### 1. Multi-Modal Models (Vision-Language)

* **Goal**: To bridge the gap between vision and language by building a model that can process and understand both images and text in a unified way. The primary objective is to implement a model capable of performing tasks like detailed image captioning or Visual Question Answering (VQA), where the model must answer a textual question based on the content of an image.

* **Concepts to Explore**: This will involve a deep dive into the **Vision Transformer (ViT)** for image processing, designing effective **cross-attention mechanisms** that allow text and image representations to interact, and creating **joint embedding spaces** where both modalities can be meaningfully compared and fused.

### 2. Sparse Mixture-of-Experts (MoE) Models

* **Goal**: To tackle the challenge of scaling models to trillions of parameters without a proportional increase in computational cost. The plan is to build a model that leverages a sparse MoE architecture, where only a small subset of the model's weights (the "experts") are activated for any given input token.

* **Concepts to Explore**: This implementation will focus on the core components of MoE, including the **gating network (or router)** that decides which experts to send a token to, the design of the individual **expert layers** (which are typically feed-forward networks), and implementing strategies for **load balancing** to ensure experts are utilized efficiently during training.
