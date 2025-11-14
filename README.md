# Transformer

This repository provides an introductory exploration of the **Transformer architecture** and an implementation attempt of the well-known **self-attention algorithm**.  
It is inspired by the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

As a practical example, we demonstrate how to build a translation model capable of converting text from English to Spanish using the Open Access Books Dataset.

---

## 📖 Overview
Transformers have revolutionized natural language processing and machine learning by replacing recurrent and convolutional structures with **attention mechanisms**.  
This project aims to:
- Explain the core ideas behind self-attention.
- Provide a minimal, educational implementation.
- Serve as a starting point for learners interested in modern deep learning architectures.

---

## 🛠 Features
- Step‑by‑step introduction to the Transformer model.
- Implementation of **scaled dot‑product attention**.
- Clear, commented code for educational use.
- Example translation pipeline (English → Spanish).
- References to the original paper for deeper study.

---

## Dataset

We use the IWSLT2017 English↔Spanish (EN↔ES) dataset for training and evaluation. Some features from this dataset:

- ~200,000 sentence pairs from TED talks
- Clean and preprocessed
- Open access
- Loads in one line via HuggingFace 

---

## 🚀 Getting Started

- A Dockerfile is included to simplify environment setup.
- Clone the repository, build the Docker image, and start experimenting with the provided notebooks and scripts.


---
## 🤝 Contributing
Contributions, suggestions, and improvements are welcome!
Feel free to open an issue or submit a pull request.
