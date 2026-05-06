# Deep Learning for Social Sciences — PyTorch Implementations

Coursework implementations from **Deep Learning for Social Sciences** (grade: 1.0/1.0) at Universität Konstanz, MSc in Social and Economic Data Science.

Each module implements a core deep learning architecture from scratch in PyTorch and applies it to a dataset relevant to social science research — from labor market classification to news topic detection to sequence modeling.

---

## Modules

### MLP — Multi-Layer Perceptron
**Dataset:** Uruguay job vacancy dataset  
**Task:** Multi-class classification of job postings by occupational category

Implements a feedforward neural network with configurable hidden layers, batch normalization, and dropout regularization. Explores how labor market data can be structured for classification tasks — relevant to computational approaches to occupational segregation and labor market analysis.

### CNN — Convolutional Neural Network
**Task:** Feature extraction and classification using convolutional filters

Implements a convolutional architecture with pooling layers and fully connected head. Includes a written analysis comparing CNN performance characteristics and the representational assumptions underlying convolutional feature extraction.

### RNN — Recurrent Neural Network
**Datasets:** AG News corpus, MNIST  
**Tasks:** News topic classification, sequence modeling

Three notebooks:
- `ag_news.ipynb` — Vanilla RNN for 4-class news topic classification (World, Sports, Business, Science/Tech)
- `ag_news_hf.ipynb` — Same task using Hugging Face tokenizers for preprocessing comparison
- `BiRNN.ipynb` — Bidirectional RNN, evaluating whether backward context improves classification on news text

News classification is a foundational task in computational social science for studying media framing, agenda-setting, and information diffusion.

### Transformer
**Task:** Sequence-to-sequence learning (copy and reverse tasks)

Implements the core Transformer architecture (multi-head self-attention, positional encoding, encoder-decoder structure) from scratch. The copy/reverse tasks isolate attention mechanism behavior without confounding dataset noise — a standard diagnostic approach before applying Transformers to real corpora.

### LLM — Large Language Model Fine-tuning
**Model:** BERT (bert.ipynb)  
**Task:** Fine-tuning a pre-trained language model for downstream classification

Implements BERT fine-tuning using Hugging Face Transformers. Covers tokenization, attention masks, and transfer learning from a pre-trained language model — the same approach underlying most modern NLP pipelines in CSS research (stance detection, sentiment analysis, claim classification).

---

## Stack

- Python, PyTorch, Hugging Face Transformers
- Jupyter Notebook
- Datasets: AG News, MNIST, Uruguay vacancy corpus

## Context

These implementations were developed as part of the graduate course *Deep Learning for Social Sciences* at Universität Konstanz, which covers the theoretical foundations and practical application of neural network architectures to social science research problems.
