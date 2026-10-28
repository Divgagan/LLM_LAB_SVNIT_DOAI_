# 🧠 LLM Lab — SVNIT DOAI

> **Language Identification & Subword Tokenization on Indian Languages**
> Semester 7 · Department of Artificial Intelligence · SVNIT Surat

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-IndicCorpV2-orange?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Lab Notebooks](#-lab-notebooks)
  - [Lab 01 — Baseline Language Identifier (CPU/GPU)](#lab-01--baseline-language-identifier-cpugpu)
  - [Lab 02 — TF-IDF Variant Experiments (GPU)](#lab-02--tf-idf-variant-experiments-gpu)
  - [Lab 03 — BPE & WordPiece Tokenization](#lab-03--bpe--wordpiece-tokenization)
- [Dataset](#-dataset)
- [Results at a Glance](#-results-at-a-glance)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Key Concepts](#-key-concepts)
- [Hardware](#-hardware)

---

## 🔍 Overview

This repository contains three progressive lab notebooks exploring **classical NLP techniques** for Indian language processing using the [IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) dataset.

The series covers the full pipeline from raw multilingual text to a trained classifier and subword tokenizer — without using any pre-trained transformer models. Everything is built from scratch using **NumPy, PyTorch, and the HuggingFace `tokenizers` library**.

```
Raw Text (IndicCorpV2)
       │
       ▼
  Sentence Splitting & Filtering
       │
       ▼
  TF-IDF Feature Extraction          ──► Lab 01 / Lab 02
  (word n-grams + char n-grams)
       │
       ▼
  Softmax Classifier (GPU)
       │
       ▼
  Evaluation (Macro-F1 per language)
       │
       ▼
  BPE / WordPiece Tokenizer Training  ──► Lab 03
  (varying training size & vocab size)
```

---

## 📓 Lab Notebooks

### Lab 01 — Baseline Language Identifier (CPU/GPU)

**File:** [`01_lab_.ipynb`](01_lab_.ipynb)

A complete language identification pipeline for **23 Indian languages**, built entirely from scratch.

#### What it does

| Step | Description |
|------|-------------|
| **1. Setup** | Imports, seeds, hyperparameters, CUDA device detection |
| **2. Dataset** | Streams `IndicCorpV2`, splits into sentences, saves `train/val/test.csv` (80/10/10) |
| **3. TF-IDF Features** | Word unigrams, word bigrams, char 2/3/4-grams → 40,000-dim sparse vectors |
| **4. Softmax Classifier** | Custom PyTorch softmax regression with L2 regularization, mini-batch SGD |
| **5. Evaluation** | Per-language F1 scores, macro-F1 on held-out test set |
| **6. Inference** | `predict_language(text)` — try your own sentence |
| **7. Training Curve** | Loss and validation accuracy over 25 epochs |

#### Key Hyperparameters

```python
SENTENCES_PER_LANG  = 1000
EPOCHS              = 25
BATCH_SIZE          = 256
LR                  = 0.5
L2_REG              = 1e-5
MAX_VOCAB_PER_TYPE  = 8000   # per feature type (W1, W2, C2, C3, C4)
```

#### Result Summary

| Metric | Value |
|--------|-------|
| **Test Macro-F1** | **0.9194** |
| Best languages | Gujarati, Kannada, Tamil, Telugu, Malayalam, Punjabi, Odia → **F1 = 1.000** |
| Hardest language | Konkani (gom_Deva) → **F1 = 0.512** |

> 💡 **Why is Konkani hard?** It shares many Devanagari character sequences with Hindi, Marathi, and Maithili — all of which also use the Devanagari script.

---

### Lab 02 — TF-IDF Variant Experiments (GPU)

**File:** [`02_lab_.ipynb`](02_lab_.ipynb)

Extends Lab 01 by running a **systematic comparison of 6 TF-IDF weighting schemes**, all trained on GPU.

#### TF-IDF Variants Compared

| # | TF Type | IDF Type | Description |
|---|---------|----------|-------------|
| 1 | Raw count | Unnormalized | Baseline (same as Lab 01) |
| 2 | Raw count | Normalized (÷ max IDF) | Softer IDF scaling |
| 3 | Normalized ÷ total tokens | Unnormalized | Relative term frequency |
| 4 | Normalized ÷ total tokens | Normalized | Double normalization |
| 5 | Augmented (0.5 + 0.5 × tf/max) | Unnormalized | Max-freq normalization |
| 6 | Augmented (0.5 + 0.5 × tf/max) | Normalized | Best of both normalizations |

#### Architecture

```python
class TorchSoftmaxRegression:
    # Parameters: W (n_features × n_classes), b (n_classes,)
    # Training:   mini-batch gradient descent on GPU
    # Loss:       cross-entropy + L2 weight decay
    # Inference:  batched argmax(XW + b)
```

#### Experimental Design

- **Same dataset** as Lab 01 (reuses `train/val/test.csv`)
- **Same vocabulary** (40,000 features)
- **20 epochs** per experiment
- Results recorded for all 6 combos → comparison table + plot

---

### Lab 03 — BPE & WordPiece Tokenization

**File:** [`03_lab_.ipynb`](03_lab_.ipynb)

A comprehensive study of **subword tokenization** on a single Indian language (default: **Hindi / hin_Deva**) using the HuggingFace `tokenizers` library.

#### What it does

| Step | Description |
|------|-------------|
| **1. Setup** | Language selection, training sizes, vocab sizes, data paths |
| **2. Data Collection** | Streams up to 1M+ Hindi sentences from IndicCorpV2 |
| **3. Splits** | Length-balanced dev/test (1000 each), 1M-sentence training pool |
| **4. Tokenizer Factories** | `train_bpe()` and `train_wordpiece()` with NFKC normalization |
| **5. Experiment 1** | Vary **training size** (100K → 300K → 500K → 1M) at fixed vocab = 30K |
| **6. Experiment 2** | Vary **vocabulary size** (20K → 30K → 50K) at fixed training = 1M |
| **7. Analysis** | avg/median/max tokens per sentence, UNK rate, plots |

#### Experimental Parameters

```python
TRAINING_SIZES  = [100_000, 300_000, 500_000, 1_000_000]
VOCAB_SIZES     = [20_000, 30_000, 50_000]
SPECIAL_TOKENS  = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
```

#### Key Findings

- Both BPE and WordPiece achieve **near-zero UNK rates** on Hindi even at 100K training sentences (rich Devanagari character coverage)
- Average tokens per sentence: **~36** (dev) and **~36** (test) — stable across training sizes
- Larger vocabulary → **slightly fewer tokens per sentence** (more complete words are in vocab)
- WordPiece produces **slightly more tokens** than BPE at the same vocab size

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | [ai4bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) |
| **Subset** | `indiccorp_v2` |
| **Access** | Streaming (no full download required) |
| **Languages (Lab 01/02)** | 23 Indian languages across 8 scripts |
| **Split** | 80% train / 10% val / 10% test (per language, stratified) |
| **Train samples** | 18,400 sentences |
| **Val samples** | 2,300 sentences |
| **Test samples** | 2,300 sentences |

### Languages Covered

| Code | Language | Script |
|------|----------|--------|
| `asm_Beng` | Assamese | Bengali |
| `ben_Beng` | Bengali | Bengali |
| `brx_Deva` | Bodo | Devanagari |
| `doi_Deva` | Dogri | Devanagari |
| `gom_Deva` | Konkani | Devanagari |
| `guj_Gujr` | Gujarati | Gujarati |
| `hin_Deva` | Hindi | Devanagari |
| `kan_Knda` | Kannada | Kannada |
| `kas_Arab` | Kashmiri | Arabic |
| `mai_Deva` | Maithili | Devanagari |
| `mal_Mlym` | Malayalam | Malayalam |
| `mar_Deva` | Marathi | Devanagari |
| `mni_Mtei` | Manipuri | Meitei |
| `npi_Deva` | Nepali | Devanagari |
| `ory_Orya` | Odia | Odia |
| `pan_Guru` | Punjabi | Gurmukhi |
| `san_Deva` | Sanskrit | Devanagari |
| `snd_Deva` | Sindhi | Devanagari |
| `tam_Taml` | Tamil | Tamil |
| `tel_Telu` | Telugu | Telugu |
| `urd_Arab` | Urdu | Arabic |
| `khasi` | Khasi | Latin |
| `santhali` | Santali | Ol Chiki |

---

## 📈 Results at a Glance

### Lab 01 / 02 — Per-Language Test F1 (Sorted)

```
gom_Deva  Konkani    ████████░░░░░░░░░░░░  0.512
mar_Deva  Marathi    █████████████░░░░░░░  0.688
mai_Deva  Maithili   ████████████████░░░░  0.811
hin_Deva  Hindi      ████████████████░░░░  0.833
doi_Deva  Dogri      █████████████████░░░  0.853
san_Deva  Sanskrit   █████████████████░░░  0.860
brx_Deva  Bodo       █████████████████░░░  0.862
npi_Deva  Nepali     ██████████████████░░  0.936
khasi     Khasi      ██████████████████░░  0.939
mni_Mtei  Manipuri   ███████████████████░  0.953
asm_Beng  Assamese   ███████████████████░  0.975
urd_Arab  Urdu       ███████████████████░  0.975
kas_Arab  Kashmiri   ███████████████████░  0.980
ben_Beng  Bengali    ███████████████████░  0.980
santhali  Santali    ████████████████████  0.995
snd_Deva  Sindhi     ████████████████████  0.995
kan_Knda  Kannada    ████████████████████  1.000
guj_Gujr  Gujarati   ████████████████████  1.000
ory_Orya  Odia       ████████████████████  1.000
pan_Guru  Punjabi    ████████████████████  1.000
tam_Taml  Tamil      ████████████████████  1.000
tel_Telu  Telugu     ████████████████████  1.000
mal_Mlym  Malayalam  ████████████████████  1.000
```

**Overall Test Macro-F1: `0.9194`**

---

## 🗂 Project Structure

```
02_lab/
├── 📓 01_lab_.ipynb          # Lab 01: Baseline language ID (CPU/GPU)
├── 📓 02_lab_.ipynb          # Lab 02: TF-IDF variant experiments
├── 📓 03_lab_.ipynb          # Lab 03: BPE & WordPiece tokenization
│
├── 📄 requirements.txt       # Python dependencies
│
├── 📊 train.csv              # 18,400 training sentences (23 languages)
├── 📊 val.csv                # 2,300 validation sentences
├── 📊 test.csv               # 2,300 test sentences
│
└── 📁 lab03_data/            # Lab 03 outputs (auto-created)
    ├── hin_Deva_sentences.csv          # Raw sentence pool (~1M)
    ├── hin_Deva_train.csv              # 1M training sentences
    ├── hin_Deva_dev.csv                # 1,000 dev sentences
    ├── hin_Deva_test.csv               # 1,000 test sentences
    └── hin_Deva_tokenization_results.csv  # Experiment results
```

> **Note:** The CSV files are committed to this repo so you can run the notebooks without re-downloading from IndicCorpV2.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- Anaconda / Miniconda (recommended)
- NVIDIA GPU with CUDA 12.x (optional but recommended for Lab 01/02)

### 1. Create & activate the conda environment

```bash
conda create -n torch_env python=3.10 -y
conda activate torch_env
```

### 2. Install PyTorch (GPU)

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:
```bash
pip install torch
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:
```
datasets
tokenizers
numpy
pandas
scipy
matplotlib
jupyter
ipykernel
```

### 4. Register the kernel (VS Code / JupyterLab)

```bash
python -m ipykernel install --user --name torch_env --display-name "Python (torch_env)"
```

---

## ▶️ How to Run

Open each notebook in **VS Code** or **JupyterLab** and select the `torch_env` kernel.

```bash
jupyter lab
```

Run cells top-to-bottom. Each notebook is self-contained:

| Notebook | First Run | Subsequent Runs |
|----------|-----------|-----------------|
| `01_lab_.ipynb` | Streams data from HuggingFace, saves CSVs | Loads existing CSVs instantly |
| `02_lab_.ipynb` | Same as Lab 01 if CSVs missing | Loads existing CSVs instantly |
| `03_lab_.ipynb` | Streams ~1M Hindi sentences (≈10–20 min) | Loads from `lab03_data/` cache |

> ⚠️ **HuggingFace unauthenticated requests** may be rate-limited. Set a token to avoid this:
> ```bash
> huggingface-cli login
> # OR
> export HF_TOKEN=hf_your_token_here
> ```

---

## 💡 Key Concepts

### TF-IDF Features Used

```
Feature Type     Prefix   Example
─────────────────────────────────────────────────────
Word unigrams    W1_      W1_नमस्ते
Word bigrams     W2_      W2_नमस्ते दुनिया
Char 2-grams     C2_      C2_नम
Char 3-grams     C3_      C3_नमस्
Char 4-grams     C4_      C4_नमस्त
```

- Vocabulary limited to **top 8,000 features per type** → 40,000 total
- Features with document frequency < 2 are discarded
- Final vectors are **L2-normalized**

### Softmax Classifier (from scratch)

```python
# Forward pass
scores = X @ W + b           # (batch, n_classes)
probs  = softmax(scores)     # (batch, n_classes)

# Loss
loss = cross_entropy(scores, y) + 0.5 * λ * ||W||²

# Backward pass (manual gradients)
δ_scores = (probs - one_hot) / batch_size
δ_W = Xᵀ @ δ_scores + λ * W
δ_b = δ_scores.sum(axis=0)

# Update
W -= lr * δ_W
b -= lr * δ_b
```

### BPE vs WordPiece

| Property | BPE | WordPiece |
|----------|-----|-----------|
| **Merge criterion** | Most frequent pair | Max likelihood |
| **Unknown handling** | Falls back to chars | `[UNK]` token |
| **Used in** | GPT, RoBERTa | BERT, mBERT |
| **Avg tokens (Hindi)** | ~36.4 | ~36.7 |
| **UNK rate (Hindi)** | ≈ 0.00% | ≈ 0.00% |

---

## 🖥️ Hardware

Tested on:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop (4 GB VRAM) |
| CUDA | 12.1 |
| PyTorch | 2.5.1 |
| Python | 3.10 (Anaconda `torch_env`) |
| OS | Windows 11 |

All labs run fine on **CPU** as well — just set `DEVICE = "cpu"` manually. Training times will be 3–5× slower on CPU.

---

## 👨‍💻 Author

**Gagan Diwakar**
B.Tech — Artificial Intelligence & Data Engineering
Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
Department of Artificial Intelligence · Semester 7

---

<div align="center">

Made with ❤️ for DOAI Lab · SVNIT Surat

</div>
