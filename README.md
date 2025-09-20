# Gemma3 270M from Scratch (PyTorch)
Implementing the Gemma3 270M transformer model from scratch using PyTorch. This repo includes all components for training, inference, and experimenting with Gemma3 on small datasets like TinyStories.

![Gemma3 Model](gemma.png)


---

## 📂 Project Structure

```bash
├── dataset.py       # Handles TinyStories dataset + BPE tokenization
├── model.py         # Implements Gemma3 architecture (attention, transformer blocks, embeddings)
├── train.py         # Training loop with warmup + cosine annealing scheduler
├── inference.py     # Inference / text generation script
├── requirements.txt # Python dependencies
└── README.md        # Project overview & instructions
```


---

## 🚀 Features
- **From-scratch Gemma3 implementation**  
  - Multi-Query Attention
  - Sliding Attention
  - Transformer Blocks with RMSNorm  
  - FeedForward MLP with dual parallel expansion networks
  - Token & Positional Embeddings
  - Weight tying & initialization
  - Query-Key (QK) normalization
  - Rotary Positional Embeddings (RoPE)
  
- **Training pipeline**  
  - [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) (HuggingFace)  
  - Sliding window dataset loader  
  - Mixed precision training (AMP + GradScaler)  
  - LR warmup + cosine annealing  
  - Gradient clipping & weight decay
    
- **Inference & Generation**  
  - Greedy decoding (deterministic)  
  - [Optional] Top-k sampling for creativity
 
- **Visualization**: Training vs validation loss plots  

---

## 📊 Model Config (default)
```python
GEMMA3_CONFIG = {
    "vocab_size": 50257,
    "context_length": 32768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hid_dim": 2048,
    "head_dim": 256,
    "rope_local_base": 10000.0,
    "rope_base": 1000000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
    "dtype": torch.bfloat16
}

```


## 📦 Installation

```bash
git clone https://github.com/ayushWeb07/Gemma3-Implementation.git
cd Gemma3-Implementation
pip install -r requirements.txt
```

## 📚 Dataset

We use the roneneldan/TinyStories
 dataset, a synthetic dataset of short stories written in simple language, specifically tailored for 4-5 year old kids.

---

## 📚 Usage
### 1️⃣ Prepare Dataset

`python dataset.py`

- Downloads **TinyStories** dataset from HuggingFace.
    
- Saves train data into 'train.bin' and validation data into 'validation.bin'
    

### 2️⃣ Train the Model

`python train.py`

- Runs the training loop with **warmup + cosine scheduler**.
    
- Automatically saves best model weights (`best_model_params.pth`).
    
- Plots train vs validation losses.

### 3️⃣ Run Inference

`python inference.py`

- Initializes a tokenizer from tiktoken

- Creates an input sample: Once upon a time there was a pumpkin.

- Does `language modeling` on the above mentioned sample text

```bash
Text (before generation): Once upon a time there was a pumpkin.
Text (after generation): Once upon a time there was a pumpkin. The pumpkin loved to...
```

---

## 📚 Resources Used

This project was inspired and guided by the following resources:

- [Pretraining Gemma3- Vizuara](https://youtu.be/bLDlwcl6hbA?list=PLPTV0NXA_ZSiR4_XoR1wy-3bv6J0oZ9Zs)

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)



