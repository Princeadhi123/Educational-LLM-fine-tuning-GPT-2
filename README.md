# Educational LLM — Fine-Tuning GPT-2 on School Data

## Overview

This project fine-tunes a pre-trained GPT-2 model on educational content (textbooks, curricula, Q&A pairs, lesson plans) to create a specialized language model for school-level education. The model can then generate explanations, answer questions, summarize topics, and create quiz questions.

## Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EDUCATIONAL LLM FINE-TUNING PIPELINE                │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
  │  DATA SOURCES │     │  PREPROCESSING   │     │   TRAINING   │
  │              │     │                  │     │              │
  │  Wikipedia   │     │  Format → Unify  │     │  GPT-2 Base  │
  │  SciQ/SQuAD  ├────►│  Tokenize (GPT2) ├────►│  + LoRA      │
  │  Local Files │     │  Train/Val Split │     │  + Trainer   │
  │  Textbooks   │     │                  │     │              │
  └──────────────┘     └──────────────────┘     └──────┬───────┘
                                                       │
  collect_data.py       preprocess_data.py        train.py
                                                       │
                                                       ▼
                       ┌──────────────────┐     ┌──────────────┐
                       │    INFERENCE     │     │ CHECKPOINTS  │
                       │                  │     │              │
                       │  User Prompt     │     │  LoRA Adaptr │
                       │  → Format        │◄────│  Merged Model│
                       │  → Generate      │     │  Best Ckpt   │
                       │  → Clean Output  │     │              │
                       └──────────────────┘     └──────────────┘
                                                       
                        generate.py              models/checkpoints/
```

### Model Architecture (GPT-2 + LoRA)

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-2 Transformer (124M params)          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Token Embeddings (50257 vocab) + Position Embeddings │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              Transformer Block × 12                   │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Multi-Head Self-Attention                      │  │  │
│  │  │  ┌─────────┐                                    │  │  │
│  │  │  │ c_attn  │◄── LoRA Adapter (rank=64)         │  │  │
│  │  │  └─────────┘                                    │  │  │
│  │  │  ┌─────────┐                                    │  │  │
│  │  │  │ c_proj  │◄── LoRA Adapter (rank=64)         │  │  │
│  │  │  └─────────┘                                    │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Feed-Forward Network (MLP)                     │  │  │
│  │  │  ┌─────────┐                                    │  │  │
│  │  │  │  c_fc   │◄── LoRA Adapter (rank=64)         │  │  │
│  │  │  └─────────┘                                    │  │  │
│  │  │  ┌─────────┐                                    │  │  │
│  │  │  │ c_proj  │    (shared with attention c_proj)  │  │  │
│  │  │  └─────────┘                                    │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │  LayerNorm + Residual Connections                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │           LM Head → Next Token Prediction             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Frozen params: ~124M  │  LoRA trainable: ~6.5M (~5%)      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────┐    ┌──────────────────────────────────────────────────┐
│  Raw Input   │    │  Preprocessing (preprocess_data.py)              │
│              │    │                                                  │
│  Instruction │    │  "Explain gravity" + "Gravity is a force..."    │
│  Response    │    │         │                                        │
│  Plain Text  │    │         ▼                                        │
│  Dialogue    │    │  "### Instruction:\nExplain gravity\n\n          │
│              │    │   ### Response:\nGravity is a force...\n         │
└──────┬───────┘    │   ### End"                                       │
       │            │         │                                        │
       └───────────►│         ▼                                        │
                    │  Tokenizer (GPT-2 BPE)                          │
                    │  [21017, 46901, ...] → input_ids                │
                    │  [1, 1, 1, ...]      → attention_mask           │
                    │  [21017, 46901, ...] → labels                   │
                    │         │                                        │
                    │         ▼                                        │
                    │  Train/Val Split (90/10)                         │
                    │  train.jsonl (11,055 examples)                   │
                    │  validation.jsonl (1,229 examples)               │
                    └──────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Training Loop (train.py)                                           │
│                                                                     │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────────┐  │
│  │  Batch   │   │  Custom   │   │  Forward  │   │   Backward    │  │
│  │  Sampler │──►│  Collator │──►│  Pass     │──►│   + Optimizer │  │
│  │  (size=2)│   │  (pad)    │   │  (loss)   │   │   (AdamW)     │  │
│  └──────────┘   └───────────┘   └──────────┘   └───────────────┘  │
│                                                                     │
│  Effective Batch: 2 × 8 (grad accum) = 16                          │
│  LR Schedule: Linear warmup (200 steps) → Cosine decay             │
│  Mixed Precision: bf16 (A100 GPU)                                   │
│  Eval: Every 300 steps → Save best checkpoint (lowest eval_loss)    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Inference (generate.py)                                            │
│                                                                     │
│  User: "Explain gravity"                                            │
│         │                                                           │
│         ▼                                                           │
│  "### Instruction:\nExplain gravity\n\n### Response:\n"             │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  model.generate(temperature=0.4, top_p=0.9, top_k=50)  │        │
│  │  repetition_penalty=1.3, max_new_tokens=256             │        │
│  └─────────────────────────────────────────────────────────┘        │
│         │                                                           │
│         ▼                                                           │
│  Clean output: strip "### End" markers → Final response             │
└──────────────────────────────────────────────────────────────────────┘
```

### Configuration (training_config.yaml)

```
┌──────────────────────────────────────────────────────────┐
│                    training_config.yaml                   │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │   Model     │  │   LoRA     │  │     Training       │ │
│  │            │  │            │  │                    │ │
│  │  gpt2      │  │  rank: 64  │  │  epochs: 5        │ │
│  │  seq: 512  │  │  alpha:128 │  │  batch: 2         │ │
│  │  LoRA: yes │  │  drop: 0.1 │  │  grad_accum: 8    │ │
│  │            │  │  targets:  │  │  lr: 3e-4         │ │
│  │            │  │   c_attn   │  │  warmup: 200      │ │
│  │            │  │   c_proj   │  │  bf16: true       │ │
│  │            │  │   c_fc     │  │  eval: 300 steps  │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│                                                          │
│  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │    Data             │  │    Generation              │ │
│  │                    │  │                            │ │
│  │  raw: data/raw     │  │  temperature: 0.4         │ │
│  │  proc: data/proc   │  │  top_p: 0.9              │ │
│  │  split: 90/10      │  │  top_k: 50               │ │
│  │  format: instruct  │  │  rep_penalty: 1.3        │ │
│  └────────────────────┘  └────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
llm/
├── config/
│   └── training_config.yaml      # All hyperparameters and paths
├── data/
│   ├── raw/                      # Raw collected data (text files, CSVs, JSONs)
│   ├── processed/                # Cleaned and tokenized datasets
│   └── samples/                  # Example data format files
├── scripts/
│   ├── collect_data.py           # Data collection utilities
│   ├── preprocess_data.py        # Clean, format, and tokenize data
│   ├── train.py                  # Fine-tuning training loop
│   └── generate.py               # Inference / text generation
├── models/                       # Saved model checkpoints
├── logs/                         # Training logs
├── requirements.txt
└── README.md
```

## Data Collection Strategy

You need **educational text data** in one or more of these formats:

### 1. Instruction-Response Pairs (Best for Q&A)
```json
{"instruction": "Explain photosynthesis to a 5th grader", "response": "Photosynthesis is how plants make their own food using sunlight..."}
```

### 2. Plain Educational Text (Good for general knowledge)
```
Photosynthesis is the process by which green plants convert sunlight into chemical energy...
```

### 3. Multi-Turn Dialogue (Good for tutoring)
```json
{"conversation": [
  {"role": "student", "content": "What is gravity?"},
  {"role": "teacher", "content": "Gravity is a force that pulls objects toward each other..."},
  {"role": "student", "content": "Why don't we float away?"},
  {"role": "teacher", "content": "Earth's gravity is strong enough to keep us on the ground..."}
]}
```

### Where to Get Educational Data

| Source | Type | How to Use |
|--------|------|------------|
| **OpenStax** (openstax.org) | Free textbooks (CC licensed) | Download PDFs, extract text |
| **CK-12** (ck12.org) | K-12 textbooks | Scrape or use API |
| **Khan Academy** transcripts | Video transcripts | YouTube API for captions |
| **Wikipedia Simple English** | Simplified articles | Wikimedia dumps |
| **SQuAD / SciQ datasets** | Q&A pairs | HuggingFace datasets |
| **Your own school materials** | Lesson plans, worksheets | Place text files in data/raw/ |
| **Project Gutenberg** | Public domain educational books | Download text |

### Recommended Minimum Data

- **Instruction fine-tuning**: ~1,000–10,000 instruction-response pairs
- **Continued pre-training**: ~1M–50M tokens of educational text
- **Both combined**: Best results

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data in data/raw/ (see data/samples/ for format examples)

# 3. Preprocess the data
python scripts/preprocess_data.py

# 4. Fine-tune the model
python scripts/train.py

# 5. Generate text
python scripts/generate.py --prompt "Explain the water cycle"
```

## Hardware Requirements

| Model | VRAM | RAM | Training Time (10k samples) |
|-------|------|-----|----------------------------|
| GPT-2 Small (124M) | 4GB+ GPU or CPU | 8GB | ~1-2 hours (GPU) |
| GPT-2 Medium (355M) | 8GB+ GPU | 16GB | ~3-6 hours (GPU) |
| GPT-2 Large (774M) | 16GB+ GPU | 32GB | ~8-12 hours (GPU) |

CPU training works for GPT-2 Small but is ~10x slower. A free Google Colab GPU (T4) is sufficient for Small/Medium.

## License

This project is for educational and research purposes.
