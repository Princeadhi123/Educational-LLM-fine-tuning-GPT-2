# Educational LLM — Fine-Tuning GPT-2 on School Data

## Overview

This project fine-tunes a pre-trained GPT-2 model on educational content (textbooks, curricula, Q&A pairs, lesson plans) to create a specialized language model for school-level education. The model can then generate explanations, answer questions, summarize topics, and create quiz questions.

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
