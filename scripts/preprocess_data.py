"""
Data Preprocessing Pipeline for Educational LLM Fine-Tuning

This script:
1. Reads raw data files (JSONL, TXT) from data/raw/
2. Converts all data into a unified format
3. Tokenizes using the GPT-2 tokenizer
4. Creates train/validation splits
5. Saves processed datasets to data/processed/

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --config config/training_config.yaml
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer


def loadConfig(configPath: str = "config/training_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    with open(configPath, "r", encoding="utf-8") as configFile:
        return yaml.safe_load(configFile)


def formatInstructionExample(instruction: str, response: str) -> str:
    """Format an instruction-response pair into a training prompt."""
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response:\n{response.strip()}\n\n"
        f"### End\n"
    )


def formatConversationExample(conversation: list[dict]) -> str:
    """Format a multi-turn conversation into a training prompt."""
    formattedTurns = []
    for turn in conversation:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        formattedTurns.append(f"### {role}:\n{content}")
    return "\n\n".join(formattedTurns) + "\n\n### End\n"


def formatPlainTextExample(text: str, chunkSize: int = 1500) -> list[str]:
    """Split plain text into training-sized chunks with overlap."""
    text = text.strip()
    if len(text) < 50:
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    currentChunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(currentChunk) + len(paragraph) + 2 > chunkSize and currentChunk:
            chunks.append(currentChunk.strip())
            # Keep last paragraph as overlap for context continuity
            overlapSentences = currentChunk.split(".")[-2:]
            currentChunk = ".".join(overlapSentences).strip() + "\n\n"

        currentChunk += paragraph + "\n\n"

    if currentChunk.strip():
        chunks.append(currentChunk.strip())

    return chunks


def loadRawDataFiles(rawDirectory: str) -> list[str]:
    """Load all raw data files and convert to formatted training strings."""
    rawPath = Path(rawDirectory)
    if not rawPath.exists():
        print(f"ERROR: Raw data directory not found: {rawDirectory}")
        print("Run 'python scripts/collect_data.py' first to collect data.")
        sys.exit(1)

    allFormattedExamples = []
    jsonlFiles = list(rawPath.glob("*.jsonl"))
    textFiles = list(rawPath.glob("*.txt"))

    print(f"Found {len(jsonlFiles)} JSONL files and {len(textFiles)} text files in {rawDirectory}")

    # Process JSONL files
    for filePath in jsonlFiles:
        print(f"  Processing: {filePath.name}")
        with open(filePath, "r", encoding="utf-8") as inputFile:
            for lineNumber, line in enumerate(inputFile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"    [WARN] Invalid JSON at line {lineNumber} in {filePath.name}")
                    continue

                # Instruction-response format
                if "instruction" in entry and "response" in entry:
                    formatted = formatInstructionExample(entry["instruction"], entry["response"])
                    allFormattedExamples.append(formatted)

                # Conversation format
                elif "conversation" in entry:
                    formatted = formatConversationExample(entry["conversation"])
                    allFormattedExamples.append(formatted)

                # Plain text format (from Wikipedia, etc.)
                elif "text" in entry:
                    chunks = formatPlainTextExample(entry["text"])
                    allFormattedExamples.extend(chunks)

    # Process plain text files
    for filePath in textFiles:
        print(f"  Processing: {filePath.name}")
        try:
            content = filePath.read_text(encoding="utf-8")
            chunks = formatPlainTextExample(content)
            allFormattedExamples.extend(chunks)
        except Exception as error:
            print(f"    [ERROR] {filePath.name}: {error}")

    return allFormattedExamples


def tokenizeExamples(
    formattedExamples: list[str],
    tokenizer,
    maxSequenceLength: int,
) -> list[dict]:
    """Tokenize formatted text examples for training."""
    tokenizedDataset = []

    for example in tqdm(formattedExamples, desc="Tokenizing"):
        encoded = tokenizer(
            example,
            truncation=True,
            max_length=maxSequenceLength,
            padding=False,
            return_attention_mask=True,
        )

        # For causal LM, labels = input_ids (model learns to predict next token)
        tokenizedEntry = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["input_ids"].copy(),
        }
        tokenizedDataset.append(tokenizedEntry)

    return tokenizedDataset


def splitDataset(
    tokenizedDataset: list[dict],
    trainRatio: float = 0.9,
    randomSeed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split dataset into train and validation sets."""
    random.seed(randomSeed)
    shuffledIndices = list(range(len(tokenizedDataset)))
    random.shuffle(shuffledIndices)

    splitIndex = int(len(shuffledIndices) * trainRatio)
    trainIndices = shuffledIndices[:splitIndex]
    validationIndices = shuffledIndices[splitIndex:]

    trainSet = [tokenizedDataset[i] for i in trainIndices]
    validationSet = [tokenizedDataset[i] for i in validationIndices]

    return trainSet, validationSet


def saveProcessedDataset(dataset: list[dict], outputPath: Path) -> None:
    """Save tokenized dataset as JSONL."""
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outputPath, "w", encoding="utf-8") as outputFile:
        for entry in dataset:
            outputFile.write(json.dumps(entry) + "\n")
    print(f"  Saved {len(dataset)} examples to {outputPath}")


def computeDatasetStatistics(
    trainSet: list[dict],
    validationSet: list[dict],
    tokenizer,
) -> dict:
    """Compute and display dataset statistics."""
    trainTokenCounts = [len(entry["input_ids"]) for entry in trainSet]
    validationTokenCounts = [len(entry["input_ids"]) for entry in validationSet]

    totalTrainTokens = sum(trainTokenCounts)
    totalValidationTokens = sum(validationTokenCounts)

    statistics = {
        "trainExamples": len(trainSet),
        "validationExamples": len(validationSet),
        "totalTrainTokens": totalTrainTokens,
        "totalValidationTokens": totalValidationTokens,
        "averageTrainLength": totalTrainTokens / max(len(trainSet), 1),
        "averageValidationLength": totalValidationTokens / max(len(validationSet), 1),
        "maxTrainLength": max(trainTokenCounts) if trainTokenCounts else 0,
        "minTrainLength": min(trainTokenCounts) if trainTokenCounts else 0,
        "vocabularySize": tokenizer.vocab_size,
    }

    print(f"\n{'='*50}")
    print("Dataset Statistics")
    print(f"{'='*50}")
    print(f"  Train examples:       {statistics['trainExamples']:,}")
    print(f"  Validation examples:  {statistics['validationExamples']:,}")
    print(f"  Total train tokens:   {statistics['totalTrainTokens']:,}")
    print(f"  Avg tokens/example:   {statistics['averageTrainLength']:.1f}")
    print(f"  Max tokens/example:   {statistics['maxTrainLength']}")
    print(f"  Min tokens/example:   {statistics['minTrainLength']}")
    print(f"  Vocabulary size:      {statistics['vocabularySize']:,}")
    print(f"{'='*50}")

    return statistics


def main():
    parser = argparse.ArgumentParser(description="Preprocess educational data for fine-tuning")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="Path to training config YAML")
    arguments = parser.parse_args()

    config = loadConfig(arguments.config)
    modelConfig = config["model"]
    dataConfig = config["data"]

    rawDirectory = dataConfig["rawDirectory"]
    processedDirectory = dataConfig["processedDirectory"]
    trainSplitRatio = dataConfig["trainSplit"]
    maxSequenceLength = modelConfig["maxSequenceLength"]
    baseModelName = modelConfig["baseName"]

    # Load tokenizer
    print(f"Loading tokenizer for {baseModelName}...")
    tokenizer = AutoTokenizer.from_pretrained(baseModelName)

    # GPT-2 doesn't have a pad token by default; use EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format raw data
    print(f"\nLoading raw data from {rawDirectory}/...")
    formattedExamples = loadRawDataFiles(rawDirectory)

    if not formattedExamples:
        print("\nERROR: No data found!")
        print("Options:")
        print("  1. Run 'python scripts/collect_data.py' to download data")
        print("  2. Place your own .txt or .jsonl files in data/raw/")
        print("  3. Copy sample files: cp data/samples/* data/raw/")
        sys.exit(1)

    print(f"\nLoaded {len(formattedExamples)} formatted examples")

    # Tokenize
    print(f"\nTokenizing with max_length={maxSequenceLength}...")
    tokenizedDataset = tokenizeExamples(formattedExamples, tokenizer, maxSequenceLength)

    # Split
    trainSet, validationSet = splitDataset(tokenizedDataset, trainSplitRatio)

    # Save
    processedPath = Path(processedDirectory)
    print(f"\nSaving processed data to {processedDirectory}/...")
    saveProcessedDataset(trainSet, processedPath / "train.jsonl")
    saveProcessedDataset(validationSet, processedPath / "validation.jsonl")

    # Statistics
    computeDatasetStatistics(trainSet, validationSet, tokenizer)

    # Save statistics
    statisticsPath = processedPath / "dataset_stats.json"
    statistics = computeDatasetStatistics(trainSet, validationSet, tokenizer)
    with open(statisticsPath, "w", encoding="utf-8") as statsFile:
        json.dump(statistics, statsFile, indent=2)

    print(f"\nPreprocessing complete!")
    print(f"Next step: Run 'python scripts/train.py' to start fine-tuning.")


if __name__ == "__main__":
    main()
