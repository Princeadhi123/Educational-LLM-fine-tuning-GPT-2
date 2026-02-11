"""
Data Collection Utilities for Educational LLM Fine-Tuning

This script provides tools to collect educational text data from various sources:
1. Download free textbooks from OpenStax
2. Fetch Simple English Wikipedia articles
3. Download Q&A datasets from HuggingFace
4. Process local text files (PDFs, DOCX, TXT)
5. Scrape educational websites

Usage:
    python scripts/collect_data.py --source wikipedia --topics "photosynthesis,gravity,fractions"
    python scripts/collect_data.py --source huggingface --dataset "sciq"
    python scripts/collect_data.py --source local --input-dir "path/to/your/files"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

RAW_DATA_DIRECTORY = Path("data/raw")

WIKIMEDIA_HEADERS = {
    "User-Agent": "EducationalLLMBot/1.0 (https://github.com/educational-llm; educational-llm@example.com)",
}


def ensureDirectoryExists(directoryPath: Path) -> None:
    """Create directory if it doesn't exist."""
    directoryPath.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Simple English Wikipedia
# ─────────────────────────────────────────────

def fetchWikipediaArticle(topicName: str, session: requests.Session | None = None) -> dict | None:
    """Fetch a Simple English Wikipedia article by topic name."""
    apiUrl = "https://simple.wikipedia.org/w/api.php"
    parameters = {
        "action": "query",
        "titles": topicName,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }

    httpClient = session or requests

    try:
        response = httpClient.get(apiUrl, params=parameters, headers=WIKIMEDIA_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        for pageId, pageContent in pages.items():
            if pageId == "-1":
                print(f"  [SKIP] Article not found: {topicName}")
                return None
            articleText = pageContent.get("extract", "")
            if len(articleText.strip()) < 100:
                print(f"  [SKIP] Article too short: {topicName}")
                return None
            return {
                "source": "simple_wikipedia",
                "topic": topicName,
                "text": articleText.strip(),
            }
    except requests.RequestException as error:
        print(f"  [ERROR] Failed to fetch {topicName}: {error}")
        return None


def collectWikipediaArticles(topicList: list[str], outputFileName: str = "wikipedia_articles.jsonl") -> int:
    """Collect multiple Simple English Wikipedia articles."""
    ensureDirectoryExists(RAW_DATA_DIRECTORY)
    outputPath = RAW_DATA_DIRECTORY / outputFileName
    collectedCount = 0

    session = requests.Session()
    session.headers.update(WIKIMEDIA_HEADERS)

    print(f"\nFetching {len(topicList)} Wikipedia articles...")
    with open(outputPath, "w", encoding="utf-8") as outputFile:
        for topic in tqdm(topicList, desc="Wikipedia"):
            article = fetchWikipediaArticle(topic, session=session)
            if article:
                outputFile.write(json.dumps(article, ensure_ascii=False) + "\n")
                collectedCount += 1

    print(f"Saved {collectedCount} articles to {outputPath}")
    return collectedCount


# ─────────────────────────────────────────────
# 2. HuggingFace Datasets (SciQ, SQuAD, etc.)
# ─────────────────────────────────────────────

def collectHuggingFaceDataset(datasetName: str, outputFileName: str = None) -> int:
    """Download and convert a HuggingFace dataset to instruction format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install 'datasets' package: pip install datasets")
        return 0

    ensureDirectoryExists(RAW_DATA_DIRECTORY)
    if outputFileName is None:
        outputFileName = f"hf_{datasetName.replace('/', '_')}.jsonl"
    outputPath = RAW_DATA_DIRECTORY / outputFileName

    print(f"\nDownloading HuggingFace dataset: {datasetName}")

    datasetConverters = {
        "sciq": convertSciqToInstructions,
        "allenai/sciq": convertSciqToInstructions,
        "rajpurkar/squad": convertSquadToInstructions,
        "squad": convertSquadToInstructions,
    }

    converterFunction = datasetConverters.get(datasetName, convertGenericToInstructions)

    try:
        dataset = load_dataset(datasetName)
        trainSplit = dataset.get("train", dataset.get("validation"))
        if trainSplit is None:
            splitName = list(dataset.keys())[0]
            trainSplit = dataset[splitName]

        collectedCount = 0
        with open(outputPath, "w", encoding="utf-8") as outputFile:
            for example in tqdm(trainSplit, desc=f"Processing {datasetName}"):
                instructionPair = converterFunction(example)
                if instructionPair:
                    outputFile.write(json.dumps(instructionPair, ensure_ascii=False) + "\n")
                    collectedCount += 1

        print(f"Saved {collectedCount} examples to {outputPath}")
        return collectedCount

    except Exception as error:
        print(f"ERROR downloading {datasetName}: {error}")
        return 0


def convertSciqToInstructions(example: dict) -> dict | None:
    """Convert a SciQ dataset example to instruction format."""
    question = example.get("question", "").strip()
    correctAnswer = example.get("correct_answer", "").strip()
    supportText = example.get("support", "").strip()

    if not question or not correctAnswer:
        return None

    responseText = correctAnswer
    if supportText:
        responseText = f"{supportText}\n\nThe answer is: {correctAnswer}"

    return {"instruction": question, "response": responseText}


def convertSquadToInstructions(example: dict) -> dict | None:
    """Convert a SQuAD dataset example to instruction format."""
    question = example.get("question", "").strip()
    context = example.get("context", "").strip()
    answers = example.get("answers", {})

    answerTexts = answers.get("text", []) if isinstance(answers, dict) else []
    if not question or not answerTexts:
        return None

    instruction = f"Based on the following passage, answer the question.\n\nPassage: {context}\n\nQuestion: {question}"
    return {"instruction": instruction, "response": answerTexts[0]}


def convertGenericToInstructions(example: dict) -> dict | None:
    """Attempt to convert a generic dataset example to instruction format."""
    possibleInstructionKeys = ["question", "instruction", "input", "prompt", "query"]
    possibleResponseKeys = ["answer", "response", "output", "target", "completion"]

    instructionText = None
    for key in possibleInstructionKeys:
        if key in example and example[key]:
            instructionText = str(example[key]).strip()
            break

    responseText = None
    for key in possibleResponseKeys:
        if key in example and example[key]:
            responseText = str(example[key]).strip()
            break

    if instructionText and responseText:
        return {"instruction": instructionText, "response": responseText}
    return None


# ─────────────────────────────────────────────
# 3. Local File Processing
# ─────────────────────────────────────────────

def processLocalTextFiles(inputDirectory: str, outputFileName: str = "local_texts.jsonl") -> int:
    """Process local .txt files into the training format."""
    ensureDirectoryExists(RAW_DATA_DIRECTORY)
    inputPath = Path(inputDirectory)
    outputPath = RAW_DATA_DIRECTORY / outputFileName

    if not inputPath.exists():
        print(f"ERROR: Directory not found: {inputDirectory}")
        return 0

    textFiles = list(inputPath.glob("**/*.txt"))
    print(f"\nProcessing {len(textFiles)} text files from {inputDirectory}")

    collectedCount = 0
    with open(outputPath, "w", encoding="utf-8") as outputFile:
        for filePath in tqdm(textFiles, desc="Local files"):
            try:
                content = filePath.read_text(encoding="utf-8").strip()
                if len(content) < 50:
                    continue

                entry = {
                    "source": "local",
                    "filename": filePath.name,
                    "text": content,
                }
                outputFile.write(json.dumps(entry, ensure_ascii=False) + "\n")
                collectedCount += 1
            except Exception as error:
                print(f"  [ERROR] {filePath.name}: {error}")

    print(f"Saved {collectedCount} documents to {outputPath}")
    return collectedCount


def processLocalJsonlFiles(inputDirectory: str, outputFileName: str = "local_instructions.jsonl") -> int:
    """Process local .jsonl files that are already in instruction format."""
    ensureDirectoryExists(RAW_DATA_DIRECTORY)
    inputPath = Path(inputDirectory)
    outputPath = RAW_DATA_DIRECTORY / outputFileName

    if not inputPath.exists():
        print(f"ERROR: Directory not found: {inputDirectory}")
        return 0

    # Skip files generated by other collectors to avoid double-counting
    generatedFileNames = {"hf_sciq.jsonl", "wikipedia_articles.jsonl", "local_texts.jsonl", "local_instructions.jsonl"}
    allJsonlFiles = list(inputPath.glob("**/*.jsonl")) + list(inputPath.glob("**/*.json"))
    jsonlFiles = [f for f in allJsonlFiles if f.name not in generatedFileNames]
    print(f"\nProcessing {len(jsonlFiles)} JSON files from {inputDirectory} (skipping {len(allJsonlFiles) - len(jsonlFiles)} auto-generated files)")

    collectedCount = 0
    with open(outputPath, "w", encoding="utf-8") as outputFile:
        for filePath in tqdm(jsonlFiles, desc="JSON files"):
            try:
                with open(filePath, "r", encoding="utf-8") as inputFile:
                    for line in inputFile:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if "instruction" in entry and "response" in entry:
                            outputFile.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            collectedCount += 1
                        elif "text" in entry:
                            outputFile.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            collectedCount += 1
            except Exception as error:
                print(f"  [ERROR] {filePath.name}: {error}")

    print(f"Saved {collectedCount} entries to {outputPath}")
    return collectedCount


# ─────────────────────────────────────────────
# 4. Educational Topic List Generator
# ─────────────────────────────────────────────

SCHOOL_TOPICS = {
    "science": [
        "Photosynthesis", "Cell (biology)", "DNA", "Evolution", "Gravity",
        "Newton's laws of motion", "Electricity", "Magnetism", "Sound",
        "Light", "Atom", "Chemical reaction", "Periodic table",
        "Ecosystem", "Food chain", "Water cycle", "Rock cycle",
        "Solar System", "Planet", "Star", "Volcano", "Earthquake",
        "Weather", "Climate", "Human body", "Digestive system",
        "Respiratory system", "Circulatory system", "Nervous system",
    ],
    "math": [
        "Fraction (mathematics)", "Decimal", "Percentage", "Algebra",
        "Geometry", "Triangle", "Circle", "Area", "Volume",
        "Pythagorean theorem", "Equation", "Probability", "Statistics",
        "Mean", "Median", "Ratio", "Proportion",
        "Prime number", "Integer", "Exponentiation",
    ],
    "history": [
        "Ancient Egypt", "Ancient Greece", "Ancient Rome", "Middle Ages",
        "Renaissance", "Industrial Revolution", "World War I", "World War II",
        "Cold War", "American Revolution", "French Revolution",
        "Civil rights movement", "Democracy", "Constitution",
    ],
    "english": [
        "Noun", "Verb", "Adjective", "Adverb", "Pronoun",
        "Sentence (linguistics)", "Paragraph", "Essay", "Poetry",
        "Metaphor", "Simile", "Alliteration", "Narrative",
        "Fiction", "Non-fiction",
    ],
    "geography": [
        "Continent", "Ocean", "Mountain", "River", "Desert",
        "Rainforest", "Tundra", "Climate zone", "Latitude",
        "Longitude", "Map", "Population", "Migration",
    ],
}


def getAllSchoolTopics() -> list[str]:
    """Return a flat list of all school topics across subjects."""
    allTopics = []
    for subjectTopics in SCHOOL_TOPICS.values():
        allTopics.extend(subjectTopics)
    return allTopics


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect educational data for LLM fine-tuning")
    parser.add_argument("--source", choices=["wikipedia", "huggingface", "local", "all"],
                        default="all", help="Data source to collect from")
    parser.add_argument("--topics", type=str, default=None,
                        help="Comma-separated list of Wikipedia topics (or 'school' for all school topics)")
    parser.add_argument("--dataset", type=str, default="sciq",
                        help="HuggingFace dataset name")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing local text files")

    arguments = parser.parse_args()
    totalCollected = 0

    if arguments.source in ("wikipedia", "all"):
        if arguments.topics == "school" or arguments.topics is None:
            topicList = getAllSchoolTopics()
        else:
            topicList = [topic.strip() for topic in arguments.topics.split(",")]
        totalCollected += collectWikipediaArticles(topicList)

    if arguments.source in ("huggingface", "all"):
        totalCollected += collectHuggingFaceDataset(arguments.dataset)

    if arguments.source in ("local", "all"):
        inputDirectory = arguments.input_dir or "data/raw"
        totalCollected += processLocalTextFiles(inputDirectory)
        totalCollected += processLocalJsonlFiles(inputDirectory)

    print(f"\n{'='*50}")
    print(f"Total collected: {totalCollected} entries")
    print(f"Data saved to: {RAW_DATA_DIRECTORY}/")
    print(f"{'='*50}")
    print("\nNext step: Run 'python scripts/preprocess_data.py' to prepare the data for training.")


if __name__ == "__main__":
    main()
