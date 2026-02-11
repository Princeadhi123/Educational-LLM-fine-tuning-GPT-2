"""
Fine-Tuning Training Script for Educational LLM

This script fine-tunes a pre-trained GPT-2 model on preprocessed educational data
using either full fine-tuning or LoRA (parameter-efficient fine-tuning).

Features:
- LoRA support via PEFT for memory-efficient training
- Gradient accumulation for larger effective batch sizes
- TensorBoard logging
- Checkpoint saving and resumption
- Mixed precision training (fp16/bf16)

Usage:
    python scripts/train.py
    python scripts/train.py --config config/training_config.yaml
    python scripts/train.py --resume models/checkpoints/checkpoint-500
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


class EducationalDataset(Dataset):
    """PyTorch Dataset for tokenized educational data."""

    def __init__(self, dataFilePath: str):
        self.examples = []
        with open(dataFilePath, "r", encoding="utf-8") as dataFile:
            for line in dataFile:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"  Loaded {len(self.examples)} examples from {dataFilePath}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
        }


def loadConfig(configPath: str = "config/training_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    with open(configPath, "r", encoding="utf-8") as configFile:
        return yaml.safe_load(configFile)


def setupLoraModel(model, loraConfig: dict):
    """Apply LoRA adapters to the model for parameter-efficient fine-tuning."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("ERROR: Install 'peft' package: pip install peft")
        sys.exit(1)

    loraConfiguration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=loraConfig["rank"],
        lora_alpha=loraConfig["alpha"],
        lora_dropout=loraConfig["dropout"],
        target_modules=loraConfig["targetModules"],
        bias="none",
    )

    model = get_peft_model(model, loraConfiguration)

    trainableParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    totalParameters = sum(p.numel() for p in model.parameters())
    trainablePercentage = 100 * trainableParameters / totalParameters

    print(f"\n  LoRA enabled:")
    print(f"    Trainable parameters: {trainableParameters:,} ({trainablePercentage:.2f}%)")
    print(f"    Total parameters:     {totalParameters:,}")
    print(f"    Rank: {loraConfig['rank']}, Alpha: {loraConfig['alpha']}")

    return model


def createTrainingArguments(trainingConfig: dict) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    outputDirectory = trainingConfig["outputDirectory"]
    loggingDirectory = trainingConfig["loggingDirectory"]

    Path(outputDirectory).mkdir(parents=True, exist_ok=True)
    Path(loggingDirectory).mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=outputDirectory,
        num_train_epochs=trainingConfig["numberOfEpochs"],
        per_device_train_batch_size=trainingConfig["batchSize"],
        per_device_eval_batch_size=trainingConfig["batchSize"],
        gradient_accumulation_steps=trainingConfig["gradientAccumulationSteps"],
        learning_rate=trainingConfig["learningRate"],
        weight_decay=trainingConfig["weightDecay"],
        warmup_steps=trainingConfig["warmupSteps"],
        max_grad_norm=trainingConfig["maxGradientNorm"],
        fp16=trainingConfig["fp16"],
        bf16=trainingConfig["bf16"],
        logging_dir=loggingDirectory,
        logging_steps=trainingConfig["loggingSteps"],
        save_steps=trainingConfig["saveSteps"],
        eval_steps=trainingConfig["evaluationSteps"],
        eval_strategy="steps",
        save_total_limit=trainingConfig["saveTotal"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on educational data")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    arguments = parser.parse_args()

    # Load configuration
    config = loadConfig(arguments.config)
    modelConfig = config["model"]
    loraConfig = config["lora"]
    dataConfig = config["data"]
    trainingConfig = config["training"]

    baseModelName = modelConfig["baseName"]
    maxSequenceLength = modelConfig["maxSequenceLength"]
    useLoRA = modelConfig["useLoRA"]
    processedDirectory = dataConfig["processedDirectory"]

    # Check for processed data
    trainDataPath = Path(processedDirectory) / "train.jsonl"
    validationDataPath = Path(processedDirectory) / "validation.jsonl"

    if not trainDataPath.exists():
        print("ERROR: Processed training data not found!")
        print("Run 'python scripts/preprocess_data.py' first.")
        sys.exit(1)

    # Detect device
    if torch.cuda.is_available():
        deviceName = torch.cuda.get_device_name(0)
        print(f"Using GPU: {deviceName}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        print("Using CPU (training will be slow)")
        print("TIP: Use Google Colab for free GPU access")

    # Load tokenizer
    print(f"\nLoading tokenizer: {baseModelName}")
    tokenizer = AutoTokenizer.from_pretrained(baseModelName)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {baseModelName}")
    model = AutoModelForCausalLM.from_pretrained(
        baseModelName,
        torch_dtype=torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA if configured
    if useLoRA:
        model = setupLoraModel(model, loraConfig)

    # Load datasets
    print(f"\nLoading datasets...")
    trainDataset = EducationalDataset(str(trainDataPath))
    validationDataset = EducationalDataset(str(validationDataPath))

    # Data collator handles padding within batches
    dataCollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    trainingArguments = createTrainingArguments(trainingConfig)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=trainingArguments,
        train_dataset=trainDataset,
        eval_dataset=validationDataset,
        data_collator=dataCollator,
        tokenizer=tokenizer,
    )

    # Train
    print(f"\n{'='*50}")
    print("Starting fine-tuning...")
    print(f"  Model:          {baseModelName}")
    print(f"  LoRA:           {'Yes' if useLoRA else 'No'}")
    print(f"  Epochs:         {trainingConfig['numberOfEpochs']}")
    print(f"  Batch size:     {trainingConfig['batchSize']}")
    print(f"  Grad accum:     {trainingConfig['gradientAccumulationSteps']}")
    print(f"  Effective batch: {trainingConfig['batchSize'] * trainingConfig['gradientAccumulationSteps']}")
    print(f"  Learning rate:  {trainingConfig['learningRate']}")
    print(f"  Train examples: {len(trainDataset)}")
    print(f"  Val examples:   {len(validationDataset)}")
    print(f"{'='*50}\n")

    resumeCheckpoint = arguments.resume
    trainer.train(resume_from_checkpoint=resumeCheckpoint)

    # Save final model
    finalModelDirectory = Path(trainingConfig["outputDirectory"]) / "final"
    finalModelDirectory.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving final model to {finalModelDirectory}...")
    if useLoRA:
        # Save LoRA adapters
        model.save_pretrained(str(finalModelDirectory))
        tokenizer.save_pretrained(str(finalModelDirectory))
        print("  Saved LoRA adapters (small file size)")

        # Also save merged model for easy inference
        mergedDirectory = Path(trainingConfig["outputDirectory"]) / "merged"
        mergedDirectory.mkdir(parents=True, exist_ok=True)
        print(f"  Merging LoRA weights into base model...")
        mergedModel = model.merge_and_unload()
        mergedModel.save_pretrained(str(mergedDirectory))
        tokenizer.save_pretrained(str(mergedDirectory))
        print(f"  Saved merged model to {mergedDirectory}")
    else:
        trainer.save_model(str(finalModelDirectory))
        tokenizer.save_pretrained(str(finalModelDirectory))

    print(f"\nTraining complete!")
    print(f"Next step: Run 'python scripts/generate.py --prompt \"Explain gravity\"'")


if __name__ == "__main__":
    main()
