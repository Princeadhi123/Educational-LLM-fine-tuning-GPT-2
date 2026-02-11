"""
Text Generation / Inference Script for Educational LLM

This script loads a fine-tuned model and generates educational text
based on user prompts. Supports both interactive mode and single-prompt mode.

Usage:
    python scripts/generate.py --prompt "Explain photosynthesis to a 5th grader"
    python scripts/generate.py --interactive
    python scripts/generate.py --prompt "What is gravity?" --model models/checkpoints/merged
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def loadConfig(configPath: str = "config/training_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(configPath, "r", encoding="utf-8") as configFile:
        return yaml.safe_load(configFile)


def loadModel(modelPath: str, baseModelName: str = "gpt2"):
    """Load the fine-tuned model and tokenizer."""
    modelDirectory = Path(modelPath)

    if not modelDirectory.exists():
        print(f"ERROR: Model not found at {modelPath}")
        print("Run 'python scripts/train.py' first to fine-tune a model.")
        sys.exit(1)

    print(f"Loading model from {modelPath}...")

    # Check if this is a LoRA adapter or a full model
    adapterConfigPath = modelDirectory / "adapter_config.json"
    isLoraAdapter = adapterConfigPath.exists()

    if isLoraAdapter:
        print("  Detected LoRA adapter, loading base model + adapter...")
        try:
            from peft import PeftModel
        except ImportError:
            print("ERROR: Install 'peft' package: pip install peft")
            sys.exit(1)

        tokenizer = AutoTokenizer.from_pretrained(modelPath)
        baseModel = AutoModelForCausalLM.from_pretrained(baseModelName)
        model = PeftModel.from_pretrained(baseModel, modelPath)
        model = model.merge_and_unload()
    else:
        print("  Loading full model...")
        tokenizer = AutoTokenizer.from_pretrained(modelPath)
        model = AutoModelForCausalLM.from_pretrained(modelPath)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Move to best available device
    if torch.cuda.is_available():
        model = model.cuda()
        deviceLabel = f"GPU ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
        deviceLabel = "Apple MPS"
    else:
        deviceLabel = "CPU"

    print(f"  Model loaded on {deviceLabel}")
    return model, tokenizer


def formatPromptForGeneration(userPrompt: str) -> str:
    """Format a user prompt into the instruction template used during training."""
    return f"### Instruction:\n{userPrompt.strip()}\n\n### Response:\n"


def generateResponse(
    model,
    tokenizer,
    userPrompt: str,
    generationConfig: dict,
) -> str:
    """Generate a response for a given prompt."""
    formattedPrompt = formatPromptForGeneration(userPrompt)

    inputTokens = tokenizer(
        formattedPrompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Move inputs to same device as model
    deviceOfModel = next(model.parameters()).device
    inputTokens = {key: value.to(deviceOfModel) for key, value in inputTokens.items()}

    with torch.no_grad():
        outputTokens = model.generate(
            **inputTokens,
            max_new_tokens=generationConfig.get("maxNewTokens", 256),
            temperature=generationConfig.get("temperature", 0.7),
            top_p=generationConfig.get("topP", 0.9),
            top_k=generationConfig.get("topK", 50),
            repetition_penalty=generationConfig.get("repetitionPenalty", 1.2),
            do_sample=generationConfig.get("doSample", True),
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    promptLength = inputTokens["input_ids"].shape[1]
    generatedTokens = outputTokens[0][promptLength:]
    generatedText = tokenizer.decode(generatedTokens, skip_special_tokens=True)

    # Clean up: stop at "### End" or "### Instruction" markers
    stopMarkers = ["### End", "### Instruction", "###"]
    for marker in stopMarkers:
        markerPosition = generatedText.find(marker)
        if markerPosition != -1:
            generatedText = generatedText[:markerPosition]

    return generatedText.strip()


def runInteractiveMode(model, tokenizer, generationConfig: dict) -> None:
    """Run an interactive chat loop."""
    print("\n" + "=" * 60)
    print("  Educational LLM — Interactive Mode")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            userInput = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not userInput:
            continue
        if userInput.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        response = generateResponse(model, tokenizer, userInput, generationConfig)
        print(response)
        print()


def runSinglePrompt(model, tokenizer, prompt: str, generationConfig: dict) -> None:
    """Generate a single response and print it."""
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    response = generateResponse(model, tokenizer, prompt, generationConfig)
    print(f"Response:\n{response}")


def resolveModelPath(explicitPath: str | None, config: dict) -> str:
    """Determine which model path to use."""
    if explicitPath:
        return explicitPath

    # Try merged model first, then final, then latest checkpoint
    candidatePaths = [
        Path(config["training"]["outputDirectory"]) / "merged",
        Path(config["training"]["outputDirectory"]) / "final",
        Path(config["training"]["outputDirectory"]),
    ]

    for candidatePath in candidatePaths:
        if candidatePath.exists() and any(candidatePath.iterdir()):
            return str(candidatePath)

    # Fall back to base model (no fine-tuning)
    print("WARNING: No fine-tuned model found. Using base model.")
    return config["model"]["baseName"]


def main():
    parser = argparse.ArgumentParser(description="Generate text with the fine-tuned educational LLM")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to generate a response for")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive chat mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override generation temperature")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max new tokens")
    arguments = parser.parse_args()

    if not arguments.prompt and not arguments.interactive:
        print("Please specify --prompt or --interactive")
        parser.print_help()
        sys.exit(1)

    # Load config
    config = loadConfig(arguments.config)
    generationConfig = config["generation"]

    # Apply CLI overrides
    if arguments.temperature is not None:
        generationConfig["temperature"] = arguments.temperature
    if arguments.max_tokens is not None:
        generationConfig["maxNewTokens"] = arguments.max_tokens

    # Resolve and load model
    modelPath = resolveModelPath(arguments.model, config)
    baseModelName = config["model"]["baseName"]
    model, tokenizer = loadModel(modelPath, baseModelName)

    # Run
    if arguments.interactive:
        runInteractiveMode(model, tokenizer, generationConfig)
    else:
        runSinglePrompt(model, tokenizer, arguments.prompt, generationConfig)


if __name__ == "__main__":
    main()
