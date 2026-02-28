#!/usr/bin/env python
"""
Example: extract hidden states while generating with vllm_probe_hook.

Loads a small slice of the LongFact dataset, runs batched generation with
Llama-3.1-8B-Instruct, and saves per-token hidden states (layers 15 and 20)
to a JSONL file.

Install the extra deps needed to run this example:
    pip install "vllm-probe-hook[examples]"

Then run:
    python generate_hidden_states.py

Requires HF_TOKEN in your environment (or a .env file) because
Meta-Llama-3.1-8B-Instruct is a gated model.
"""

from pathlib import Path
import json
import os

import click
from datasets import load_dataset
from dotenv import load_dotenv
from vllm import SamplingParams

from vllm_probe_hook import LLM


def display_outputs(outputs):
    """Print a summary of each output to the terminal."""
    for i, output in enumerate(outputs):
        hidden = output.hidden_states
        print("=" * 50)
        print(f"Prompt {i}: {output.prompt[:80]}...")
        print(f"Generated text: {output.outputs[0].text}")
        layers_extracted = hidden.keys()
        print(f"Layers extracted: {layers_extracted}")
        for layer in layers_extracted:
            print(f"Layer {layer} hidden states shape: {hidden[layer].shape}")


def save_outputs(outputs, filepath: Path, all_tokens: bool = True):
    """Save outputs to a JSONL file.

    Args:
        outputs: List of RequestOutput from LLM.generate().
        filepath: Destination .jsonl file.
        all_tokens: If True, save all token hidden states.
               If False, save only the final token's hidden state per layer (saves ~50-500x space depending on output length).
    """
    if filepath.suffix != ".jsonl":
        print(f"Warning: {filepath} does not have a .jsonl extension.")

    with open(filepath, "w") as f:
        for output in outputs:
            hidden_states = {}
            for layer, hs in output.hidden_states.items():
                data = hs.float().numpy().tolist()
                hidden_states[layer] = data if all_tokens else data[-1]

            record = {
                "prompt": output.prompt,
                "text": output.outputs[0].text,
                "token_ids": list(output.outputs[0].token_ids),
                "hidden_states": hidden_states,
            }
            f.write(json.dumps(record) + "\n")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("-n", "--num-prompts", default=5, show_default=True, help="Number of prompts to generate.")
@click.option("-o", "--output", default="output_data.jsonl", show_default=True, help="Output JSONL file path.")
@click.option("-a", "--all-tokens", is_flag=True, default=False, help="Save hidden states for every token (warning: very large files).")
@click.option("-m", "--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="path to model or huggingface ID e.g. /pool/models/Llama-3.1-8B-Instruct")
def main(num_prompts: int, output: str, all_tokens: bool, model: str):
    load_dotenv()

    # Show cache locations to help debug OOM issues on hosted hardware.
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "UV_CACHE_DIR"):
        print(f"{var} = {os.getenv(var)}")

    llm = LLM(
        model,
        gpu_memory_utilization=0.7,
        max_model_len=1024,
        # enforce_eager=True is set automatically by vllm_probe_hook (required for hooks).
        layers=[15, 20],
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
    )

    ds = load_dataset("claserken/longfact") # load the LongFact dataset, can replace this with a list of prompts
    prompt_list = ds["train"].to_pandas()["prompt"].tolist()

    outputs = llm.generate(
        prompts=prompt_list[:num_prompts],
        sampling_params=sampling_params,
    )

    display_outputs(outputs)
    save_outputs(outputs, filepath=Path(output), all_tokens=all_tokens)
    print(f"Saved {len(outputs)} records to {output}")


if __name__ == "__main__":
    main()
