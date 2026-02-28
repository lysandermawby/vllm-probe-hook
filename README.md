# vLLM Probe

Access hidden states from models at runtime. Combine vLLM's speed and efficiency with activation service.

## Quickstart

To install this as an editable package, first clone the repository locally.

```bash
git clone https://www.github.com/lysandermawby/vllm-probe-hook.git
```

Then run the following command to install it as an editable package, changing '.' for your local file path to this repo:

```bash
# change '.' for your local file path to the directory
pip install -e .
```

Alternatively, if you prefer to use [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e .
```

## Usage Guide

This is designed to be as close to a drop-in replacement for vLLM as possible. Additional methods which allow hidden states to be saved during generation or streamed token-by-token are added on top of the core vLLM functionality.

There are some restrictions on this implementation which prevents it from being a truly general replacement import:
1. **vLLM v0.6.6 is pinned:** The hooks registered to make this possible are only available on some versions of vLLM. This also means that only features in vLLM's depreciated engine v0 are available, and the `transformers` package must be downgraded to a version below 5.0.0.
2. **Only one LLM instance per process:** Although this is not a particularly common use case, vLLM typically allows for multiple LLM instances in a single python process. Here this would result in a mixing of hidden state values between generations rendering them unusable.
3. **enforce_eager is always True:** The enforce_eager option which disables optimisations and generally harms performance must be set to True.

Both `vllm_probe_hook` and `vllm` imports can be used in the same script, but only importing from `vllm_probe_hook` will also work.

### Examples

See the following example of a batched query where hidden states are captured during inference and can be retrieved with other information about the model output.

```python
from vllm_probe_hook import LLM, SamplingParams  # SamplingParams is re-exported from vllm

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    layers=[15, 20],  # specify which layers to collect activations from
    max_model_len=4096,
    gpu_memory_utilization=0.7,
)
sampling_params = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=512)
outputs = llm.generate(
    ["What is the capital of France?", "What is 2 to the power of 12?"],
    sampling_params,
)

for output in outputs:
    # standard vLLM API is unchanged
    print(output.outputs[0].text)

    # hidden states are available as a dict mapping layer index to a torch.Tensor
    print(output.hidden_states[15].shape)  # e.g. torch.Size([num_output_tokens, model_dimension (d_model)])
    print(output.hidden_states[20].shape)
```

For asynchronous token-by-token streaming, `AsyncLLM` yields a `RequestOutput` after each token:

> **Note:** the async API has not yet been fully tested. The interface is correct but treat this as experimental.

```python
import asyncio
from vllm_probe_hook import AsyncLLM, SamplingParams

async def main():
    llm = AsyncLLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        layers=[15, 20],
        max_model_len=4096,
    )
    sampling_params = SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=512)

    async for output in llm.generate("What is the capital of France?", sampling_params):
        # partial: hidden_states contains whatever tokens have arrived so far
        print(f"tokens so far: {len(output.outputs[0].token_ids)}", end="\r")
        final = output

    # final output has complete hidden states for all generated tokens
    print(final.outputs[0].text)
    print(final.hidden_states[15].shape)  # torch.Size([num_tokens, hidden_size])

asyncio.run(main())
```

For more, see the `./examples/` directory.

## Available Options

This repository makes it possible to run vLLM while grabbing hidden states from the prefill and decode stage.

The LLM object at initialisation accepts the following relevant new arguments (with their defaults shown) to affect it's behaviour:

```bash
layers: list[int] | list[str], 
include_prefill: bool = False,
include_decode: bool = True,
last_prefill_token: bool = False,
last_decode_token: bool = False,
```

Their usage is as follows:
- **layers:** Either a list of integers representing the indices of layers where you would like to capture the residual stream *after the layer*, or the "all" string indicating that all layers should be captured. Note that the residual stream will be captured after the layer has completed it's operation.
- **include_prefill:** Capture hidden states during the prefill stage. These states are accessible as the `prefill_hidden` attribute on the RequestOutput object. Defaults to false.
- **include_decode:** Capture hidden states during the decode stage (default behaviour). These states are accessible as the `hidden` attribute on the RequestOutput object (see the `./examples/generate_hidden_states.py` file).
- **last_prefill_token:** Special option to only return the activations on the final token of the pre-fill stage. Bear in mind that the default behaviour is to return tensors with the hidden states of all tokens received in the prefill stage.
- **last_decode_token:** Special option to only return the activations on the final token of the decode stage. The default behaviour is to return tensors with the hidden states of all tokens generated during the decode stage.

## Implementation Details

To leverage the speed of vLLM while still storing intermediate model activations for future use, this approach uses a pytorch hook attached to the underlying model object.

This uses an unintended feature of certain older versions of vLLM (this version using v0.6.6) where a bare pytorch model object is exposed at `vllm.LLM.llm_engine.model_executor.driver_worker.model_runner.model`. As this is a regular pytorch model, a forward hook can be registed to save activations at a particular layer during inference. 

Being precise, the forward hook is attached to the target layer `target_layer = model.model.layers[probe_layer_idx]`, and you can then attach a hook to save activations in the standard way using `target_layer.register_forward_hook(activation_hook)`.

Modifying this forward hook can allow values of linear probes to be obtained at runtime with minimal added latency. 

## Improvements and Limitations

This only supports particular aged versions of vLLM, which is not appropriate for such a fast-improving project.

As of now, the entire hidden state is collected and stored, and can be made available either at the end of generation or streamed token-by-token.
