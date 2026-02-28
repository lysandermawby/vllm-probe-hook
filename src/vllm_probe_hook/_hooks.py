"""
Hook management for hidden state extraction.

Registers PyTorch forward hooks on transformer layers of a vLLM-loaded model.
Each hook fires once per forward pass (both prefill and decode) and stores the
hidden states for each active sequence, keyed by request ID.

The hooks store raw hidden states (no dot product) as 2-D tensors of shape
[1, hidden_size] per decode token, accumulated in _activation_store.

Prefill phase:
    vLLM processes the entire prompt in a single forward pass.  The hook
    fires with hidden_states of shape [prompt_len, hidden_size].  We skip
    these tokens (not user-visible output) by consulting a set of request IDs
    that are still in prefill.

Decode phase:
    Each step generates exactly one token per sequence.  The hook fires with
    hidden_states of shape [batch_size, hidden_size] where batch_size equals
    the number of concurrently decoding sequences.  We slice out the row
    belonging to each request and store it.
"""
from __future__ import annotations

import logging
import threading
import traceback

import torch

from vllm_probe_hook import _activation_store

# Hook handles: {layer_id: hook_handle}
_hook_handles: dict[int, object] = {}
_hooks_lock = threading.Lock()

# Request IDs in prefill phase — skip hook results for these
_prefill_ids: set[str] = set()
_prefill_lock = threading.Lock()

# Module-level flags controlling what to capture
_include_prefill: bool = False
_include_decode: bool = True


def configure(include_prefill: bool = False, include_decode: bool = True) -> None:
    """Set module-level capture flags.  Called from LLM.__init__."""
    global _include_prefill, _include_decode
    _include_prefill = include_prefill
    _include_decode = include_decode


def mark_prefill(request_id: str) -> None:
    with _prefill_lock:
        _prefill_ids.add(request_id)


def unmark_prefill(request_id: str) -> None:
    with _prefill_lock:
        _prefill_ids.discard(request_id)


def _get_model(engine):
    """Navigate to the torch.nn.Module for the LLM from a vLLM engine object."""
    # vLLM 0.6.6: AsyncLLMEngine has .engine, vllm.LLM has .llm_engine
    llm_engine = (
        getattr(engine, "engine", None)
        or getattr(engine, "llm_engine", None)
        or engine
    )
    model = llm_engine.model_executor.driver_worker.model_runner.model
    return model


def _get_layer(model, layer_id: int) -> torch.nn.Module:
    """Return the transformer layer module at the given index.

    Handles two common vLLM model layouts:
    - ``model.model.layers[i]`` — standard Llama/Mistral layout where the
      top-level model wraps an inner ``model`` attribute.
    - ``model.layers[i]`` — flatter architectures where layers live directly
      on the top-level module.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_id]
    if hasattr(model, "layers"):
        return model.layers[layer_id]
    raise AttributeError(f"Cannot find .layers on model of type {type(model)}")


def _get_running_request_ids(engine) -> list[str]:
    """Return request IDs of currently running (decoding) sequences in order."""
    llm_engine = (
        getattr(engine, "engine", None)
        or getattr(engine, "llm_engine", None)
        or engine
    )
    request_ids = []
    scheduler = llm_engine.scheduler
    # vLLM 0.6.6: scheduler is a list with one element
    if isinstance(scheduler, (list, tuple)):
        scheduler = scheduler[0]
    for seq_group in scheduler.running:
        for seq in seq_group.seqs:
            if not seq.is_finished():
                request_ids.append(seq_group.request_id)
    return request_ids


def register_hooks(engine, layer_ids: list[int]) -> None:
    """Register one forward hook per requested layer on the given engine's model.

    Safe to call multiple times; existing hooks for the same layer IDs are kept.
    """
    model = _get_model(engine)

    with _hooks_lock:
        for layer_id in layer_ids:
            if layer_id in _hook_handles:
                continue  # already registered

            layer = _get_layer(model, layer_id)
            # Capture layer_id in closure
            _hook_handles[layer_id] = layer.register_forward_hook(
                _make_hook(engine, layer_id)
            )


def remove_hooks() -> None:
    """Remove all registered hooks."""
    with _hooks_lock:
        for handle in _hook_handles.values():
            try:
                handle.remove()
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to remove hook handle %r: %s", handle, e
                )
        _hook_handles.clear()


def _make_hook(engine, layer_id: int):
    """Return a forward hook closure for the given layer.

    The closure conforms to the standard PyTorch hook signature
    ``(module, input, output)`` and is registered via
    ``layer.register_forward_hook()``.  On each forward pass it reads the
    currently running request IDs from the vLLM scheduler, slices out the
    hidden state for each decoding sequence, and appends it to
    ``_activation_store``.
    """

    def hook(_module, _input, output):
        # Standard PyTorch hook signature; _module and _input unused.
        try:
            # output is (hidden_states, residual) in vLLM's LlamaDecoderLayer
            # Reconstruct post-residual hidden states
            if isinstance(output, tuple) and len(output) == 2:
                hidden_states, residual = output
                # residual may be None in some architectures
                if residual is not None:
                    hidden_states = hidden_states + residual
            else:
                hidden_states = output

            # hidden_states: [seq_positions, hidden_size]
            # Could be either a flat [total_tokens, H] prefill/batch tensor or
            # a [batch_size, H] decode tensor.  We determine which by asking the
            # scheduler which requests are running.
            running_ids = _get_running_request_ids(engine)
            if not running_ids:
                return

            with _prefill_lock:
                skip_ids = set(_prefill_ids)

            n_running = len(running_ids)
            n_tokens = hidden_states.shape[0]

            if n_tokens == n_running:
                # Decode step: one token per running sequence
                if _include_decode:
                    for i, request_id in enumerate(running_ids):
                        if request_id in skip_ids:
                            continue
                        token_hidden = hidden_states[i : i + 1]  # [1, H]
                        _activation_store.append(request_id, layer_id, token_hidden)
            elif _include_prefill:
                # Prefill step: n_tokens != n_running.
                # Identify request(s) still in prefill phase.
                prefilling = [rid for rid in running_ids if rid in skip_ids]
                if len(prefilling) == 1:
                    _activation_store.append_prefill(
                        prefilling[0], layer_id, hidden_states
                    )
                elif len(prefilling) > 1:
                    logging.getLogger(__name__).debug(
                        "Multiple concurrent prefills (%d) on layer %d — skipping.",
                        len(prefilling), layer_id,
                    )
            # else: prefill step but _include_prefill is False — skip.

        except Exception as e:
            logging.getLogger(__name__).debug(
                "Hidden state hook error on layer %d: %s\n%s",
                layer_id, e, traceback.format_exc(),
            )

    return hook
