"""
vllm_probe_hook.LLM — drop-in replacement for vllm.LLM using PyTorch hooks.

Uses vLLM 0.6.6 with enforce_eager=True and forward hooks on transformer layers
to extract hidden states without any model patching or speculator tricks.

Usage:
    from vllm_probe_hook import LLM
    llm = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", layers=[30])
    outputs = llm.generate(["Tell me about Paris"])
    print(outputs[0].outputs[0].text)        # standard vLLM API
    print(outputs[0].hidden_states[30].shape) # {30: Tensor[num_tokens, hidden_size]}

The returned RequestOutput is a subclass of vllm.RequestOutput with an added
.hidden_states attribute, so all existing vLLM result handling works unchanged.

The hidden_states attribute is a dictionary mapping layer indices to a tensor of all stored activations.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from typing import AsyncGenerator, Optional

import torch


class RequestOutput:
    """Placeholder; replaced at runtime when vllm is available.

    Defined here so the module is importable without vllm installed.
    The real class is built in _make_request_output_class() and cached in
    _RequestOutputClass.
    """


_RequestOutputClass: type | None = None


def _get_request_output_class() -> type:
    """Return (and cache) the vllm.RequestOutput subclass."""
    global _RequestOutputClass, RequestOutput
    if _RequestOutputClass is None:
        import vllm
        class _RequestOutput(vllm.RequestOutput):
            """vllm.RequestOutput extended with per-layer hidden states."""
        _RequestOutput.__name__ = "RequestOutput"
        _RequestOutput.__qualname__ = "RequestOutput"
        _RequestOutputClass = _RequestOutput
        RequestOutput = _RequestOutput
    return _RequestOutputClass


def _attach_hidden_states(
    vllm_out,
    hidden: dict[int, torch.Tensor],
):
    """Promote a plain vllm.RequestOutput to our RequestOutput subclass in-place."""
    cls = _get_request_output_class()
    vllm_out.__class__ = cls
    vllm_out.hidden_states = hidden
    return vllm_out


class LLM:
    """Wrapper around vllm.LLM that extracts per-layer hidden states via hooks.

    Note: only one LLM (or AsyncLLM) instance should exist per process.
    Hook state is module-level, so multiple instances will interfere with each other. Known limitation.

    Args:
        model: HuggingFace model name or path.
        layers: List of layer indices to extract hidden states from.
        dtype: Model dtype (default "bfloat16").
        **vllm_kwargs: Additional keyword arguments forwarded to vllm.LLM.
            Note: enforce_eager=True is always set (required for hooks).
    """

    def __init__(
        self,
        model: str,
        layers: list[int],
        dtype: str = "bfloat16",
        **vllm_kwargs,
    ):
        import vllm

        self._layers = sorted(layers)
        self._model_name = model

        # enforce_eager is mandatory — CUDA graph compilation bypasses hooks
        vllm_kwargs["enforce_eager"] = True
        vllm_kwargs.setdefault("dtype", dtype)

        self._llm = vllm.LLM(model=model, **vllm_kwargs)

        # Register hooks now that the engine is live
        from vllm_probe_hook import _hooks
        _hooks.register_hooks(self._llm, self._layers)

    def generate(
        self,
        prompts,
        sampling_params=None,
        **kwargs,
    ) -> list[RequestOutput]:
        """Generate completions and collect per-layer hidden states.

        Args:
            prompts: List of prompt strings or vLLM prompt dicts.
            sampling_params: vllm.SamplingParams or None.
            **kwargs: Additional kwargs forwarded to vllm.LLM.generate().

        Returns:
            List of RequestOutput (vllm.RequestOutput subclass with .hidden_states),
            one per prompt, in the same order.
        """
        from vllm_probe_hook import _activation_store

        # Mark all requests as in prefill so hooks skip prompt tokens.
        # vLLM assigns request IDs internally; we intercept them by wrapping
        # prompts in request dicts when the caller hasn't already done so.
        # For simplicity we rely on the scheduler-based batch mapping in the hook:
        # we mark a sentinel "any" skip and clear once the first output arrives.
        #
        # For the synchronous LLM, generation is blocking and single-request-at-a-time
        # from our perspective (vLLM may batch internally, but we call generate() once
        # per prompt list and get all outputs back together).  We therefore collect
        # hidden states *after* the full generation completes, using the request_id
        # attached to each RequestOutput.

        outputs = self._llm.generate(prompts, sampling_params, **kwargs)

        # At this point all tokens have been generated.  The hooks have been
        # accumulating hidden states for each request_id throughout the decode phase.
        results = []
        for out in outputs:
            num_tokens = len(out.outputs[0].token_ids)
            hidden = _activation_store.collect(
                out.request_id, num_tokens, self._layers, timeout=10.0
            )
            _activation_store.clear(out.request_id)
            results.append(_attach_hidden_states(out, hidden))
        return results

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks and release hook handles.

        Call this when you no longer need hidden state extraction to free resources.
        Note that this will prevent any further extraction of hidden states on this object.
        """
        from vllm_probe_hook import _hooks
        _hooks.remove_hooks()


class AsyncLLM:
    """Async streaming wrapper using vLLM 0.6.6 AsyncLLMEngine + forward hooks.

    Note: only one AsyncLLM (or LLM) instance should exist per process.
    Hook state is module-level; multiple instances will interfere with each other.

    Args:
        model: HuggingFace model name or path.
        layers: List of layer indices to extract hidden states from.
        dtype: Model dtype (default "bfloat16").
        **engine_kwargs: Additional keyword arguments forwarded to AsyncEngineArgs.
            Note: enforce_eager=True is always set (required for hooks).
    """

    def __init__(
        self,
        model: str,
        layers: list[int],
        dtype: str = "bfloat16",
        **engine_kwargs,
    ):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        self._layers = sorted(layers)
        self._model_name = model

        engine_kwargs["enforce_eager"] = True
        engine_kwargs.setdefault("dtype", dtype)

        engine_args = AsyncEngineArgs(model=model, **engine_kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Background event loop (same pattern as stable_backend)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._start_background_loop()

        # Register hooks once the engine model is warm.
        # AsyncLLMEngine initialises lazily; we trigger it here.
        self._ensure_hooks_registered()

    def _start_background_loop(self) -> None:
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()
            self._loop.close()

        self._loop_thread = threading.Thread(target=run, daemon=True)
        self._loop_thread.start()
        ready.wait(timeout=30)
        if self._loop is None:
            raise RuntimeError("Failed to start background event loop")

    def _ensure_hooks_registered(self) -> None:
        """Warm the engine and register hooks (runs a no-op generation if needed)."""
        from vllm_probe_hook import _hooks

        # Try to register directly; the engine may not have initialised its
        # workers yet.  If it hasn't, run a tiny warm-up generation.
        try:
            _hooks.register_hooks(self._engine, self._layers)
        except AttributeError:
            # Engine not fully initialised yet; warm it up with a tiny prompt.
            future = asyncio.run_coroutine_threadsafe(
                self._warmup(), self._loop
            )
            future.result(timeout=120)
            _hooks.register_hooks(self._engine, self._layers)

    async def _warmup(self) -> None:
        from vllm import SamplingParams
        from vllm_probe_hook import _activation_store
        async for _ in self._engine.generate(
            "hi", SamplingParams(max_tokens=1), request_id="__warmup__"
        ):
            pass
        _activation_store.clear("__warmup__")

    async def generate(
        self,
        prompt: str | dict,
        sampling_params=None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Stream outputs one token at a time with hidden states.

        Each yielded RequestOutput has partial hidden_states (tokens so far).
        The final yield has complete hidden_states for all generated tokens.

        Args:
            prompt: Prompt string or vLLM prompt dict.
            sampling_params: vllm.SamplingParams or None.
            request_id: Unique request identifier; auto-generated if not provided.
        """
        from vllm_probe_hook import _activation_store, _hooks

        if request_id is None:
            request_id = str(uuid.uuid4())

        _hooks.mark_prefill(request_id)
        is_first = True

        try:
            async for vllm_output in self._engine.generate(
                prompt, sampling_params, request_id
            ):
                if is_first:
                    # First output signals prefill complete
                    is_first = False
                    _hooks.unmark_prefill(request_id)
                    # Clear any accidentally accumulated prefill activations
                    _activation_store.clear(request_id)
                    continue

                num_tokens = len(vllm_output.outputs[0].token_ids)
                if vllm_output.finished:
                    hidden = await asyncio.get_running_loop().run_in_executor(
                        None,
                        _activation_store.collect,
                        request_id,
                        num_tokens,
                        self._layers,
                        10.0,
                    )
                    _activation_store.clear(request_id)
                else:
                    hidden = _activation_store.peek(request_id, self._layers)

                yield _attach_hidden_states(vllm_output, hidden)

        finally:
            _hooks.unmark_prefill(request_id)
            _activation_store.clear(request_id)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks and release hook handles.

        Call this when you no longer need hidden state extraction, e.g. before
        the process exits or to free any resources held by hook handles.
        """
        from vllm_probe_hook import _hooks
        _hooks.remove_hooks()
