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
    prefill_hidden: dict[int, torch.Tensor] | None = None,
):
    """Promote a plain vllm.RequestOutput to our RequestOutput subclass in-place."""
    cls = _get_request_output_class()
    vllm_out.__class__ = cls
    vllm_out.hidden_states = hidden
    vllm_out.prefill_hidden_states = prefill_hidden if prefill_hidden is not None else {}
    return vllm_out


def _trim_last(hidden: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Return a new dict keeping only the last token slice for each layer."""
    return {lid: t[-1:] for lid, t in hidden.items()}


def _resolve_layers(layers, engine) -> list[int]:
    """Expand layers=[\"all\"] to a sorted list of every layer index."""
    if layers == ["all"]:
        from vllm_probe_hook import _hooks
        model = _hooks._get_model(engine)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            n = len(model.model.layers)
        elif hasattr(model, "layers"):
            n = len(model.layers)
        else:
            raise AttributeError("Cannot determine layer count from model")
        return list(range(n))
    return sorted(layers)


class LLM:
    """Wrapper around vllm.LLM that extracts per-layer hidden states via hooks.

    Note: only one LLM (or AsyncLLM) instance should exist per process.
    Hook state is module-level, so multiple instances will interfere with each other. Known limitation.

    Args:
        model: HuggingFace model name or path.
        layers: List of layer indices to extract hidden states from, or ["all"]
            to capture every transformer layer.
        dtype: Model dtype (default "bfloat16").
        include_prefill: If True, also collect prompt-token activations into
            .prefill_hidden_states (default False).
        include_decode: If False, suppress decode-token collection (default True).
        last_prefill_token: If True and include_prefill is on, keep only the
            final prompt token's activations (default False).
        last_decode_token: If True and include_decode is on, keep only the
            final generated token's activations (default False).
        **vllm_kwargs: Additional keyword arguments forwarded to vllm.LLM.
            Note: enforce_eager=True is always set (required for hooks).
    """

    def __init__(
        self,
        model: str,
        layers: list[int] | list[str],
        dtype: str = "bfloat16",
        include_prefill: bool = False,
        include_decode: bool = True,
        last_prefill_token: bool = False,
        last_decode_token: bool = False,
        **vllm_kwargs,
    ):
        import vllm

        self._model_name = model
        self._include_prefill = include_prefill
        self._include_decode = include_decode
        self._last_prefill_token = last_prefill_token
        self._last_decode_token = last_decode_token

        # enforce_eager is mandatory — CUDA graph compilation bypasses hooks
        vllm_kwargs["enforce_eager"] = True
        vllm_kwargs.setdefault("dtype", dtype)

        self._llm = vllm.LLM(model=model, **vllm_kwargs)

        # Resolve layers (must happen after engine is live for ["all"])
        self._layers = _resolve_layers(layers, self._llm)

        # Configure hook capture flags before registering
        from vllm_probe_hook import _hooks
        _hooks.configure(include_prefill=include_prefill, include_decode=include_decode)
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
        from vllm_probe_hook import _activation_store, _hooks

        # Normalise prompts to a list so we can count them
        if isinstance(prompts, (str, dict)):
            prompts_list = [prompts]
        else:
            prompts_list = list(prompts)

        # Pre-compute the request IDs that vLLM will assign.
        # vLLM.LLM._add_request() calls str(next(self.request_counter)) for each
        # prompt in order, and _add_request is called before _run_engine starts.
        if self._include_prefill:
            start = self._llm.request_counter.counter
            expected_ids = [str(start + i) for i in range(len(prompts_list))]
            for rid in expected_ids:
                _hooks.mark_prefill(rid)

        try:
            outputs = self._llm.generate(prompts_list, sampling_params, **kwargs)
        finally:
            if self._include_prefill:
                for rid in expected_ids:
                    _hooks.unmark_prefill(rid)

        results = []
        for out in outputs:
            if self._include_decode:
                num_tokens = len(out.outputs[0].token_ids)
                hidden = _activation_store.collect(
                    out.request_id, num_tokens, self._layers, timeout=10.0
                )
            else:
                hidden = {}
            prefill_hidden = _activation_store.collect_prefill(out.request_id, self._layers)
            _activation_store.clear(out.request_id)
            if self._last_decode_token and hidden:
                hidden = _trim_last(hidden)
            if self._last_prefill_token and prefill_hidden:
                prefill_hidden = _trim_last(prefill_hidden)
            results.append(_attach_hidden_states(out, hidden, prefill_hidden))
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
        layers: List of layer indices to extract hidden states from, or ["all"]
            to capture every transformer layer.
        dtype: Model dtype (default "bfloat16").
        include_prefill: If True, also collect prompt-token activations into
            .prefill_hidden_states (default False).
        include_decode: If False, suppress decode-token collection (default True).
        last_prefill_token: If True and include_prefill is on, keep only the
            final prompt token's activations (default False).
        last_decode_token: If True and include_decode is on, keep only the
            final generated token's activations (default False).
        **engine_kwargs: Additional keyword arguments forwarded to AsyncEngineArgs.
            Note: enforce_eager=True is always set (required for hooks).
    """

    def __init__(
        self,
        model: str,
        layers: list[int] | list[str],
        dtype: str = "bfloat16",
        include_prefill: bool = False,
        include_decode: bool = True,
        last_prefill_token: bool = False,
        last_decode_token: bool = False,
        **engine_kwargs,
    ):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        self._model_name = model
        self._include_prefill = include_prefill
        self._include_decode = include_decode
        self._last_prefill_token = last_prefill_token
        self._last_decode_token = last_decode_token
        self._layers_spec = layers  # resolved after engine warms up

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
            self._layers = _resolve_layers(self._layers_spec, self._engine)
            _hooks.configure(
                include_prefill=self._include_prefill,
                include_decode=self._include_decode,
            )
            _hooks.register_hooks(self._engine, self._layers)
        except AttributeError:
            # Engine not fully initialised yet; warm it up with a tiny prompt.
            future = asyncio.run_coroutine_threadsafe(
                self._warmup(), self._loop
            )
            future.result(timeout=120)
            self._layers = _resolve_layers(self._layers_spec, self._engine)
            _hooks.configure(
                include_prefill=self._include_prefill,
                include_decode=self._include_decode,
            )
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
        prefill_hidden: dict = {}

        try:
            async for vllm_output in self._engine.generate(
                prompt, sampling_params, request_id
            ):
                if is_first:
                    # First output signals prefill complete
                    is_first = False
                    _hooks.unmark_prefill(request_id)

                    if self._include_prefill:
                        prefill_hidden = _activation_store.collect_prefill(
                            request_id, self._layers
                        )
                        if self._last_prefill_token and prefill_hidden:
                            prefill_hidden = _trim_last(prefill_hidden)
                    _activation_store.clear_prefill(request_id)

                    if not self._include_prefill:
                        # Clear any accidentally accumulated prefill activations
                        # in the decode store as well (original behaviour)
                        _activation_store.clear(request_id)

                    continue

                num_tokens = len(vllm_output.outputs[0].token_ids)
                if vllm_output.finished:
                    if self._include_decode:
                        hidden = await asyncio.get_running_loop().run_in_executor(
                            None,
                            _activation_store.collect,
                            request_id,
                            num_tokens,
                            self._layers,
                            10.0,
                        )
                        if self._last_decode_token and hidden:
                            hidden = _trim_last(hidden)
                    else:
                        hidden = {}
                    _activation_store.clear(request_id)
                else:
                    hidden = _activation_store.peek(request_id, self._layers) if self._include_decode else {}

                yield _attach_hidden_states(vllm_output, hidden, prefill_hidden)

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
