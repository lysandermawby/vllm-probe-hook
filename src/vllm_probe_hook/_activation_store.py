"""
Activation store — in-process accumulator for per-request hidden states.

PyTorch hooks run synchronously in the forward pass (same process, same thread
or a worker thread), so there is no IPC needed — we just collect tensors into a
thread-safe dict.

Layout of _store / _prefill_store:
    {request_id: {layer_id: [Tensor, ...]}}

Each tensor in the list has shape [batch_slice_size, hidden_size], typically
[1, hidden_size] per decode step.  collect() concatenates them on dim 0.
"""
from __future__ import annotations

import logging
import threading
import time

import torch

_store: dict[str, dict[int, list[torch.Tensor]]] = {}
_store_lock = threading.Lock()

_prefill_store: dict[str, dict[int, list[torch.Tensor]]] = {}
_prefill_store_lock = threading.Lock()


def append(request_id: str, layer_id: int, tensor: torch.Tensor) -> None:
    """Append a hidden-state tensor for a (request, layer) pair.

    tensor should be 2-D: [n_tokens, hidden_size].
    """
    with _store_lock:
        if request_id not in _store:
            _store[request_id] = {}
        if layer_id not in _store[request_id]:
            _store[request_id][layer_id] = []
        _store[request_id][layer_id].append(tensor.detach().cpu())


def append_prefill(request_id: str, layer_id: int, tensor: torch.Tensor) -> None:
    """Append a prefill hidden-state tensor for a (request, layer) pair.

    tensor should be 2-D: [n_tokens, hidden_size].
    """
    with _prefill_store_lock:
        if request_id not in _prefill_store:
            _prefill_store[request_id] = {}
        if layer_id not in _prefill_store[request_id]:
            _prefill_store[request_id][layer_id] = []
        _prefill_store[request_id][layer_id].append(tensor.detach().cpu())


def collect(
    request_id: str,
    num_tokens: int,
    layer_ids: list[int],
    timeout: float = 30.0,
) -> dict[int, torch.Tensor]:
    """Wait until all requested layers have >= num_tokens accumulated, then return.

    Returns {layer_id: Tensor[num_tokens, hidden_size]}.
    On timeout, logs a WARNING and returns whatever partial data is available;
    callers cannot otherwise distinguish a complete result from a timed-out one.
    """
    deadline = time.monotonic() + timeout
    while True:
        with _store_lock:
            req = _store.get(request_id, {})
            all_ready = all(
                layer_id in req
                and sum(t.shape[0] for t in req[layer_id]) >= num_tokens
                for layer_id in layer_ids
            )
            if all_ready:
                result = {}
                for lid in layer_ids:
                    combined = torch.cat(req[lid], dim=0)
                    result[lid] = combined[:num_tokens]
                return result

        if time.monotonic() >= deadline:
            logging.getLogger(__name__).warning(
                "collect() timed out after %.1fs for request %r "
                "(wanted %d tokens); returning partial data.",
                timeout, request_id, num_tokens,
            )
            with _store_lock:
                req = _store.get(request_id, {})
                result = {}
                for lid in layer_ids:
                    if lid in req and req[lid]:
                        combined = torch.cat(req[lid], dim=0)
                        result[lid] = combined[:num_tokens]
            return result

        time.sleep(0.005)


def collect_prefill(
    request_id: str,
    layer_ids: list[int],
) -> dict[int, torch.Tensor]:
    """Return all accumulated prefill hidden states for the given layers.

    Prefill arrives in a single forward pass so no polling is needed.
    Returns {} for any layer that has no data (graceful).
    """
    with _prefill_store_lock:
        req = _prefill_store.get(request_id, {})
        result = {}
        for lid in layer_ids:
            if lid in req and req[lid]:
                result[lid] = torch.cat(req[lid], dim=0)
        return result


def peek(request_id: str, layer_ids: list[int]) -> dict[int, torch.Tensor]:
    """Return partial hidden states accumulated so far without blocking or clearing."""
    with _store_lock:
        req = _store.get(request_id, {})
        result = {}
        for lid in layer_ids:
            if lid in req and req[lid]:
                result[lid] = torch.cat(req[lid], dim=0)
        return result


def clear(request_id: str) -> None:
    """Remove all accumulated data for a request (both decode and prefill stores)."""
    with _store_lock:
        _store.pop(request_id, None)
    with _prefill_store_lock:
        _prefill_store.pop(request_id, None)


def clear_prefill(request_id: str) -> None:
    """Remove only the prefill store entry for a request (keeps decode store intact)."""
    with _prefill_store_lock:
        _prefill_store.pop(request_id, None)
