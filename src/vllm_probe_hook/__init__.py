# includes the entire vLLM namespace such that this package can be used as a drop-in replacement for vLLM
from vllm import *  # re-export the full vllm public API (SamplingParams, etc.)
from vllm_probe_hook.llm import LLM, AsyncLLM, RequestOutput  # shadow vllm's LLM/RequestOutput

__all__ = ["LLM", "AsyncLLM", "RequestOutput"]
