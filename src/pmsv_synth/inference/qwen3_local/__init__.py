"""
pmsv_synth.inference.qwen3_local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Local inference using Qwen3-Omni-30B-A3B-Instruct-GPTQ-4bit on a single GPU.

Model loader:
    infer

Sync API:
    run_sync
"""

from pmsv_synth.inference.qwen3_local.model import infer
from pmsv_synth.inference.qwen3_local.sync import run_sync

__all__ = [
    "infer",
    "run_sync",
]
