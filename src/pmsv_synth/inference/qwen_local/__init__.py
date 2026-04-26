"""
pmsv_synth.inference.qwen_local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Local inference using Qwen2.5-Omni-7B-GPTQ-Int4 on a single GPU.

Model loader:
    infer

Sync API:
    run_sync
"""

from pmsv_synth.inference.qwen_local.model import infer
from pmsv_synth.inference.qwen_local.sync import run_sync

__all__ = [
    "infer",
    "run_sync",
]
