"""
pmsv_synth.inference.gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public re-exports for the Gemini inference package.

Shared utilities (gemini.py):
    get_client, upload_and_wait, delete_files

Batch API (batch.py):
    upload_videos, build_inline_requests, submit_batch_job,
    poll_batch_job, parse_batch_results

Sync API (sync.py):
    predict_pmsv, run_sync
"""

from pmsv_synth.inference.gemini.gemini import (
    delete_files,
    get_client,
    upload_and_wait,
)
from pmsv_synth.inference.gemini.batch import (
    build_inline_requests,
    parse_batch_results,
    poll_batch_job,
    submit_batch_job,
    upload_videos,
)
from pmsv_synth.inference.gemini.sync import (
    run_sync,
)

__all__ = [
    # shared
    "get_client",
    "upload_and_wait",
    "delete_files",
    # batch
    "upload_videos",
    "build_inline_requests",
    "submit_batch_job",
    "poll_batch_job",
    "parse_batch_results",
    # sync
    "run_sync",
]
