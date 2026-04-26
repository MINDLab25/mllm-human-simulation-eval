"""
Shared Gemini API utilities (client, Files API helpers).

Imported by both batch.py and sync.py — do not add mode-specific logic here.
"""

from __future__ import annotations

import time
from pathlib import Path

from google import genai
from google.genai import types

from pmsv_synth.config import GOOGLE_API_KEY

FILE_POLL_INTERVAL = 5  # seconds between file-state checks
FILE_POLL_TIMEOUT = 120  # max seconds to wait for ACTIVE state


def get_client() -> genai.Client:
    return genai.Client(api_key=GOOGLE_API_KEY)


def upload_and_wait(client: genai.Client, video_path: Path) -> types.File:
    """Upload a video and block until it reaches ACTIVE state."""
    uploaded = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(
            display_name=video_path.name,
            mime_type="video/mp4",
        ),
    )
    deadline = time.time() + FILE_POLL_TIMEOUT
    while uploaded.state != types.FileState.ACTIVE:
        if uploaded.state == types.FileState.FAILED:
            raise RuntimeError(
                f"File upload failed for {video_path.name}: {uploaded.error}"
            )
        if time.time() > deadline:
            raise TimeoutError(
                f"Video {video_path.name} did not become ACTIVE within "
                f"{FILE_POLL_TIMEOUT}s"
            )
        time.sleep(FILE_POLL_INTERVAL)
        uploaded = client.files.get(name=uploaded.name or "")
    return uploaded


def delete_files(client: genai.Client, file_names: list[str]) -> None:
    """Delete a list of Files API entries to free quota."""
    for name in file_names:
        try:
            client.files.delete(name=name)
        except Exception:
            pass
    print(f"[cleanup] Deleted {len(file_names)} files from Files API.")
