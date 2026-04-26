"""
Batch API inference for PMSV prediction.

Submits all participant × video requests as a single asynchronous Gemini
Batch API job — 50% cheaper than real-time, no per-request rate-limit
juggling, but adds latency (Google targets 24 h; usually much faster).

Flow
----
1. upload_videos()          — upload unique videos to Files API once
2. build_inline_requests()  — one request dict per participant × video
3. submit_batch_job()       — submit all requests as one job
4. poll_batch_job()         — wait for completion (polls every 60 s)
5. parse_batch_results()    — convert responses to result dicts
6. delete_files()           — clean up Files API quota  (from gemini.py)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from tqdm import tqdm

from pmsv_synth.config import MODEL_NAME
from pmsv_synth.inference.gemini.gemini import get_client, upload_and_wait
from pmsv_synth.prompts.few_shot import (
    build_user_prompt_few_shot,
    get_fixed_few_shot_examples,
)
from pmsv_synth.prompts.zero_shot import (
    PMSVRatings,
    SYSTEM_PROMPT,
    build_user_prompt,
    reverse_score,
)

BATCH_POLL_INTERVAL = 60  # seconds between status polls

_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def upload_videos(
    participants: pd.DataFrame,
    client: genai.Client | None = None,
) -> tuple[dict[str, str], list[str]]:
    """
    Upload all unique videos to the Files API.

    Returns
    -------
    uri_map    : {survey_batch_id: file_uri}   — used when building requests
    file_names : list of Files API names       — pass to delete_files() after
    """
    if client is None:
        client = get_client()

    unique_videos = (
        participants[["survey_batch_id", "video_path"]]
        .drop_duplicates(subset="survey_batch_id")
        .reset_index(drop=True)
    )

    uri_map: dict[str, str] = {}
    file_names: list[str] = []

    for _, row in tqdm(
        unique_videos.iterrows(),
        total=len(unique_videos),
        desc="Uploading videos",
    ):
        uploaded = upload_and_wait(client, Path(row["video_path"]))
        uri_map[row["survey_batch_id"]] = uploaded.uri or ""
        file_names.append(uploaded.name or "")
        print(f"  ↑ {row['survey_batch_id']} → {uploaded.name}")

    print(f"[upload] {len(uri_map)} videos ready.")
    return uri_map, file_names


def build_inline_requests(
    participants: pd.DataFrame,
    uri_map: dict[str, str],
    n_shots: int = 0,
    survey_df: pd.DataFrame | None = None,
    survey_order: bool = False,
) -> list[dict]:
    """
    Build one inline request dict per participant × video row.

    The list order matches participants.iterrows() so responses can be
    mapped back by index after the job completes.

    Parameters
    ----------
    n_shots      : number of few-shot examples to prepend (0 = zero-shot)
    survey_df    : full msv_df_by_participant DataFrame; required when n_shots > 0
    survey_order : if True, use original Qualtrics anchor directions (some
                   reversed). If False (default), all items normalized to
                   1 = low, 7 = high sensation.
    """
    requests: list[dict] = []
    for _, row in participants.iterrows():
        uri = uri_map[row["survey_batch_id"]]
        if n_shots > 0 and survey_df is not None:
            examples = get_fixed_few_shot_examples(
                current_video_id=int(row["video_id"]),
                survey_df=survey_df,
            )
            user_prompt = build_user_prompt_few_shot(
                row.to_dict(), examples, survey_order=survey_order
            )
        else:
            user_prompt = build_user_prompt(row.to_dict(), survey_order=survey_order)
        requests.append(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "file_data": {
                                    "file_uri": uri,
                                    "mime_type": "video/mp4",
                                }
                            },
                            {"text": user_prompt},
                        ],
                    }
                ],
                "config": {
                    "system_instruction": {
                        "parts": [{"text": SYSTEM_PROMPT}]
                    },
                    "response_mime_type": "application/json",
                    "response_schema": PMSVRatings,
                    "temperature": 0.0,
                },
            }
        )
    return requests


def submit_batch_job(
    inline_requests: list[dict],
    client: genai.Client | None = None,
    display_name: str = "pmsv-inference",
) -> Any:
    """Submit inline batch requests and return the job object."""
    if client is None:
        client = get_client()

    job = client.batches.create(
        model=MODEL_NAME,
        src=inline_requests,
        config={"display_name": display_name},
    )
    print(f"[batch] Job submitted → {job.name}")
    return job


def poll_batch_job(client: genai.Client, job_name: str) -> Any:
    """
    Poll until the job reaches a terminal state.

    Raises RuntimeError on failure, cancellation, or expiry.
    Returns the completed job object on success.
    """
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name if job.state else "JOB_STATE_PENDING"
        if state in _TERMINAL_STATES:
            print(f"[batch] Finished with state: {state}")
            if state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(
                    f"Batch job ended with state {state}: "
                    f"{getattr(job, 'error', '')}"
                )
            return job
        print(f"[batch] {state} — next check in {BATCH_POLL_INTERVAL}s ...")
        time.sleep(BATCH_POLL_INTERVAL)


def parse_batch_results(
    job: Any,
    participants: pd.DataFrame,
    human_item_cols: list[str],
    survey_order: bool = False,
) -> list[dict[str, Any]]:
    """
    Convert inline batch responses into result dicts.

    Responses are matched to participants by position (same order as
    build_inline_requests). Failed or unparseable responses are recorded
    with predicted_msv=None so no row is silently dropped.
    """
    rows = list(participants.iterrows())
    results: list[dict] = []

    for i, inline_response in enumerate(job.dest.inlined_responses):
        _, row = rows[i]
        pid = int(row["participant_id"])
        vid = int(row["video_id"])

        if inline_response.error:
            print(
                f"\n[warn] Response {i} (participant {pid} / video {vid}) "
                f"failed: {inline_response.error}"
            )
            prediction: dict[str, Any] = {
                "item_ratings": None,
                "predicted_msv": None,
                "raw_response": str(inline_response.error),
            }
        else:
            raw_text = inline_response.response.text
            try:
                ratings = PMSVRatings.model_validate_json(raw_text)
                raw_ratings = ratings.model_dump()
                item_ratings = reverse_score(raw_ratings) if survey_order else raw_ratings
                predicted_msv = sum(item_ratings.values()) / len(item_ratings)
                prediction = {
                    "item_ratings": item_ratings,
                    "predicted_msv": predicted_msv,
                    "raw_response": raw_text,
                }
            except Exception as exc:
                print(
                    f"\n[warn] Parse error at response {i} "
                    f"(participant {pid} / video {vid}): {exc}"
                )
                prediction = {
                    "item_ratings": None,
                    "predicted_msv": None,
                    "raw_response": raw_text,
                }

        results.append(
            {
                "participant_id":      pid,
                "video_id":            vid,
                "survey_batch_id":     row["survey_batch_id"],
                "human_perceived_msv": float(row["perceived_msv"]),
                "age":       row.get("age"),
                "gender":    row.get("gender"),
                "race":      row.get("race"),
                "education": row.get("education"),
                "income":    row.get("income"),
                "sen_seek":  row.get("sen_seek"),
                **{f"human_{col}": row.get(col) for col in human_item_cols},
                **prediction,
            }
        )

    return results
