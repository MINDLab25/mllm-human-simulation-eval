"""
Concurrent sync inference for PMSV prediction.

Uploads each unique video exactly once, runs inference for all participants
of that video, then deletes it.  For few-shot mode, N example videos are also
uploaded once per video group (shared by all participants of that video) and
deleted when the group is done.

Default concurrency is controlled by MAX_CONCURRENT in config.py.
Override at runtime:  MAX_CONCURRENT=20 python main.py
"""

from __future__ import annotations

import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

from pmsv_synth.config import GOOGLE_API_KEY, MODEL_NAME
from pmsv_synth.prompts.few_shot import (
    build_few_shot_contents,
    get_fixed_few_shot_examples,
)
from pmsv_synth.prompts.zero_shot import (
    PMSVRatings,
    SYSTEM_PROMPT,
    build_user_prompt,
    reverse_score,
)

_ITEM_IDS = [
    "emotional", "arousing", "involving", "exciting", "powerful_impact",
    "stimulating", "strong_visual", "strong_soundeffect",
    "dramatic", "graphic", "creative", "goosebump", "intense", "strong_soundtrack",
    "novel", "unique", "unusual",
]
_CSV_COLUMNS = (
    ["participant_id", "video_id", "survey_batch_id",
     "age", "gender", "race", "education", "income", "sen_seek",
     "human_perceived_msv", "predicted_msv",
     "video_path", "example_pairs", "thought_summary"]
    + [f"human_{i}" for i in _ITEM_IDS]
    + [f"ai_{i}"    for i in _ITEM_IDS]
)

_FILE_POLL_INTERVAL = 5
_FILE_POLL_TIMEOUT  = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    return genai.Client(api_key=GOOGLE_API_KEY)


def _upload_and_wait(client: genai.Client, video_path: Path) -> types.File:
    """Upload a video and block until it reaches ACTIVE state."""
    import time
    uploaded = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(
            display_name=video_path.name,
            mime_type="video/mp4",
        ),
    )
    deadline = time.monotonic() + _FILE_POLL_TIMEOUT
    while True:
        state = getattr(uploaded.state, "name", str(uploaded.state))
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise RuntimeError(f"File upload failed: {uploaded.name}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"File {uploaded.name} never reached ACTIVE state")
        time.sleep(_FILE_POLL_INTERVAL)
        uploaded = client.files.get(name=uploaded.name)


def _result_to_flat_row(record: dict[str, Any]) -> dict[str, Any]:
    item_ratings = record.get("item_ratings") or {}
    row: dict[str, Any] = {
        "participant_id":      record["participant_id"],
        "video_id":            record["video_id"],
        "survey_batch_id":     record["survey_batch_id"],
        "age":                 record.get("age"),
        "gender":              record.get("gender"),
        "race":                record.get("race"),
        "education":           record.get("education"),
        "income":              record.get("income"),
        "sen_seek":            record.get("sen_seek"),
        "human_perceived_msv": record["human_perceived_msv"],
        "predicted_msv":       record.get("predicted_msv"),
        "video_path":          record.get("video_path"),
        "example_pairs":       record.get("example_pairs"),
        "thought_summary":     record.get("thought_summary"),
    }
    for item_id in _ITEM_IDS:
        row[f"human_{item_id}"] = record.get(f"human_{item_id}")
        row[f"ai_{item_id}"]    = item_ratings.get(item_id)
    return row


# ---------------------------------------------------------------------------
# Per-participant inference
# ---------------------------------------------------------------------------

def _infer_for_participant(
    row: Any,
    target_uri: str,
    client: genai.Client,
    example_rows: list[dict[str, Any]] | None = None,
    example_uris: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run one inference call using an already-uploaded target video URI.

    In few-shot mode, example_rows and example_uris must both be provided.
    The model receives the example videos interleaved with their ratings,
    followed by the target video and the rating question.
    """
    if example_rows and example_uris:
        contents = build_few_shot_contents(
            participant=row.to_dict(),
            target_uri=target_uri,
            example_rows=example_rows,
            example_uris=example_uris,
        )
    else:
        user_prompt = build_user_prompt(row.to_dict())
        contents = [
            types.Part.from_uri(file_uri=target_uri, mime_type="video/mp4"),
            types.Part.from_text(text=user_prompt),
        ]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=PMSVRatings,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )

    # Extract thought summary (separate from the JSON output part)
    thought_summary: str | None = None
    for part in (response.candidates or [{}])[0].content.parts if response.candidates else []:
        if getattr(part, "thought", False) and part.text:
            thought_summary = part.text
            break

    ratings: PMSVRatings = response.parsed
    raw_ratings: dict[str, int] = ratings.model_dump()
    item_ratings = reverse_score(raw_ratings)
    predicted_msv = sum(item_ratings.values()) / len(item_ratings)
    return {
        "item_ratings": item_ratings,
        "predicted_msv": predicted_msv,
        "raw_response": response.text,
        "thought_summary": thought_summary,
    }


# ---------------------------------------------------------------------------
# Per-video worker
# ---------------------------------------------------------------------------

def _process_video_group(
    video_id: int,
    pending_rows: list[Any],
    human_item_cols: list[str],
    n_shots: int = 0,
    survey_df: pd.DataFrame | None = None,
    video_path_map: dict[int, Path] | None = None,
) -> list[dict[str, Any]]:
    """
    Upload one target video, infer for all its participants, then delete it.

    In few-shot mode, N example videos are also uploaded once (shared across
    all participants of this video) and deleted when the group finishes.

    Each worker thread creates its own Gemini client for thread safety.
    """
    client = _get_client()
    survey_batch_id = pending_rows[0]["survey_batch_id"]
    video_path = Path(pending_rows[0]["video_path"])
    records: list[dict[str, Any]] = []

    # ── Upload target video ────────────────────────────────────────────────
    try:
        uploaded_target = _upload_and_wait(client, video_path)
        target_uri  = uploaded_target.uri  or ""
        target_name = uploaded_target.name or ""
    except Exception as exc:
        print(f"\n[warn] Failed to upload {survey_batch_id}: {exc}", file=sys.stderr)
        for row in pending_rows:
            records.append({
                "participant_id":      int(row["participant_id"]),
                "video_id":            int(row["video_id"]),
                "survey_batch_id":     row["survey_batch_id"],
                "human_perceived_msv": float(row["perceived_msv"]),
                "age": row.get("age"), "gender": row.get("gender"),
                "race": row.get("race"), "education": row.get("education"),
                "income": row.get("income"), "sen_seek": row.get("sen_seek"),
                **{f"human_{c}": row.get(c) for c in human_item_cols},
                "item_ratings": None, "predicted_msv": None,
                "raw_response": str(exc),
            })
        return records

    # ── Upload example videos (few-shot) ──────────────────────────────────
    example_rows: list[dict[str, Any]] = []
    example_uris: list[str] = []
    example_names: list[str] = []

    if n_shots > 0 and survey_df is not None and video_path_map is not None:
        examples_df = get_fixed_few_shot_examples(
            current_video_id=video_id,
            survey_df=survey_df,
        )
        for _, ex_row in examples_df.iterrows():
            ex_vid_id = int(ex_row["video_id"])
            ex_path = video_path_map.get(ex_vid_id)
            if ex_path is None:
                continue
            try:
                uploaded_ex = _upload_and_wait(client, Path(ex_path))
                example_uris.append(uploaded_ex.uri or "")
                example_names.append(uploaded_ex.name or "")
                example_rows.append(ex_row.to_dict())
            except Exception as exc:
                print(
                    f"\n[warn] Failed to upload example video {ex_vid_id}: {exc}",
                    file=sys.stderr,
                )

    use_few_shot = bool(example_rows and example_uris)

    # Serialise example pairs (video_id + participant_id) for provenance
    example_pairs_json: str | None = None
    if use_few_shot:
        pairs = [
            {"video_id": int(r.get("video_id", 0)),
             "participant_id": int(r.get("participant_id", 0))}
            for r in example_rows
        ]
        example_pairs_json = json.dumps(pairs)

    # ── Infer for each participant ─────────────────────────────────────────
    try:
        for row in pending_rows:
            pid = int(row["participant_id"])
            vid = int(row["video_id"])
            try:
                pred = _infer_for_participant(
                    row, target_uri, client,
                    example_rows=example_rows if use_few_shot else None,
                    example_uris=example_uris if use_few_shot else None,
                )
            except Exception as exc:
                print(f"\n[warn] Skipping {pid}/{vid}: {exc}", file=sys.stderr)
                pred = {"item_ratings": None, "predicted_msv": None,
                        "raw_response": str(exc), "thought_summary": None}
            records.append({
                "participant_id":      pid,
                "video_id":            vid,
                "survey_batch_id":     row["survey_batch_id"],
                "human_perceived_msv": float(row["perceived_msv"]),
                "age": row.get("age"), "gender": row.get("gender"),
                "race": row.get("race"), "education": row.get("education"),
                "income": row.get("income"), "sen_seek": row.get("sen_seek"),
                "video_path":    str(video_path),
                "example_pairs": example_pairs_json,
                **{f"human_{c}": row.get(c) for c in human_item_cols},
                **pred,
            })
    finally:
        # Delete target + all example videos
        names_to_delete = [target_name] + example_names
        for name in names_to_delete:
            if name:
                try:
                    client.files.delete(name=name)
                except Exception:
                    pass

    return records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sync(
    participants: pd.DataFrame,
    human_item_cols: list[str],
    output_csv: Path | None = None,
    done_pairs: set[tuple[int, int]] | None = None,
    max_concurrent: int = 10,
    n_shots: int = 0,
    survey_df: pd.DataFrame | None = None,
    video_path_map: dict[int, Path] | None = None,
) -> list[dict[str, Any]]:
    """
    Run inference concurrently, uploading each unique video exactly once.

    Parameters
    ----------
    participants    : merged DataFrame from get_participants_for_sample()
    human_item_cols : ordered list of 17 PMSV item column names
    output_csv      : if given, rows are appended incrementally (crash recovery)
    done_pairs      : set of (participant_id, video_id) to skip (resume)
    max_concurrent  : videos processed simultaneously (default 10)
    n_shots         : few-shot examples per prompt (0 = zero-shot)
    survey_df       : full msv_df_by_participant DataFrame; needed for n_shots > 0
    video_path_map  : {video_id: Path} for all videos; needed for n_shots > 0
    """
    done_pairs = done_pairs or set()

    csv_fh     = None
    csv_writer: csv.DictWriter | None = None
    csv_lock   = threading.Lock()

    if output_csv is not None:
        is_new = not output_csv.exists()
        csv_fh = output_csv.open("a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=_CSV_COLUMNS,
                                    extrasaction="ignore")
        if is_new:
            csv_writer.writeheader()
            csv_fh.flush()

    # Build per-video task list, skipping fully-done videos
    video_groups = participants.groupby("video_id", sort=False)
    pending_tasks: list[tuple[int, list[Any]]] = []
    for video_id, group in video_groups:
        pending_rows = [
            row for _, row in group.iterrows()
            if (int(row["participant_id"]), int(video_id)) not in done_pairs
        ]
        if pending_rows:
            pending_tasks.append((int(video_id), pending_rows))

    n_skipped = len(video_groups) - len(pending_tasks)
    if n_skipped:
        print(f"[sync] {n_skipped} videos already done — "
              f"{len(pending_tasks)} remaining.", flush=True)

    mode_label = "few-shot" if n_shots > 0 else "zero-shot"
    print(f"[sync] {len(pending_tasks)} videos | "
          f"max_concurrent={max_concurrent} | {mode_label}", flush=True)

    n_total = len(participants)
    n_done  = len(done_pairs)
    results: list[dict] = []
    progress = tqdm(total=len(pending_tasks), desc="Videos")

    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(
                    _process_video_group,
                    video_id, pending_rows, human_item_cols,
                    n_shots, survey_df, video_path_map,
                ): video_id
                for video_id, pending_rows in pending_tasks
            }

            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    records = future.result()
                except Exception as exc:
                    print(f"\n[error] Video {video_id}: {exc}", file=sys.stderr)
                    records = []

                with csv_lock:
                    results.extend(records)
                    n_done += len(records)
                    if csv_writer and csv_fh:
                        for rec in records:
                            csv_writer.writerow(_result_to_flat_row(rec))
                        csv_fh.flush()

                progress.update(1)
                progress.set_postfix({"pairs": f"{n_done}/{n_total}"})
    finally:
        progress.close()
        if csv_fh:
            csv_fh.close()

    return results
