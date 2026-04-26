"""
Concurrent sync inference for PMSV prediction.

Uploads each unique video exactly once, runs inference for all participants
of that video, then deletes it. Multiple videos are processed concurrently
using a ThreadPoolExecutor, which dramatically reduces total wall-clock time
while staying well within Gemini API rate limits.

Default concurrency is controlled by MAX_CONCURRENT in config.py (default 10).
Override at runtime with: MAX_CONCURRENT=20 python main.py --sync --full
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

from pmsv_synth.config import MODEL_NAME
from pmsv_synth.inference.gemini.gemini import delete_files, get_client, upload_and_wait
from pmsv_synth.prompts.cot import build_cot_user_prompt, parse_cot_response
from pmsv_synth.prompts.cot_no_profile import build_cot_user_prompt_no_profile
from pmsv_synth.prompts.zero_shot import (
    PMSVRatings,
    SYSTEM_PROMPT,
    build_user_prompt,
    reverse_score,
)
from pmsv_synth.prompts.zero_shot_no_profile import build_user_prompt_no_profile

# Column order for the live incremental CSV — matches export._flatten() schema
_ITEM_IDS = [
    "emotional", "arousing", "involving", "exciting", "powerful_impact",
    "stimulating", "strong_visual", "strong_soundeffect",
    "dramatic", "graphic", "creative", "goosebump", "intense", "strong_soundtrack",
    "novel", "unique", "unusual",
]
_CSV_COLUMNS = (
    ["participant_id", "video_id", "survey_batch_id",
     "age", "gender", "race", "education", "income", "sen_seek",
     "human_perceived_msv", "predicted_msv", "cot_reasoning", "example_pairs",
     "thought_summary"]
    + [f"human_{i}" for i in _ITEM_IDS]
    + [f"ai_{i}"    for i in _ITEM_IDS]
)


class QuotaExhaustedError(RuntimeError):
    """Raised when Gemini returns quota-exhausted / rate-limit errors."""


def _is_quota_exhausted_error(exc: Exception) -> bool:
    """Return True when *exc* indicates hard quota/rate exhaustion."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "quota exceeded" in msg
        or "exceeded your current quota" in msg
        or "retry in " in msg
    )


def _raise_if_quota_exhausted(exc: Exception) -> None:
    """Normalize quota/rate exhaustion to a dedicated exception."""
    if _is_quota_exhausted_error(exc):
        raise QuotaExhaustedError(str(exc)) from exc


def _result_to_flat_row(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a result dict to a single CSV row dict."""
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
        "cot_reasoning":       record.get("cot_reasoning"),
        "example_pairs":       record.get("example_pairs"),
        "thought_summary":     record.get("thought_summary"),
    }
    for item_id in _ITEM_IDS:
        row[f"human_{item_id}"] = record.get(f"human_{item_id}")
        row[f"ai_{item_id}"]    = item_ratings.get(item_id)
    return row


def _infer_for_participant(
    row: Any,
    uploaded_uri: str,
    client: genai.Client,
    use_cot: bool = False,
    survey_order: bool = False,
    no_profile: bool = False,
    example_rows: list[dict[str, Any]] | None = None,
    example_uris: list[str] | None = None,
) -> dict[str, Any]:
    """Run a single inference call using an already-uploaded file URI.

    Priority: few-shot > cot > zero-shot.
    """
    # ── Few-shot mode ────────────────────────────────────────────────────────
    if example_rows and example_uris:
        from pmsv_synth.prompts.few_shot import build_few_shot_contents
        contents = build_few_shot_contents(
            participant=row.to_dict(),
            target_uri=uploaded_uri,
            example_rows=example_rows,
            example_uris=example_uris,
        )
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
        thought_summary_fs: str | None = None
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if getattr(part, "thought", False) and part.text:
                thought_summary_fs = part.text
                break
        ratings: PMSVRatings = response.parsed
        raw_dict: dict[str, int] = ratings.model_dump()
        item_ratings = reverse_score(raw_dict) if survey_order else raw_dict
        predicted_msv = sum(item_ratings.values()) / len(item_ratings)
        return {
            "item_ratings": item_ratings,
            "predicted_msv": predicted_msv,
            "cot_reasoning": None,
            "thought_summary": thought_summary_fs,
            "raw_response": response.text,
        }

    # ── Chain-of-thought mode ────────────────────────────────────────────────
    if use_cot:
        user_prompt = (
            build_cot_user_prompt_no_profile(survey_order=survey_order)
            if no_profile
            else build_cot_user_prompt(row.to_dict(), survey_order=survey_order)
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_uri(file_uri=uploaded_uri, mime_type="video/mp4"),
                types.Part.from_text(text=user_prompt),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
        thought_summary_cot: str | None = None
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if getattr(part, "thought", False) and part.text:
                thought_summary_cot = part.text
                break
        raw_text = response.text or ""
        raw_ratings, reasoning = parse_cot_response(raw_text)
        if raw_ratings is None:
            raise ValueError(
                f"CoT response did not contain a parseable 'Final ratings' JSON block. "
                f"Response snippet: {raw_text[:300]!r}"
            )
        try:
            validated = PMSVRatings.model_validate(raw_ratings)
        except Exception as exc:
            raise ValueError(f"CoT ratings failed validation: {exc}") from exc
        raw_dict_cot = validated.model_dump()
        item_ratings = reverse_score(raw_dict_cot) if survey_order else raw_dict_cot
        predicted_msv = sum(item_ratings.values()) / len(item_ratings)
        return {
            "item_ratings": item_ratings,
            "predicted_msv": predicted_msv,
            "cot_reasoning": reasoning,
            "thought_summary": thought_summary_cot,
            "raw_response": raw_text,
        }

    # ── Zero-shot mode ───────────────────────────────────────────────────────
    user_prompt = (
        build_user_prompt_no_profile(survey_order=survey_order)
        if no_profile
        else build_user_prompt(row.to_dict(), survey_order=survey_order)
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_uri(file_uri=uploaded_uri, mime_type="video/mp4"),
            types.Part.from_text(text=user_prompt),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=PMSVRatings,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )
    thought_summary: str | None = None
    for part in (response.candidates[0].content.parts if response.candidates else []):
        if getattr(part, "thought", False) and part.text:
            thought_summary = part.text
            break
    ratings_zs: PMSVRatings = response.parsed
    raw_dict_zs: dict[str, int] = ratings_zs.model_dump()
    item_ratings = reverse_score(raw_dict_zs) if survey_order else raw_dict_zs
    predicted_msv = sum(item_ratings.values()) / len(item_ratings)
    return {
        "item_ratings": item_ratings,
        "predicted_msv": predicted_msv,
        "cot_reasoning": None,
        "thought_summary": thought_summary,
        "raw_response": response.text,
    }


def _process_video_group(
    video_id: int,
    pending_rows: list[Any],
    human_item_cols: list[str],
    use_cot: bool = False,
    survey_order: bool = False,
    no_profile: bool = False,
    n_shots: int = 0,
    survey_df: pd.DataFrame | None = None,
    video_path_map: dict[int, Path] | None = None,
) -> list[dict[str, Any]]:
    """
    Worker function: handles the full lifecycle for one video.

    Creates its own Gemini client for thread safety, uploads the video once
    (plus any few-shot example videos), runs inference for every pending
    participant, then deletes all uploaded files.

    Returns a list of result dicts (one per participant).
    """
    from pmsv_synth.prompts.few_shot import get_fixed_few_shot_examples

    client = get_client()
    survey_batch_id = pending_rows[0]["survey_batch_id"]
    video_path = Path(pending_rows[0]["video_path"])
    records: list[dict[str, Any]] = []

    files_to_delete: list[str] = []

    # Upload target video once
    try:
        uploaded = upload_and_wait(client, video_path)
        uploaded_uri = uploaded.uri or ""
        files_to_delete.append(uploaded.name or "")
    except Exception as exc:
        _raise_if_quota_exhausted(exc)
        print(
            f"\n[warn] Failed to upload {survey_batch_id}: {exc}",
            file=sys.stderr,
        )
        for row in pending_rows:
            records.append({
                "participant_id":      int(row["participant_id"]),
                "video_id":            int(row["video_id"]),
                "survey_batch_id":     row["survey_batch_id"],
                "human_perceived_msv": float(row["perceived_msv"]),
                "age":       row.get("age"),
                "gender":    row.get("gender"),
                "race":      row.get("race"),
                "education": row.get("education"),
                "income":    row.get("income"),
                "sen_seek":  row.get("sen_seek"),
                **{f"human_{col}": row.get(col) for col in human_item_cols},
                "item_ratings": None,
                "predicted_msv": None,
                "cot_reasoning": None,
                "thought_summary": None,
                "example_pairs": None,
                "raw_response": str(exc),
            })
        return records

    # Sample and upload example videos once per group (few-shot mode)
    example_rows: list[dict[str, Any]] = []
    example_uris: list[str] = []
    example_pairs_json: str | None = None

    if n_shots > 0 and survey_df is not None and video_path_map is not None:
        examples_df = get_fixed_few_shot_examples(
            current_video_id=video_id,
            survey_df=survey_df,
        )
        for _, ex_row in examples_df.iterrows():
            ex_vid_id = int(ex_row["video_id"])
            ex_path = video_path_map.get(ex_vid_id)
            if ex_path is None or not ex_path.exists():
                continue
            try:
                ex_uploaded = upload_and_wait(client, ex_path)
                example_rows.append(ex_row.to_dict())
                example_uris.append(ex_uploaded.uri or "")
                files_to_delete.append(ex_uploaded.name or "")
            except Exception as exc:
                _raise_if_quota_exhausted(exc)
                print(
                    f"\n[warn] Failed to upload example video {ex_vid_id}: {exc}",
                    file=sys.stderr,
                )

        if example_rows:
            example_pairs_json = json.dumps([
                {"video_id": int(r["video_id"]), "participant_id": int(r["participant_id"])}
                for r in example_rows
            ])

    # Infer for each participant, delete all uploaded files when done
    try:
        for row in pending_rows:
            pid = int(row["participant_id"])
            vid = int(row["video_id"])
            try:
                prediction = _infer_for_participant(
                    row, uploaded_uri, client,
                    use_cot=use_cot,
                    survey_order=survey_order,
                    no_profile=no_profile,
                    example_rows=example_rows or None,
                    example_uris=example_uris or None,
                )
            except Exception as exc:
                _raise_if_quota_exhausted(exc)
                print(
                    f"\n[warn] Skipping participant {pid} / video {vid}: {exc}",
                    file=sys.stderr,
                )
                prediction = {
                    "item_ratings": None,
                    "predicted_msv": None,
                    "cot_reasoning": None,
                    "thought_summary": None,
                    "raw_response": str(exc),
                }
            records.append({
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
                "example_pairs": example_pairs_json,
                **prediction,
            })
    finally:
        delete_files(client, [f for f in files_to_delete if f])

    return records


def run_sync(
    participants: pd.DataFrame,
    human_item_cols: list[str],
    client: genai.Client | None = None,  # kept for API compatibility, unused
    output_csv: Path | None = None,
    done_pairs: set[tuple[int, int]] | None = None,
    max_concurrent: int = 10,
    use_cot: bool = False,
    survey_order: bool = False,
    no_profile: bool = False,
    n_shots: int = 0,
    survey_df: pd.DataFrame | None = None,
    video_path_map: dict[int, Path] | None = None,
) -> list[dict[str, Any]]:
    """
    Run inference concurrently, uploading each unique video exactly once.

    Up to `max_concurrent` videos are processed simultaneously. Within each
    video, participant inference calls are sequential (one at a time per thread).

    Parameters
    ----------
    participants    : merged DataFrame from get_participants_for_sample()
    human_item_cols : list of 17 PMSV item column names
    client          : ignored (each worker creates its own client); kept for
                      backwards-compatibility with callers that pass one
    output_csv      : if given, results are appended as CSV rows as each video
                      group finishes (incremental save for crash recovery)
    done_pairs      : set of (participant_id, video_id) int tuples already
                      processed; rows in this set are skipped silently (resume)
    max_concurrent  : number of videos to process simultaneously (default 10)
    use_cot         : if True, use chain-of-thought prompting
    survey_order    : if True, present items with original Qualtrics anchor
                      directions (some reversed) and apply reverse_score()
                      after inference. If False (default), all items are
                      normalized to 1 = low, 7 = high
    n_shots         : number of few-shot examples per prompt (0 = zero-shot)
    survey_df       : full participant survey DataFrame (required if n_shots > 0)
    video_path_map  : dict mapping video_id → local Path (required if n_shots > 0)

    Returns
    -------
    list of result dicts — same structure as batch.parse_batch_results()
    """
    done_pairs = done_pairs or set()

    # Prepare live CSV
    csv_fh = None
    csv_writer: csv.DictWriter | None = None
    csv_lock = threading.Lock()
    if output_csv is not None:
        is_new_file = not output_csv.exists()
        csv_fh = output_csv.open("a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        if is_new_file:
            csv_writer.writeheader()
            csv_fh.flush()

    results: list[dict] = []
    n_total_pairs = len(participants)
    n_processed = len(done_pairs)

    # Build per-video task list, filtering out fully-done videos
    video_groups = participants.groupby("video_id", sort=False)
    n_videos = len(video_groups)

    pending_tasks: list[tuple[int, list[Any]]] = []
    for video_id, group in video_groups:
        pending_rows = [
            row for _, row in group.iterrows()
            if (int(row["participant_id"]), int(video_id)) not in done_pairs
        ]
        if pending_rows:
            pending_tasks.append((int(video_id), pending_rows))

    n_skipped_videos = n_videos - len(pending_tasks)
    if n_skipped_videos:
        print(
            f"[sync] {n_skipped_videos} videos fully done (resume) — "
            f"{len(pending_tasks)} remaining.",
            flush=True,
        )

    if n_shots > 0:
        mode_label = "few-shot"
    elif use_cot:
        mode_label = "cot" + ("-no-profile" if no_profile else "")
    else:
        mode_label = "zero-shot" + ("-no-profile" if no_profile else "")

    print(
        f"[sync] Starting {len(pending_tasks)} videos with "
        f"max_concurrent={max_concurrent} | "
        f"{mode_label} | "
        f"{'survey-order' if survey_order else 'normalized'}",
        flush=True,
    )

    progress = tqdm(total=len(pending_tasks), desc="Videos (concurrent)")

    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(
                    _process_video_group,
                    video_id,
                    pending_rows,
                    human_item_cols,
                    use_cot,
                    survey_order,
                    no_profile,
                    n_shots,
                    survey_df,
                    video_path_map,
                ): video_id
                for video_id, pending_rows in pending_tasks
            }

            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    records = future.result()
                except QuotaExhaustedError as exc:
                    # Stop quickly on hard quota errors so resume can continue
                    # from the last successful CSV row.
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    print(
                        f"\n[error] Quota exhausted while processing video {video_id}: {exc}",
                        file=sys.stderr,
                    )
                    print(
                        "[error] Stopping sync run immediately. Re-run with "
                        "--resume after quota resets.",
                        file=sys.stderr,
                    )
                    raise
                except Exception as exc:
                    print(
                        f"\n[error] Unexpected failure for video {video_id}: {exc}",
                        file=sys.stderr,
                    )
                    records = []

                # Thread-safe: extend results and write CSV
                with csv_lock:
                    results.extend(records)
                    n_processed += len(records)
                    if csv_writer is not None and csv_fh is not None:
                        for record in records:
                            csv_writer.writerow(_result_to_flat_row(record))
                        csv_fh.flush()

                progress.update(1)
                progress.set_postfix({"pairs": f"{n_processed}/{n_total_pairs}"})

    finally:
        progress.close()
        if csv_fh is not None:
            csv_fh.close()

    return results
