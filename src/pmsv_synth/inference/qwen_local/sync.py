"""Sequential local inference for PMSV prediction using Qwen2.5-Omni-7B-GPTQ-Int4.

The model is loaded once as a singleton. All participant-video pairs are
processed sequentially on a single GPU — no ThreadPoolExecutor. Only zero-shot
mode is supported.

Results are written incrementally to the output CSV after each pair so that a
crash can be resumed with --resume.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from pmsv_synth.inference.qwen_local.model import infer
from pmsv_synth.prompts.cot import build_cot_user_prompt, parse_cot_response
from pmsv_synth.prompts.zero_shot import (
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
     "human_perceived_msv", "predicted_msv", "cot_reasoning", "example_pairs",
     "raw_response"]
    + [f"human_{i}" for i in _ITEM_IDS]
    + [f"ai_{i}" for i in _ITEM_IDS]
)

_MAX_RETRIES = 10

_JSON_FORMAT_INSTRUCTION = (
    "\n\nRespond with ONLY a valid JSON object — no markdown fences, no extra text. "
    "Use exactly these 17 keys, each with an integer value from 1 to 7:\n"
    "emotional, arousing, involving, exciting, powerful_impact, stimulating, "
    "strong_visual, strong_soundeffect, dramatic, graphic, creative, goosebump, "
    "intense, strong_soundtrack, novel, unique, unusual"
)

_MAX_RETRIES = 10


def _normalize_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Map model-generated key names to the canonical _ITEM_IDS names.

    The quantized model occasionally generates malformed key names (e.g.
    "arous arous" instead of "arousing", or "powerful impact" instead of
    "powerful_impact").  We use difflib to snap each key to the nearest
    canonical name so pydantic validation succeeds.
    """
    import difflib

    result: dict[str, Any] = {}
    for key, value in raw.items():
        # Try exact match first
        if key in _ITEM_IDS:
            result[key] = value
            continue
        # Strip whitespace variants and underscores, compare case-insensitively
        normalised = key.lower().replace(" ", "_").replace("-", "_")
        if normalised in _ITEM_IDS:
            result[normalised] = value
            continue
        # Fuzzy closest match (cutoff 0.6 avoids false positives)
        candidates = difflib.get_close_matches(
            normalised, _ITEM_IDS, n=1, cutoff=0.6
        )
        if candidates:
            result[candidates[0]] = value
        else:
            result[key] = value  # keep as-is; pydantic will reject it
    return result


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    raw = json.loads(text.strip())
    return _normalize_keys(raw)


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
        "cot_reasoning":       record.get("cot_reasoning"),
        "example_pairs":       None,
        "raw_response":        record.get("raw_response"),
    }
    for item_id in _ITEM_IDS:
        row[f"human_{item_id}"] = record.get(f"human_{item_id}")
        row[f"ai_{item_id}"]    = item_ratings.get(item_id)
    return row


def run_sync(
    participants: pd.DataFrame,
    human_item_cols: list[str],
    output_csv: Path | None = None,
    done_pairs: set[tuple[int, int]] | None = None,
    max_concurrent: int = 1,
    survey_order: bool = False,
    use_cot: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    """Run local Qwen2.5-Omni-7B-GPTQ-Int4 zero-shot inference sequentially.

    Parameters
    ----------
    participants    : merged DataFrame from get_participants_for_sample()
    human_item_cols : list of 17 PMSV item column names
    output_csv      : if given, results are appended after each pair (crash recovery)
    done_pairs      : set of (participant_id, video_id) pairs already processed
    max_concurrent  : ignored — local single-GPU inference is always sequential
    survey_order    : if True, present items in original Qualtrics anchor directions
                      and apply reverse_score() after inference
    use_cot         : if True, use chain-of-thought prompting (Reasoning + Final ratings)
    **kwargs        : ignored (kept for API compatibility with other providers)
    """
    done_pairs = done_pairs or set()

    csv_fh = None
    csv_writer: csv.DictWriter | None = None
    if output_csv is not None:
        is_new_file = not output_csv.exists()
        csv_fh = output_csv.open("a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        if is_new_file:
            csv_writer.writeheader()
            csv_fh.flush()

    results: list[dict[str, Any]] = []
    n_total_pairs = len(participants)

    pending: list[dict[str, Any]] = [
        row.to_dict()
        for _, row in participants.iterrows()
        if (int(row["participant_id"]), int(row["video_id"])) not in done_pairs
    ]

    n_skipped = n_total_pairs - len(pending)
    if n_skipped:
        print(
            f"[sync/qwen-local] {n_skipped} pairs already done (resume) — "
            f"{len(pending)} remaining.",
            flush=True,
        )

    mode_label = "cot" if use_cot else "zero-shot"
    print(
        f"[sync/qwen-local] Processing {len(pending)} participant-video pairs "
        f"sequentially | {mode_label} | "
        f"{'survey-order' if survey_order else 'normalized'}",
        flush=True,
    )

    try:
        for row in tqdm(pending, desc="Pairs (local Qwen)"):
            pid = int(row["participant_id"])
            vid = int(row["video_id"])
            video_path = Path(row["video_path"])
            if use_cot:
                user_prompt = build_cot_user_prompt(row, survey_order=survey_order)
            else:
                user_prompt = build_user_prompt(row, survey_order=survey_order)
                user_prompt += _JSON_FORMAT_INSTRUCTION

            raw_text: str | None = None
            last_exc: Exception | None = None
            prediction: dict[str, Any] | None = None

            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    raw_text = infer(
                        system_prompt=SYSTEM_PROMPT,
                        user_text=user_prompt,
                        video_path=video_path,
                    )
                    if use_cot:
                        raw_ratings, reasoning = parse_cot_response(raw_text)
                        if raw_ratings is None:
                            raise ValueError("CoT response missing 'Final ratings:' block")
                        item_ratings = {
                            k: max(1, min(7, int(round(float(v)))))
                            for k, v in raw_ratings.items()
                            if k in _ITEM_IDS
                        }
                        for item in _ITEM_IDS:
                            item_ratings.setdefault(item, 4)
                        if survey_order:
                            item_ratings = reverse_score(item_ratings)
                    else:
                        raw_dict = _parse_json_response(raw_text)
                        item_ratings = {
                            k: max(1, min(7, int(round(float(v)))))
                            for k, v in raw_dict.items()
                            if k in _ITEM_IDS
                        }
                        for item in _ITEM_IDS:
                            item_ratings.setdefault(item, 4)
                        if survey_order:
                            item_ratings = reverse_score(item_ratings)
                        reasoning = None
                    predicted_msv = sum(item_ratings.values()) / len(item_ratings)
                    prediction = {
                        "item_ratings":  item_ratings,
                        "predicted_msv": predicted_msv,
                        "cot_reasoning": reasoning,
                        "raw_response":  raw_text,
                    }
                    break  # success
                except Exception as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        print(
                            f"\n[warn] participant {pid} / video {vid}: "
                            f"attempt {attempt}/{_MAX_RETRIES} failed ({exc}) — retrying…",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"\n[warn] Skipping participant {pid} / video {vid} "
                            f"after {_MAX_RETRIES} attempts: {exc}",
                            file=sys.stderr,
                        )

            if prediction is None:
                prediction = {
                    "item_ratings":  None,
                    "predicted_msv": None,
                    "cot_reasoning": None,
                    "raw_response":  raw_text if raw_text is not None else str(last_exc),
                }

            record: dict[str, Any] = {
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
                "example_pairs": None,
                **prediction,
            }
            results.append(record)

            if csv_writer is not None and csv_fh is not None:
                csv_writer.writerow(_result_to_flat_row(record))
                csv_fh.flush()

    finally:
        if csv_fh is not None:
            csv_fh.close()

    return results
