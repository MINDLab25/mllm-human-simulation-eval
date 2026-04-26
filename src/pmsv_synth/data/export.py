"""
Export helpers: convert a results JSON file into flat and comparison CSVs.

results CSV   — one row per participant × video pair, item_ratings exploded
                into individual ai_<item_id> columns; raw_response dropped.

comparison CSV — one row per video, aggregating mean human vs mean predicted
                PMSV together with MAE and RMSE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Item IDs match dataset column names in msv_df_by_participant.csv
_ITEM_IDS = [
    "emotional",
    "arousing",
    "involving",
    "exciting",
    "powerful_impact",
    "stimulating",
    "strong_visual",
    "strong_soundeffect",
    "dramatic",
    "graphic",
    "creative",
    "goosebump",
    "intense",
    "strong_soundtrack",
    "novel",
    "unique",
    "unusual",
]


def _load_results(results_path: Path) -> list[dict]:
    """Load results from a JSON file or a flat results CSV."""
    if results_path.suffix == ".csv":
        df = pd.read_csv(results_path)
        records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            # Re-pack ai_<item> columns into item_ratings dict
            rec["item_ratings"] = {
                item_id: rec.get(f"ai_{item_id}")
                for item_id in _ITEM_IDS
            }
            records.append(rec)
        return records
    with open(results_path) as f:
        return json.load(f)


def _flatten(record: dict) -> dict:
    """Return a flat dict suitable for one CSV row."""
    row = {
        "participant_id": record["participant_id"],
        "video_id": record["video_id"],
        "survey_batch_id": record["survey_batch_id"],
        "age": record.get("age"),
        "gender": record.get("gender"),
        "race": record.get("race"),
        "education": record.get("education"),
        "income": record.get("income"),
        "sen_seek": record.get("sen_seek"),
        "human_perceived_msv": record["human_perceived_msv"],
        "predicted_msv": record.get("predicted_msv"),
    }
    # Human per-item ratings
    for item_id in _ITEM_IDS:
        row[f"human_{item_id}"] = record.get(f"human_{item_id}")
    # AI per-item ratings
    item_ratings = record.get("item_ratings") or {}
    for item_id in _ITEM_IDS:
        row[f"ai_{item_id}"] = item_ratings.get(item_id)
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def results_to_csv(results_path: Path) -> Path:
    """
    Convert a results JSON to a flat CSV (one row per participant × video pair).

    Saves alongside the JSON as results_<timestamp>.csv.
    Returns the path of the saved CSV.
    """
    records = _load_results(results_path)
    rows = [_flatten(r) for r in records]
    df = pd.DataFrame(rows)

    out_path = results_path.with_suffix(".csv")
    df.to_csv(out_path, index=False)
    return out_path


def results_to_comparison_csv(results_path: Path) -> Path:
    """
    Aggregate results to one row per video and save a comparison CSV.

    Columns: video_id, survey_batch_id, n_participants,
             mean_human_msv, mean_predicted_msv, mae, rmse

    Saves alongside the JSON as comparison_<timestamp>.csv.
    Returns the path of the saved CSV.
    """
    records = _load_results(results_path)

    rows = []
    for r in records:
        if r.get("predicted_msv") is not None:
            rows.append(
                {
                    "video_id": r["video_id"],
                    "survey_batch_id": r["survey_batch_id"],
                    "human_msv": r["human_perceived_msv"],
                    "predicted_msv": r["predicted_msv"],
                }
            )

    if not rows:
        raise ValueError("No successful predictions found in results file.")

    df = pd.DataFrame(rows)

    def _rmse(group: pd.DataFrame) -> float:
        errors = group["human_msv"] - group["predicted_msv"]
        return math.sqrt((errors**2).mean())

    comparison = (
        df.groupby(["video_id", "survey_batch_id"])
        .apply(
            lambda g: pd.Series(
                {
                    "n_participants": len(g),
                    "mean_human_msv": g["human_msv"].mean(),
                    "mean_predicted_msv": g["predicted_msv"].mean(),
                    "mae": (g["human_msv"] - g["predicted_msv"]).abs().mean(),
                    "rmse": _rmse(g),
                }
            ),
        )
        .reset_index()
    )

    # Summary row across all videos
    overall_df = df.copy()
    overall_errors = overall_df["human_msv"] - overall_df["predicted_msv"]
    summary = pd.DataFrame(
        [
            {
                "video_id": "ALL",
                "survey_batch_id": "ALL",
                "n_participants": len(overall_df),
                "mean_human_msv": overall_df["human_msv"].mean(),
                "mean_predicted_msv": overall_df["predicted_msv"].mean(),
                "mae": overall_errors.abs().mean(),
                "rmse": math.sqrt((overall_errors**2).mean()),
            }
        ]
    )

    comparison = pd.concat([comparison, summary], ignore_index=True)

    stem = results_path.stem.replace("results_", "")
    out_path = results_path.parent / f"comparison_{stem}.csv"
    comparison.to_csv(out_path, index=False)
    return out_path
