"""
Sample n videos from msv_df.csv and persist the sample for reuse.

Video file resolution:
  survey_batch_id "1_100" → VIDEO_DIR/batch 1/1_100+*.mp4
  (batch subfolder is "batch {n}" where n is the part before the underscore)
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from pmsv_synth.config import (
    MSV_BY_PARTICIPANT_PATH,
    MSV_DF_PATH,
    SAMPLE_SIZE,
    SAMPLES_DIR,
    VIDEO_DIR,
)

LATEST_PATH = SAMPLES_DIR / "latest.csv"


def _resolve_video_path(survey_batch_id: str) -> Path:
    """Return the local path for a video given its survey_batch_id."""
    batch = survey_batch_id.split("_")[0]
    batch_dir = VIDEO_DIR / f"batch {batch}"
    matches = list(batch_dir.glob(f"{survey_batch_id}+*.mp4"))
    if not matches:
        raise FileNotFoundError(
            f"No video file found for survey_batch_id={survey_batch_id!r} "
            f"in {batch_dir}"
        )
    return matches[0]


def create_sample(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Draw a fresh random sample of n videos, resolve their local file paths,
    save to samples/ and update latest.csv.

    Returns a DataFrame with one row per video, including a 'video_path' column.
    """
    msv_df = pd.read_csv(MSV_DF_PATH)
    sample = msv_df.sample(n=n).reset_index(drop=True)

    sample["video_path"] = sample["survey_batch_id"].apply(
        lambda bid: str(_resolve_video_path(bid))
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAMPLES_DIR / f"sample_{timestamp}.csv"
    sample.to_csv(save_path, index=False)
    shutil.copy(save_path, LATEST_PATH)

    print(f"[sampler] New sample saved → {save_path.name}  (latest.csv updated)")
    return sample


def load_latest_sample() -> pd.DataFrame:
    """Load the most recently saved sample from samples/latest.csv."""
    if not LATEST_PATH.exists():
        raise FileNotFoundError(
            "No latest sample found. Run with --new-sample first."
        )
    sample = pd.read_csv(LATEST_PATH)
    print(f"[sampler] Loaded latest sample ({len(sample)} videos) from latest.csv")
    return sample


def load_sample(sample_path: Path) -> pd.DataFrame:
    """
    Load a saved sample from a specific CSV path (e.g. samples/sample_20260308_013304.csv).
    The CSV must have video_id and video_path columns.
    """
    path = Path(sample_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    sample = pd.read_csv(path)
    if "video_path" not in sample.columns or "video_id" not in sample.columns:
        raise ValueError(
            f"Sample CSV must have 'video_id' and 'video_path' columns. Found: {list(sample.columns)}"
        )
    print(f"[sampler] Loaded sample ({len(sample)} videos) from {path.name}")
    return sample


def load_full_dataset() -> pd.DataFrame:
    """
    Load all videos from msv_df.csv, resolve their local file paths,
    save to samples/full_<timestamp>.csv and update latest.csv.

    Returns a DataFrame with one row per video, including a 'video_path' column.
    """
    msv_df = pd.read_csv(MSV_DF_PATH)
    msv_df["video_path"] = msv_df["survey_batch_id"].apply(
        lambda bid: str(_resolve_video_path(bid))
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAMPLES_DIR / f"full_{timestamp}.csv"
    msv_df.to_csv(save_path, index=False)
    shutil.copy(save_path, LATEST_PATH)

    print(
        f"[sampler] Full dataset saved → {save_path.name}  "
        f"({len(msv_df)} videos, latest.csv updated)"
    )
    return msv_df


def get_participants_for_sample(
    sample: pd.DataFrame,
    participants_path: Path | None = None,
) -> pd.DataFrame:
    """
    Return all participant rows (from msv_df_by_participant) whose video_id
    is in the given sample, merged with the sample's survey_batch_id and
    video_path for convenience.

    Parameters
    ----------
    sample            : sample DataFrame with `video_id` and `video_path` columns.
    participants_path : optional override for the participant CSV. When given,
                        this file is loaded instead of MSV_BY_PARTICIPANT_PATH
                        (useful for shuffled-profile experiments).
    """
    src = Path(participants_path) if participants_path else MSV_BY_PARTICIPANT_PATH
    participants = pd.read_csv(src)
    video_meta = sample[["video_id", "video_path"]].copy()
    merged = participants.merge(video_meta, on="video_id", how="inner")
    return merged


def load_full_dataset() -> pd.DataFrame:
    """
    Load ALL videos from msv_df.csv, resolve video paths, save to
    samples/full_<timestamp>.csv and update latest.csv.
    """
    msv_df = pd.read_csv(MSV_DF_PATH)
    msv_df["video_path"] = msv_df["survey_batch_id"].apply(
        lambda bid: str(_resolve_video_path(bid))
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAMPLES_DIR / f"full_{timestamp}.csv"
    msv_df.to_csv(save_path, index=False)
    shutil.copy(save_path, LATEST_PATH)
    print(f"[sampler] Full dataset saved → {save_path.name}  ({len(msv_df)} videos, latest.csv updated)")
    return msv_df


def load_sample(sample_path: Path) -> pd.DataFrame:
    """Load a specific saved sample CSV by path."""
    sample = pd.read_csv(sample_path)
    print(f"[sampler] Loaded sample ({len(sample)} videos) from {sample_path.name}")
    return sample
