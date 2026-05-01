"""figures.py — shared utilities: data loaders, stats helpers, constants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Helvetica",
    "Helvetica Neue",
    "Arial",
    "DejaVu Sans",
]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Helvetica"
matplotlib.rcParams["mathtext.it"] = "Helvetica:italic"
matplotlib.rcParams["mathtext.bf"] = "Helvetica:bold"

import numpy as np
import pandas as pd
from scipy import stats

FS = {
    "annotation": 35,
    "tick": 32,
    "axis_label": 37,
    "y_label": 35,
    "title": 38,
    "grid_title": 35,
}
LW = {
    "regression": 8,
    "identity": 3,
}

GEMINI_COLOR = "#8fd8fa"
QWEN_COLOR = "#cb98e7"
MALE_COLOR = "#399ca7"
FEMALE_COLOR = "#fe8a5b"

ITEM_IDS = [
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

AGE_BINS = [0, 38, 200_000]
AGE_LABELS = ["Under 39", "39 or older"]

EDUCATION_MAP = {
    1: "Below Bachelor's",
    2: "Below Bachelor's",
    3: "Below Bachelor's",
    4: "Below Bachelor's",
    5: "Bachelor's or higher",
    6: "Bachelor's or higher",
}
INCOME_MAP = {
    1: "Less than $50,000",
    2: "Less than $50,000",
    3: "$50,000 – $99,999",
    4: "$50,000 – $99,999",
    5: "$100,000 or more",
    6: "$100,000 or more",
}


# ── Data loading ──────────────────────────────────────────────────────────────


def load_participant_agg(csv_path: Path) -> pd.DataFrame:
    """One row per participant — mean across all their videos."""
    df = pd.read_csv(csv_path)
    agg: dict[str, str] = {}
    for item in ITEM_IDS:
        if f"human_{item}" in df.columns:
            agg[f"human_{item}"] = "mean"
        if f"ai_{item}" in df.columns:
            agg[f"ai_{item}"] = "mean"
    if "human_perceived_msv" in df.columns:
        agg["human_perceived_msv"] = "mean"
    if "predicted_msv" in df.columns:
        agg["predicted_msv"] = "mean"
    demo = {
        c: "first"
        for c in ["age", "gender", "race", "sen_seek", "education", "income"]
        if c in df.columns
    }
    out = df.groupby("participant_id").agg({**agg, **demo}).reset_index()
    out = out.rename(
        columns={"human_perceived_msv": "human_msv", "predicted_msv": "ai_msv"}
    )
    if "age" in out.columns:
        out["age_group"] = pd.cut(
            out["age"].clip(upper=120), bins=AGE_BINS, labels=AGE_LABELS
        )
    if "education" in out.columns:
        out["edu_label"] = out["education"].apply(
            lambda x: EDUCATION_MAP.get(int(x), str(x)) if pd.notna(x) else None
        )
    if "income" in out.columns:
        out["income_label"] = out["income"].apply(
            lambda x: INCOME_MAP.get(int(x), str(x)) if pd.notna(x) else None
        )
    if "race" in out.columns:
        out["race"] = out["race"].replace("Black or African American", "Black")
    if "sen_seek" in out.columns:
        out["ss_group"] = pd.cut(
            out["sen_seek"],
            bins=[0.99, 2.5, 5.01],
            labels=["Low SS", "High SS"],
        )
    return out


def load_rating_level(csv_path: Path) -> pd.DataFrame:
    """Load raw (participant × video) ratings — no aggregation."""
    df = pd.read_csv(csv_path)
    keep = ["participant_id", "video_id", "human_perceived_msv", "predicted_msv"] + [
        c
        for c in df.columns
        if (c.startswith("human_") or c.startswith("ai_"))
        and c not in ("human_perceived_msv",)
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.rename(
        columns={"human_perceived_msv": "human_msv", "predicted_msv": "ai_msv"}
    )
    return out.dropna(subset=["human_msv", "ai_msv"])


# ── Shared stats helpers ──────────────────────────────────────────────────────


def _ci_band(human, ai, x_fit):
    """95% CI band for a regression line."""
    slope, intercept, *_ = stats.linregress(human, ai)
    y_fit = slope * x_fit + intercept
    n = len(human)
    x_mean = np.mean(human)
    ss_xx = np.sum((human - x_mean) ** 2)
    ss_res = np.sum((ai - (slope * human + intercept)) ** 2)
    mse = ss_res / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (x_fit - x_mean) ** 2 / ss_xx))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    return y_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit


def _round_clip(vals: np.ndarray) -> np.ndarray:
    return np.clip(np.round(vals).astype(int), 1, 7)


def _weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    return cohen_kappa_score(a, b, weights="quadratic")


def icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(A,1): two-way random, absolute agreement, single measure (k=2)."""
    n = len(x)
    m = (x + y) / 2
    mu = m.mean()
    SSB = 2 * np.sum((m - mu) ** 2)
    SSW = np.sum((x - m) ** 2 + (y - m) ** 2)
    SSC = n * ((x.mean() - mu) ** 2 + (y.mean() - mu) ** 2)
    SSE = SSW - SSC
    MSB = SSB / (n - 1)
    MSW = SSW / n
    MSC = SSC
    MSE = SSE / (n - 1)
    denom = MSB + MSW + 2 * (MSC - MSE) / n
    return float((MSB - MSW) / denom) if denom != 0 else float("nan")


_icc_a1 = icc_a1  # backward-compat alias


# ── CLI (shared by fig_2 / fig_3 / fig_4) ────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures 2–4.")
    parser.add_argument(
        "--csv", type=Path, default=Path("data/results/gemini_zero_shot.csv")
    )
    parser.add_argument(
        "--qwen-csv", type=Path, default=Path("data/results/qwen_zero_shot.csv")
    )
    parser.add_argument(
        "--fews-csv", type=Path, default=Path("data/results/gemini_few_shot.csv")
    )
    parser.add_argument(
        "--cot-csv", type=Path, default=Path("data/results/gemini_cot.csv")
    )
    parser.add_argument(
        "--qwen-cot-csv", type=Path, default=Path("data/results/qwen_cot.csv")
    )
    parser.add_argument(
        "--qwen-fews-csv", type=Path, default=Path("data/results/qwen_few_shot.csv")
    )
    return parser.parse_args()
