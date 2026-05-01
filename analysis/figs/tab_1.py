"""
Table 1: PMSV ICC(A,1) between human and MLLM-synthesized participants
within each demographic/psychographic subgroup (zero-shot).

Reproduces tab:pmsv_subgroups from the paper.

Usage:
    python analysis/figs/tab_1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from figures import icc_a1, load_participant_agg

GEMINI = Path("data/results/gemini_zero_shot.csv")
QWEN   = Path("data/results/qwen_zero_shot.csv")


def icc(df: pd.DataFrame) -> float:
    d = df.dropna(subset=["human_msv", "ai_msv"])
    return round(icc_a1(d["human_msv"].to_numpy(), d["ai_msv"].to_numpy()), 2)


def subgroup_table(gem: pd.DataFrame, qwen: pd.DataFrame) -> list[tuple]:
    rows = []

    def add(cat, label, mask_fn):
        mg, mq = mask_fn(gem), mask_fn(qwen)
        rows.append((cat, label, mg.sum(), icc(gem[mg]), icc(qwen[mq])))

    add("Gender",    "Male",     lambda d: d["gender"] == "Male")
    add("Gender",    "Female",   lambda d: d["gender"] == "Female")

    add("Ethnicity", "White",    lambda d: d["race"] == "White or Caucasian")
    add("Ethnicity", "Black",    lambda d: d["race"] == "Black")
    add("Ethnicity", "Asian",    lambda d: d["race"] == "Asian")

    add("Age",       "<39",      lambda d: d["age_group"] == "Under 39")
    add("Age",       "≥39",      lambda d: d["age_group"] == "39 or older")

    add("Education", "<Bachelor's",  lambda d: d["edu_label"] == "Below Bachelor's")
    add("Education", "≥Bachelor's",  lambda d: d["edu_label"] == "Bachelor's or higher")

    add("Income",    "<$50k",    lambda d: d["income_label"] == "Less than $50,000")
    add("Income",    "$50–100k", lambda d: d["income_label"] == "$50,000 – $99,999")
    add("Income",    "≥$100k",   lambda d: d["income_label"] == "$100,000 or more")

    add("SS",        "<2.5",     lambda d: d["sen_seek"] < 2.5)
    add("SS",        "≥2.5",     lambda d: d["sen_seek"] >= 2.5)

    return rows


def main():
    gem  = load_participant_agg(GEMINI)
    qwen = load_participant_agg(QWEN)

    rows = subgroup_table(gem, qwen)

    print(f"{'Category':<15} {'Subgroup':<20} {'n':>5}  {'Gemini':>8}  {'Qwen':>8}")
    print("-" * 62)
    prev_cat = ""
    for cat, label, n, g_icc, q_icc in rows:
        cat_str = cat if cat != prev_cat else ""
        prev_cat = cat
        print(f"{cat_str:<15} {label:<20} {n:>5}  {g_icc:>8.2f}  {q_icc:>8.2f}")


if __name__ == "__main__":
    main()
