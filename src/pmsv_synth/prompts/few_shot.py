"""
Few-shot prompt builder for PMSV prediction.

Each example is presented as the matching video clip + the respondent's
demographic profile + their actual ratings, so the model can calibrate its
own responses by watching those videos.

Fixed examples are always drawn from different videos than the target to
prevent data leakage.  Within a video group (all participants watching the
same video), the same example videos are reused so they are uploaded only once.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
from google.genai import types

from pmsv_synth.prompts.zero_shot import (
    _format_demographic_profile,
    _format_pmsv_items,
)

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

# ---------------------------------------------------------------------------
# Fixed few-shot examples
#
# Selected to maximise diversity across all demographic and socioeconomic
# dimensions while spanning the full MSV rating range:
#
#   Dim        Ex 1            Ex 2            Ex 3
#   ─────────  ──────────────  ──────────────  ──────────────
#   gender     Male            Female          Male
#   race       Black/AA        Asian           White
#   age        22 (young)      27 (young)      26 (young)
#   ss_score   4.50 (high)     1.38 (low)      2.13 (mid)
#   education  2    (low)      5    (high)      3    (mid)
#   income     3    (mid)      6    (high)      1    (low)
#   MSV        1.71 (low)      3.88 (mid)      5.29 (high)
#   duration   13.7 s           8.2 s          18.4 s
#   content    ski dance skit  tea recipe      frisbee dog catch
#
#   P67  + V51  |  Male,   Black/AA, 22, ss=4.50, edu=2, inc=3  |  13.7 s  |  MSV ≈ 1.71
#   P639 + V908 |  Female, Asian,    27, ss=1.38, edu=5, inc=6  |   8.2 s  |  MSV ≈ 3.88
#   P280 + V449 |  Male,   White,    26, ss=2.13, edu=3, inc=1  |  18.4 s  |  MSV ≈ 5.29
# ---------------------------------------------------------------------------

FIXED_FEW_SHOT_EXAMPLES: list[tuple[int, int]] = [
    (67,  51),   # Male,   Black/AA, 22, ss=4.50, edu=2, inc=3 — ski dance skit, low sensation
    (639, 908),  # Female, Asian,    27, ss=1.38, edu=5, inc=6 — strawberry tea recipe, mid sensation
    (280, 449),  # Male,   White,    26, ss=2.13, edu=3, inc=1 — frisbee dog catch, high sensation
]


# ---------------------------------------------------------------------------
# Fixed example lookup (replaces random sampling)
# ---------------------------------------------------------------------------


def get_fixed_few_shot_examples(
    current_video_id: int,
    survey_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the fixed few-shot example rows from survey_df.

    Any example whose video_id matches current_video_id is silently dropped
    to prevent data leakage (rare in practice given the fixed selection).

    Parameters
    ----------
    current_video_id : the video being evaluated (excluded if it appears)
    survey_df        : full msv_df_by_participant DataFrame

    Returns
    -------
    DataFrame with up to len(FIXED_FEW_SHOT_EXAMPLES) rows.
    """
    rows: list[pd.Series] = []
    for participant_id, video_id in FIXED_FEW_SHOT_EXAMPLES:
        if video_id == current_video_id:
            continue
        match = survey_df[
            (survey_df["participant_id"] == participant_id)
            & (survey_df["video_id"] == video_id)
        ]
        if not match.empty:
            rows.append(match.iloc[0])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_example_text(
    row: pd.Series | dict[str, Any],
    idx: int,
    survey_order: bool = False,
) -> str:
    """
    Format a single survey response as a labelled text block.

    This text is placed AFTER the corresponding example video part so the
    model can associate the video it just saw with these ratings.

    Parameters
    ----------
    survey_order : if False (default), reverse-scored items are converted so
                   that all values are in the 1 = low, 7 = high direction to
                   match the normalized scale shown to the model.
                   If True, raw survey values are used as-is.
    """
    from pmsv_synth.prompts.zero_shot import reverse_score

    if isinstance(row, pd.Series):
        row = row.to_dict()

    demo = _format_demographic_profile(row)

    raw_ratings: dict[str, Any] = {}
    for item_id in _ITEM_IDS:
        val = row.get(item_id)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            raw_ratings[item_id] = None
        else:
            raw_ratings[item_id] = int(val)

    if survey_order:
        ratings = raw_ratings
    else:
        # Convert non-None values; keep None as-is
        scoreable = {k: v for k, v in raw_ratings.items() if v is not None}
        scored = reverse_score(scoreable)
        ratings = {k: (scored[k] if k in scored else None) for k in raw_ratings}

    ratings_str = json.dumps(ratings, indent=2)

    return (
        f"=== Example {idx} ===\n"
        f"Respondent profile:\n{demo}\n\n"
        f"Survey ratings for the video above:\n{ratings_str}"
    )


# ---------------------------------------------------------------------------
# Text-only few-shot prompt (used by the Batch API)
# ---------------------------------------------------------------------------


def build_user_prompt_few_shot(
    participant: dict[str, Any],
    examples: pd.DataFrame,
    survey_order: bool = False,
) -> str:
    """
    Build a text-only few-shot user prompt for one participant × video pair.

    Used by the Batch API where only a single video can be attached per
    request; examples are represented as text (demographic profile + ratings)
    rather than actual video clips.

    Parameters
    ----------
    participant  : target respondent dict (demographics etc.)
    examples     : DataFrame of sampled survey rows (demographics + ratings)
    survey_order : if True, use original Qualtrics anchor directions (some
                   reversed). If False (default), all items normalized to
                   1 = low, 7 = high sensation.
    """
    scale = _format_pmsv_items(survey_order=survey_order)
    demo = _format_demographic_profile(participant)
    n = len(examples)
    plural = "s" if n != 1 else ""

    if survey_order:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where 1 = the LEFT anchor "
            "and 7 = the RIGHT anchor.\n"
            "Note: some items have the positive/high-sensation quality on the LEFT "
            '(e.g. "Unique \u2190\u2192 Common"), so a lower number means MORE of '
            "that quality for those items \u2014 just as in the real survey."
        )
    else:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where "
            "1 = least sensational and 7 = most sensational."
        )

    lines: list[str] = [
        f"You are a survey respondent. Below are {n} example{plural} showing how "
        f"a real person described and rated a video. Use them to calibrate the scale.\n",
    ]
    for i, (_, ex_row) in enumerate(examples.iterrows(), start=1):
        lines.append(_format_example_text(ex_row, i, survey_order=survey_order))
        lines.append("")

    lines += [
        "---",
        "Now rate the video you just watched as the respondent described below.",
        f"You are simulating a survey respondent with the following profile:\n{demo}\n",
        "Rate this video on each of the following 17 items using a 7-point scale.",
        scale_instruction,
        scale,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def build_few_shot_contents(
    participant: dict[str, Any],
    target_uri: str,
    example_rows: list[dict[str, Any]],
    example_uris: list[str],
    survey_order: bool = False,
) -> list[types.Part]:
    """
    Build the full multi-part contents list for a few-shot inference call.

    Structure (each item is a Part sent to the model):
      [intro text]
      [example 1 video]  [example 1 profile + ratings text]
      [example 2 video]  [example 2 profile + ratings text]
      ...
      [separator text]
      [target video]
      [target respondent profile + rating question text]

    Parameters
    ----------
    participant   : dict of the target respondent's demographics
    target_uri    : Files API URI of the target video (already uploaded)
    example_rows  : list of dicts, one per example (demographics + human ratings)
    example_uris  : list of Files API URIs, one per example video (same order)
    survey_order  : if True, use original Qualtrics anchor directions (some
                    reversed). If False (default), all items normalized to
                    1 = low, 7 = high sensation.
    """
    scale = _format_pmsv_items(survey_order=survey_order)
    demo = _format_demographic_profile(participant)
    n = len(example_rows)
    plural = "s" if n != 1 else ""

    if survey_order:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where 1 represents the left anchor "
            "and 7 represents the right anchor.\n"
            "Note: some items have the positive/high-sensation quality on the LEFT (rating of 1) "
            "rather than the right \u2014 just as in the real survey."
        )
    else:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where "
            "1 = least sensational and 7 = most sensational."
        )

    parts: list[types.Part] = []

    # ── Intro ──────────────────────────────────────────────────────────────
    intro = (
        f"You are a survey respondent. You will first see {n} example video{plural} "
        f"together with how a real person rated each one. Then you will watch a new "
        f"video and rate it yourself.\n\n"
        f"Use the examples to calibrate the scale — note how the ratings relate to "
        f"what you see in each video."
    )
    parts.append(types.Part.from_text(text=intro))

    # ── Examples (video → text pairs) ─────────────────────────────────────
    for i, (row, uri) in enumerate(zip(example_rows, example_uris), start=1):
        parts.append(types.Part.from_uri(file_uri=uri, mime_type="video/mp4"))
        parts.append(
            types.Part.from_text(
                text=_format_example_text(row, i, survey_order=survey_order)
            )
        )

    # ── Target ────────────────────────────────────────────────────────────
    closing = (
        "---\n"
        "Now watch the video below and rate it as the respondent described.\n"
        f"You are simulating a survey respondent with the following profile:\n{demo}\n\n"
        "Rate this video on each of the following 17 items using a 7-point scale.\n"
        f"{scale_instruction}\n"
        f"{scale}"
    )
    parts.append(types.Part.from_text(text=closing))
    parts.append(types.Part.from_uri(file_uri=target_uri, mime_type="video/mp4"))

    return parts
