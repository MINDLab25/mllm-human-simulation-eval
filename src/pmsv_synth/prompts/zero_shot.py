"""
Zero-shot prompt builder for PMSV prediction.

Follows the Socrates paper (Kolluri et al., 2025) prompt structure (§D):
  - System message: frame the model as a survey respondent
  - User message: demographic profile + stimulus (video) + response instructions

PMSV scale: Palmgreen, Stephenson, Everett, Baseheart & Francies (2002).
Health Communication, 14(4), 403–428.
17 semantic-differential items (1–7) across three dimensions:
  emotional arousal, dramatic impact, and novelty.
"""

from __future__ import annotations

import math

from typing import Any

from pydantic import BaseModel, Field


class PMSVRatings(BaseModel):
    """Pydantic schema enforcing all 17 PMSV item ratings as integers 1–7.

    Field names match the dataset column names in msv_df_by_participant.csv.
    """

    # Emotional Arousal (8 items)
    emotional: int = Field(..., ge=1, le=7)
    arousing: int = Field(..., ge=1, le=7)
    involving: int = Field(..., ge=1, le=7)
    exciting: int = Field(..., ge=1, le=7)
    powerful_impact: int = Field(..., ge=1, le=7)
    stimulating: int = Field(..., ge=1, le=7)
    strong_visual: int = Field(..., ge=1, le=7)
    strong_soundeffect: int = Field(..., ge=1, le=7)
    # Dramatic Impact (6 items)
    dramatic: int = Field(..., ge=1, le=7)
    graphic: int = Field(..., ge=1, le=7)
    creative: int = Field(..., ge=1, le=7)
    goosebump: int = Field(..., ge=1, le=7)
    intense: int = Field(..., ge=1, le=7)
    strong_soundtrack: int = Field(..., ge=1, le=7)
    # Novelty (3 items)
    novel: int = Field(..., ge=1, le=7)
    unique: int = Field(..., ge=1, le=7)
    unusual: int = Field(..., ge=1, le=7)


# ---------------------------------------------------------------------------
# PMSV scale — survey order and anchors (Palmgreen et al., 2002)
# Each tuple: (item_id, left_anchor, right_anchor, dimension)
# item_id matches the column name in msv_df_by_participant.csv
#
# PMSV_ITEMS            — exact Qualtrics survey order and anchor directions.
#                         REVERSED items have the high-sensation anchor on the
#                         LEFT (1 = more sensational for those items).
#                         Use with reverse_score() after inference.
#
# PMSV_ITEMS_NORMALIZED — same survey order, but reversed items have their
#                         anchors flipped so that 1 = low sensation and
#                         7 = high sensation for every item uniformly.
#                         No post-hoc reverse_score() call is needed.
# ---------------------------------------------------------------------------
PMSV_ITEMS: list[tuple[str, str, str, str]] = [
    # (item_id, left_anchor, right_anchor, dimension)
    # * marks reversed items (1 = more sensational on this scale)
    ("unique",            "Unique",                   "Common",              "Novelty"),           # * reversed
    ("powerful_impact",   "Powerful impact",          "Weak impact",         "Emotional Arousal"), # * reversed
    ("goosebump",         "Didn't give me goosebumps","Gave me goosebumps",  "Dramatic Impact"),   # normal
    ("novel",             "Novel",                    "Ordinary",            "Novelty"),            # * reversed
    ("emotional",         "Emotional",                "Unemotional",         "Emotional Arousal"), # * reversed
    ("exciting",          "Boring",                   "Exciting",            "Emotional Arousal"), # normal
    ("strong_visual",     "Strong visuals",           "Weak visuals",        "Emotional Arousal"), # * reversed
    ("creative",          "Not creative",             "Creative",            "Dramatic Impact"),   # normal
    ("graphic",           "Not graphic",              "Graphic",             "Dramatic Impact"),   # normal
    ("arousing",          "Arousing",                 "Not arousing",        "Emotional Arousal"), # * reversed
    ("unusual",           "Unusual",                  "Usual",               "Novelty"),            # * reversed
    ("involving",         "Involving",                "Uninvolving",         "Emotional Arousal"), # * reversed
    ("intense",           "Not intense",              "Intense",             "Dramatic Impact"),   # normal
    ("strong_soundtrack", "Weak soundtrack",          "Strong soundtrack",   "Dramatic Impact"),   # normal
    ("dramatic",          "Undramatic",               "Dramatic",            "Dramatic Impact"),   # normal
    ("stimulating",       "Stimulating",              "Not stimulating",     "Emotional Arousal"), # * reversed
    ("strong_soundeffect","Strong sound effects",     "Weak sound effects",  "Emotional Arousal"), # * reversed
]

# Items where the high-sensation quality is on the LEFT (anchor 1).
# Only relevant when using PMSV_ITEMS (survey order). After inference,
# call reverse_score() to flip these so all values are high = more sensational.
REVERSED_ITEMS: frozenset[str] = frozenset({
    "unique",
    "powerful_impact",
    "novel",
    "emotional",
    "strong_visual",
    "arousing",
    "unusual",
    "involving",
    "stimulating",
    "strong_soundeffect",
})

# Normalized version: same survey order, reversed items have anchors swapped
# so that 1 = low sensation and 7 = high sensation for every item uniformly.
PMSV_ITEMS_NORMALIZED: list[tuple[str, str, str, str]] = [
    (item_id, (right if item_id in REVERSED_ITEMS else left), (left if item_id in REVERSED_ITEMS else right), dim)
    for item_id, left, right, dim in PMSV_ITEMS
]


def reverse_score(raw_ratings: dict[str, int]) -> dict[str, int]:
    """
    Flip reversed items so all values are in the high = more-sensational direction.

    Only needed when inference was run with survey-order (reversed) items.
    For each reversed item: score = 8 − raw_score.
    Non-reversed items are returned unchanged.
    """
    return {
        item_id: (8 - score) if item_id in REVERSED_ITEMS else score
        for item_id, score in raw_ratings.items()
    }

SYSTEM_PROMPT = (
    "You are simulating a survey respondent. "
    "Answer exactly as instructed, following the specified response format "
    "without additional commentary."
)


def _format_demographic_profile(participant: dict[str, Any]) -> str:
    """Format participant demographics into the Socrates-style bullet list."""
    edu_map = {
        1: "Some high school or less",
        2: "High school diploma or GED",
        3: "Some college, but no degree",
        4: "Associates or technical degree",
        5: "Bachelor's degree",
        6: "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS etc.)",
        "NA": "Prefer not to say",
    }
    income_map = {
        1: "Less than $25,000",
        2: "$25,000–$49,999",
        3: "$50,000–$74,999",
        4: "$75,000–$99,999",
        5: "$100,000–$149,999",
        6: "$150,000 or more",
        "NA": "Prefer not to say",
    }

    edu_raw = participant.get("education")
    income_raw = participant.get("income")

    def _lookup(mapping: dict, raw: Any) -> str:
        if raw is None or raw != raw:  # None or NaN
            return "Unknown"
        if str(raw).strip().upper() == "NA":
            return mapping.get("NA", "Prefer not to say")
        try:
            return mapping.get(int(raw), str(raw))
        except (ValueError, TypeError):
            return str(raw)

    edu_label = _lookup(edu_map, edu_raw)
    income_label = _lookup(income_map, income_raw)

    lines = [
        f"- Age: {int(participant['age'])}",
        f"- Gender: {participant['gender']}",
        f"- Race/Ethnicity: {participant['race']}",
        f"- Education: {edu_label}",
        f"- Household Income: {income_label}",
        f"- Sensation-Seeking Score: {float(participant['sen_seek']):.2f} (scale 1–5, higher = more sensation-seeking)",
    ]
    return "\n".join(lines)


def _format_pmsv_items(survey_order: bool = False) -> str:
    """Render the 17-item PMSV scale.

    Parameters
    ----------
    survey_order : if True, use the original Qualtrics anchor directions
                   (some items reversed). If False (default), use the
                   normalized version where 1 = low and 7 = high for all items.
    """
    items = PMSV_ITEMS if survey_order else PMSV_ITEMS_NORMALIZED
    lines: list[str] = []
    current_dim = ""
    for item_id, left, right, dim in items:
        if dim != current_dim:
            lines.append(f"\n[{dim}]")
            current_dim = dim
        lines.append(f"  {item_id}: {left} (1) ←→ (7) {right}")
    return "\n".join(lines)


def build_user_prompt(participant: dict[str, Any], survey_order: bool = False) -> str:
    """
    Build the user-turn prompt for one participant × video combination.

    Parameters
    ----------
    participant  : row from msv_df_by_participant (as a dict)
    survey_order : if True, present items with the original Qualtrics anchor
                   directions (including reversed items). If False (default),
                   all items are normalized so 1 = low and 7 = high sensation.
    """
    demo  = _format_demographic_profile(participant)
    scale = _format_pmsv_items(survey_order=survey_order)

    if survey_order:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where 1 = the LEFT anchor "
            "and 7 = the RIGHT anchor.\n"
            "Note: some items have the positive/high-sensation quality on the LEFT "
            "(e.g. \"Unique ←→ Common\"), so a lower number means MORE of that quality "
            "for those items — just as in the real survey."
        )
    else:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where "
            "1 = least sensational and 7 = most sensational."
        )

    return (
        f"You are a survey respondent with the following demographic profile:\n"
        f"{demo}\n\n"
        f"You just watched a short video on Instagram Reels.\n\n"
        f"Read the question below and answer exactly as this person would. "
        f"Follow the response instructions precisely.\n\n"
        f"Rate this video on each of the following 17 items using a 7-point scale.\n"
        f"{scale_instruction}\n"
        f"{scale}"
    )


def get_item_ids() -> list[str]:
    """Return the ordered list of PMSV item IDs."""
    return [item[0] for item in PMSV_ITEMS]
