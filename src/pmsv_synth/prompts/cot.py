"""
Chain-of-thought (CoT) prompt builder for PMSV prediction.

Forces the model to reason through its assessment step by step before
giving the final ratings.  Because the output mixes free text (reasoning)
and JSON (ratings), structured-JSON mode cannot be used; instead the
response is plain text and the JSON block is extracted with a regex.
"""

from __future__ import annotations

import json
import re
from typing import Any

from google.genai import types

from pmsv_synth.prompts.zero_shot import (
    _format_demographic_profile,
    _format_pmsv_items,
)

# ---------------------------------------------------------------------------
# CoT instruction block — appended to any user prompt
# ---------------------------------------------------------------------------

COT_SUFFIX = """\
Before giving the final ratings, follow these steps:

Step 1: Summarize the video's main characteristics.
Step 2: Explain how this respondent would likely react to it.
Step 3: Explain how you map that reaction to the rating scales.
Step 4: Give the final ratings.

Use this exact output format:

Reasoning:
1. ...
2. ...
3. ...

Final ratings:
{
  "emotional": ...,
  "arousing": ...,
  "involving": ...,
  "exciting": ...,
  "powerful_impact": ...,
  "stimulating": ...,
  "strong_visual": ...,
  "strong_soundeffect": ...,
  "dramatic": ...,
  "graphic": ...,
  "creative": ...,
  "goosebump": ...,
  "intense": ...,
  "strong_soundtrack": ...,
  "novel": ...,
  "unique": ...,
  "unusual": ...
}"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_cot_user_prompt(participant: dict[str, Any], survey_order: bool = False) -> str:
    """Build the CoT user prompt for one participant × video combination.

    Parameters
    ----------
    survey_order : if True, present items with original Qualtrics anchor
                   directions (reversed items included). If False (default),
                   all items are normalized so 1 = low and 7 = high sensation.
    """
    demo  = _format_demographic_profile(participant)
    scale = _format_pmsv_items(survey_order=survey_order)

    if survey_order:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where 1 = the LEFT anchor "
            "and 7 = the RIGHT anchor.\n"
            "Note: some items have the positive/high-sensation quality on the LEFT "
            "(e.g. \"Unique \u2190\u2192 Common\"), so a lower number means MORE of "
            "that quality for those items \u2014 just as in the real survey."
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
        f"{scale}\n\n"
        f"{COT_SUFFIX}"
    )


def build_cot_few_shot_contents(
    participant: dict[str, Any],
    target_uri: str,
    example_rows: list[dict[str, Any]],
    example_uris: list[str],
    survey_order: bool = False,
) -> list[types.Part]:
    """
    Build the multi-part contents list for a CoT few-shot inference call.

    Structure mirrors few_shot.build_few_shot_contents() but appends the
    CoT instruction block instead of requesting bare JSON output.

    Parameters
    ----------
    survey_order : if True, use original Qualtrics anchor directions; if False
                   (default) use normalized 1 = low, 7 = high for all items.
    """
    from pmsv_synth.prompts.few_shot import _format_example_text

    scale  = _format_pmsv_items(survey_order=survey_order)
    demo   = _format_demographic_profile(participant)
    n      = len(example_rows)
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

    intro = (
        f"You are a survey respondent. You will first see {n} example video{plural} "
        f"together with how a real person rated each one. Then you will watch a new "
        f"video and rate it yourself.\n\n"
        f"Use the examples to calibrate the scale \u2014 note how the ratings "
        f"relate to what you see in each video."
    )
    parts.append(types.Part.from_text(text=intro))

    for i, (row, uri) in enumerate(zip(example_rows, example_uris), start=1):
        parts.append(types.Part.from_uri(file_uri=uri, mime_type="video/mp4"))
        parts.append(types.Part.from_text(text=_format_example_text(row, i)))

    closing = (
        "---\n"
        "Now watch the video below and rate it as the respondent described.\n"
        f"You are simulating a survey respondent with the following profile:\n{demo}\n\n"
        "Rate this video on each of the following 17 items using a 7-point scale.\n"
        f"{scale_instruction}\n"
        f"{scale}\n\n"
        f"{COT_SUFFIX}"
    )
    parts.append(types.Part.from_text(text=closing))
    parts.append(types.Part.from_uri(file_uri=target_uri, mime_type="video/mp4"))

    return parts


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_FINAL_RATINGS_RE = re.compile(
    r"[Ff]inal\s+ratings\s*:\s*(\{[^{}]*\})",
    re.DOTALL,
)
_REASONING_RE = re.compile(
    r"[Rr]easoning\s*:\s*(.*?)(?=\n[Ff]inal\s+ratings\s*:|$)",
    re.DOTALL,
)


def parse_cot_response(text: str) -> tuple[dict[str, int] | None, str | None]:
    """
    Extract ratings dict and reasoning text from a CoT model response.

    Returns
    -------
    (ratings_dict, reasoning_text)
        ratings_dict : {item_id: int} with raw (pre-reverse-score) values,
                       or None if the JSON block could not be parsed.
        reasoning_text : the Reasoning section as a string, or None.
    """
    ratings: dict[str, int] | None = None
    reasoning: str | None = None

    m_json = _FINAL_RATINGS_RE.search(text)
    if m_json:
        try:
            raw = json.loads(m_json.group(1))
            ratings = {k: int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            ratings = None

    m_reason = _REASONING_RE.search(text)
    if m_reason:
        reasoning = m_reason.group(1).strip()

    return ratings, reasoning
