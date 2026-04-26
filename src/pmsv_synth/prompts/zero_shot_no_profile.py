"""
Zero-shot prompt builder — no demographic profile variant.

Identical to zero_shot.build_user_prompt() except the participant's
demographic profile is omitted.  The model rates the video as a generic
survey respondent, allowing measurement of content-only LLM behaviour
independent of any persona conditioning.
"""

from __future__ import annotations

from pmsv_synth.prompts.zero_shot import _format_pmsv_items


def build_user_prompt_no_profile(survey_order: bool = False) -> str:
    """
    Build the user-turn prompt without any demographic profile.

    Parameters
    ----------
    survey_order : if True, present items with the original Qualtrics anchor
                   directions (including reversed items). If False (default),
                   all items are normalized so 1 = low and 7 = high sensation.
    """
    scale = _format_pmsv_items(survey_order=survey_order)

    if survey_order:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where 1 = the LEFT anchor "
            "and 7 = the RIGHT anchor.\n"
            "Note: some items have the positive/high-sensation quality on the LEFT "
            "(e.g. \"Unique \u2190\u2192 Common\"), so a lower number means MORE of that quality "
            "for those items \u2014 just as in the real survey."
        )
    else:
        scale_instruction = (
            "For each item, assign an integer from 1 to 7 where "
            "1 = least sensational and 7 = most sensational."
        )

    return (
        "You are a survey respondent.\n\n"
        "You just watched a short video on Instagram Reels.\n\n"
        "Read the question below and answer as this respondent would. "
        "Follow the response instructions precisely.\n\n"
        "Rate this video on each of the following 17 items using a 7-point scale.\n"
        f"{scale_instruction}\n"
        f"{scale}"
    )
