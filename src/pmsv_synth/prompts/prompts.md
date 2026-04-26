# PMSV Inference Prompts

Reference for how prompts are constructed when calling the LLM.
Placeholder values in `{curly braces}` are filled at runtime.

---

## Shared: Demographic Label Mappings

**Education**

| Code | Label |
| ---- | ----- |
| 1    | Some high school or less |
| 2    | High school diploma or GED |
| 3    | Some college, but no degree |
| 4    | Associates or technical degree |
| 5    | Bachelor's degree |
| 6    | Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS etc.) |
| NA   | Prefer not to say |

**Household Income**

| Code | Label |
| ---- | ----- |
| 1    | Less than $25,000 |
| 2    | $25,000–$49,999 |
| 3    | $50,000–$74,999 |
| 4    | $75,000–$99,999 |
| 5    | $100,000–$149,999 |
| 6    | $150,000 or more |
| NA   | Prefer not to say |

---

## Zero-Shot (`prompts/zero_shot.py`)

**Source:** Follows Kolluri et al. (2025) §D prompt structure.  
**Model:** `models/gemini-3-flash-preview`  
**Config:** `temperature=0.0`, `response_mime_type=application/json`, `response_schema=PMSVRatings`

Output format is enforced via the `PMSVRatings` Pydantic schema passed as `response_schema`. No JSON example appears in the prompt text (per google-genai docs: duplicating the schema degrades output quality).

### System message

```
You are simulating a survey respondent. Answer exactly as instructed, following the specified response format without additional commentary.
```

### User message

```
You are a survey respondent with the following demographic profile:
- Age: {age}
- Gender: {gender}
- Race/Ethnicity: {race}
- Education: {education_label}
- Household Income: {income_label}
- Sensation-Seeking Score: {sen_seek} (scale 1–5, higher = more sensation-seeking)

You just watched a short video on Instagram Reels.

Read the question below and answer exactly as this person would. Follow the response instructions precisely.

Rate this video on each of the following 17 items using a 7-point scale.
For each item, assign an integer from 1 to 7 where 1 = the LEFT anchor and 7 = the RIGHT anchor.

[Emotional Arousal]
  emotional:          Unemotional (1) ←→ (7) Emotional
  arousing:           Not arousing (1) ←→ (7) Arousing
  involving:          Uninvolving (1) ←→ (7) Involving
  exciting:           Boring (1) ←→ (7) Exciting
  powerful_impact:    Weak impact (1) ←→ (7) Powerful impact
  stimulating:        Not stimulating (1) ←→ (7) Stimulating
  strong_visual:      Weak visuals (1) ←→ (7) Strong visuals
  strong_soundeffect: Weak sound effects (1) ←→ (7) Strong sound effects

[Dramatic Impact]
  dramatic:           Undramatic (1) ←→ (7) Dramatic
  graphic:            Not graphic (1) ←→ (7) Graphic
  creative:           Not creative (1) ←→ (7) Creative
  goosebump:          Didn't give me goosebumps (1) ←→ (7) Gave me goosebumps
  intense:            Not intense (1) ←→ (7) Intense
  strong_soundtrack:  Weak soundtrack (1) ←→ (7) Strong soundtrack

[Novelty]
  novel:              Ordinary (1) ←→ (7) Novel
  unique:             Common (1) ←→ (7) Unique
  unusual:            Usual (1) ←→ (7) Unusual
```

### Response schema (`PMSVRatings` Pydantic model)

```python
class PMSVRatings(BaseModel):
    # Emotional Arousal (8 items)
    emotional:          int = Field(..., ge=1, le=7)
    arousing:           int = Field(..., ge=1, le=7)
    involving:          int = Field(..., ge=1, le=7)
    exciting:           int = Field(..., ge=1, le=7)
    powerful_impact:    int = Field(..., ge=1, le=7)
    stimulating:        int = Field(..., ge=1, le=7)
    strong_visual:      int = Field(..., ge=1, le=7)
    strong_soundeffect: int = Field(..., ge=1, le=7)
    # Dramatic Impact (6 items)
    dramatic:           int = Field(..., ge=1, le=7)
    graphic:            int = Field(..., ge=1, le=7)
    creative:           int = Field(..., ge=1, le=7)
    goosebump:          int = Field(..., ge=1, le=7)
    intense:            int = Field(..., ge=1, le=7)
    strong_soundtrack:  int = Field(..., ge=1, le=7)
    # Novelty (3 items)
    novel:              int = Field(..., ge=1, le=7)
    unique:             int = Field(..., ge=1, le=7)
    unusual:            int = Field(..., ge=1, le=7)
```

`predicted_msv` = mean of all 17 ratings (range 1–7).

---

## Few-Shot (`prompts/few_shot.py`)

**Source:** `prompts/few_shot.py` → `build_few_shot_contents()`  
**Model:** `models/gemini-3-flash-preview`  
**Config:** `temperature=0.0`, `response_mime_type=application/json`, `response_schema=PMSVRatings`

For each target (video, participant) pair, N real survey responses from **other** videos are randomly sampled and prepended as examples. Each example is an interleaved (video clip + demographic profile + ratings) multi-part message. The same N example videos are reused for all participants watching the same target video (videos are uploaded once).

### Message structure (multi-part `contents` list)

```
Part 1 [text] — Intro
Part 2 [video] — Example 1 video
Part 3 [text]  — Example 1: respondent profile + ratings JSON
...
Part 2N [video] — Example N video
Part 2N+1 [text] — Example N: respondent profile + ratings JSON
Part 2N+2 [text] — Separator + target respondent profile + scale
Part 2N+3 [video] — Target video
```

### Part 1 — Intro text

```
You are a survey respondent. You will first see {N} example video(s) together with how a real
person rated each one. Then you will watch a new video and rate it yourself.

Use the examples to calibrate the scale — note how the ratings relate to what you see in each video.
```

### Parts 2…2N+1 — Example blocks (per example)

Each example is a video part immediately followed by:

```
=== Example {i} ===
Respondent profile:
- Age: {age}
- Gender: {gender}
- Race/Ethnicity: {race}
- Education: {education_label}
- Household Income: {income_label}
- Sensation-Seeking Score: {sen_seek} (scale 1–5, higher = more sensation-seeking)

Survey ratings for the video above:
{
  "emotional": ...,
  "arousing": ...,
  ...
}
```

### Parts 2N+2…2N+3 — Target (closing text + target video)

```
---
Now watch the video below and rate it as the respondent described.
You are simulating a survey respondent with the following profile:
- Age: {age}
- Gender: {gender}
- Race/Ethnicity: {race}
- Education: {education_label}
- Household Income: {income_label}
- Sensation-Seeking Score: {sen_seek} (scale 1–5, higher = more sensation-seeking)

Rate this video on each of the following 17 items using a 7-point scale.
For each item, assign an integer from 1 to 7 where 1 represents the left anchor and 7 represents the right anchor.

[Emotional Arousal]
  emotional / arousing / involving / exciting / powerful_impact /
  stimulating / strong_visual / strong_soundeffect

[Dramatic Impact]
  dramatic / graphic / creative / goosebump / intense / strong_soundtrack

[Novelty]
  novel / unique / unusual
```

Followed immediately by the target video part.

---

## Chain-of-Thought (CoT) (`prompts/cot.py`)

**Source:** `prompts/cot.py` → `build_cot_user_prompt()` / `build_cot_few_shot_contents()`  
**Model:** `models/gemini-3-flash-preview`  
**Config:** `temperature=0.0`, plain text output (no `response_schema`)

The CoT prompt appends a structured reasoning instruction (`COT_SUFFIX`) to the same demographic profile + scale used in Zero-Shot. Because the output mixes free-text reasoning and a JSON block, structured-JSON mode cannot be used. The JSON block is extracted from the response with a regex (`parse_cot_response()`).

### System message

*(same as Zero-Shot)*

```
You are simulating a survey respondent. Answer exactly as instructed, following the specified response format without additional commentary.
```

### User message

```
You are a survey respondent with the following demographic profile:
- Age: {age}
- Gender: {gender}
- Race/Ethnicity: {race}
- Education: {education_label}
- Household Income: {income_label}
- Sensation-Seeking Score: {sen_seek} (scale 1–5, higher = more sensation-seeking)

You just watched a short video on Instagram Reels.

Read the question below and answer exactly as this person would. Follow the response instructions precisely.

Rate this video on each of the following 17 items using a 7-point scale.
For each item, assign an integer from 1 to 7 where 1 = the LEFT anchor and 7 = the RIGHT anchor.

[Emotional Arousal]
  emotional:          Unemotional (1) ←→ (7) Emotional
  arousing:           Not arousing (1) ←→ (7) Arousing
  involving:          Uninvolving (1) ←→ (7) Involving
  exciting:           Boring (1) ←→ (7) Exciting
  powerful_impact:    Weak impact (1) ←→ (7) Powerful impact
  stimulating:        Not stimulating (1) ←→ (7) Stimulating
  strong_visual:      Weak visuals (1) ←→ (7) Strong visuals
  strong_soundeffect: Weak sound effects (1) ←→ (7) Strong sound effects

[Dramatic Impact]
  dramatic:           Undramatic (1) ←→ (7) Dramatic
  graphic:            Not graphic (1) ←→ (7) Graphic
  creative:           Not creative (1) ←→ (7) Creative
  goosebump:          Didn't give me goosebumps (1) ←→ (7) Gave me goosebumps
  intense:            Not intense (1) ←→ (7) Intense
  strong_soundtrack:  Weak soundtrack (1) ←→ (7) Strong soundtrack

[Novelty]
  novel:              Ordinary (1) ←→ (7) Novel
  unique:             Common (1) ←→ (7) Unique
  unusual:            Usual (1) ←→ (7) Unusual

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
}
```

### Response parsing

Because the output is plain text, `parse_cot_response()` in `cot.py` extracts:
- **Ratings**: the JSON block after `Final ratings:` using `r"[Ff]inal\s+ratings\s*:\s*(\{[^{}]*\})"`
- **Reasoning**: the text block after `Reasoning:` using `r"[Rr]easoning\s*:\s*(.*?)(?=\n[Ff]inal\s+ratings\s*:|$)"`

### CoT + Few-shot variant (`build_cot_few_shot_contents()`)

Uses the same interleaved video+text structure as the Few-Shot prompt above, but replaces the bare scale at the end with the full `COT_SUFFIX` reasoning instruction block.
