# Multimodal PMSV Synthesis

Code and data for the paper: **"Multimodal Large Language Models as Synthetic Participants**

**in Video-Based Studies: An Evaluation"**

We simulate human Perceived Message Sensation Value (PMSV) ratings of Instagram Reels using multimodal LLMs (Google Gemini, Qwen), conditioning on each participant's demographic profile, and compare AI-simulated ratings against real human survey responses across 120 videos and ~1,000 participants.

---

## Repository Structure

```
multimodal-pmsv-synth/
├── main.py                      # Inference CLI (run model on video + participant profile)
├── requirements.txt             # Core inference dependencies
├── requirements-local.txt       # Additional deps for local Qwen GPU
├── .env.example                 # API key template
│
├── src/pmsv_synth/              # Inference engine
│   ├── config.py                # Environment variables and constants
│   ├── data/
│   │   ├── sampler.py           # Sample creation and video path resolution
│   │   └── export.py            # Results to CSV
│   ├── inference/
│   │   ├── gemini/              # Gemini API (sync, batch)
│   │   ├── qwen3_local/         # Local Qwen3-Omni-30B (primary Qwen provider)
│   │   └── qwen_local/          # Local Qwen2.5-Omni (optional)
│   └── prompts/
│       ├── zero_shot.py         # Zero-shot prompt builder
│       ├── few_shot.py          # Few-shot prompt builder
│       ├── cot.py               # Chain-of-thought prompt builder
│       ├── zero_shot_no_profile.py
│       ├── cot_no_profile.py
│       └── prompts.md           # Prompt reference
│
├── data/
│   ├── samples/
│   │   ├── 10_percent_sample.csv    # Canonical 120-video, ~1,000-participant sample
│   │   └── few_shot_examples.csv    # Fixed 3-example pool for few-shot prompts
│   └── results/                     # Model inference outputs (14 conditions)
│       ├── gemini_zero_shot.csv
│       ├── gemini_few_shot.csv
│       ├── gemini_cot.csv
│       ├── gemini_no_profile.csv
│       ├── gemini_shuffled.csv
│       ├── gemini_shuffled_run_2.csv
│       ├── gemini_shuffled_run_3.csv
│       ├── qwen_zero_shot.csv
│       ├── qwen_few_shot.csv
│       ├── qwen_cot.csv
│       ├── qwen_no_profile.csv
│       ├── qwen_shuffled_run_1.csv
│       ├── qwen_shuffled_run_2.csv
│       └── qwen_shuffled_run_3.csv
│
└── analysis/
    ├── figures.py                    # Shared utilities (loaders, stats, constants)
    ├── paper_figures.ipynb           # Notebook: reproduce all paper figures + Table 1
    ├── figs/
    │   ├── fig_2.py                  # → figures/pmsv_corr_agmt.pdf
    │   ├── fig_3.py                  # → figures/pmsv_hists.pdf
    │   ├── fig_4.py                  # → figures/pmsv_age.pdf + pmsv_gender.pdf
    │   ├── fig_5.py                  # → figures/pmsv_dist_shuffle_gemini/qwen.pdf
    │   └── tab_1.py                  # → prints Table 1 (ICC by subgroup)
    └── additional_experiments/
        └── exp_video_features.py     # RF model: video features → PMSV prediction
```

---

## Setup

**Requirements:** Python ≥ 3.11

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```


| Variable         | Description                          |
| ---------------- | ------------------------------------ |
| `GOOGLE_API_KEY` | Gemini API key (aistudio.google.com) |
| `VIDEO_DIR`      | Local path to the video directory    |


> **Note:** Videos are not distributed here due to size and licensing. Contact the authors for access to the full video corpus.

---

## Running Inference

```bash
# Zero-shot Gemini (default)
python main.py

# Few-shot (3 fixed examples)
python main.py --few-shot

# Chain-of-thought
python main.py --cot

# No demographic profile (ablation)
python main.py --no-profile

# Shuffled demographic profile (ablation)
python main.py --shuffle-profile

# Qwen3-Omni-30B on local GPU
python main.py --provider qwen3-local

# Qwen2.5-Omni-7B on local GPU (optional lighter alternative)
python main.py --provider qwen-local
```

**Providers:**


| `--provider`       | Backend                                  | Notes                                         |
| ------------------ | ---------------------------------------- | --------------------------------------------- |
| `gemini` (default) | Google Gemini API                        | Requires `GOOGLE_API_KEY`                     |
| `qwen3-local`      | Local GPU — Qwen3-Omni-30B-A3B-GPTQ-4bit | Requires the `pmsv-qwen3` conda env           |
| `qwen-local`       | Local GPU — Qwen2.5-Omni-7B-GPTQ-Int4    | Optional; requires the `pmsv-local` conda env |


See `src/pmsv_synth/prompts/prompts.md` for the full prompt text.

---

## Data

All data files are subsetted to the 120-video, ~1,000-participant sample used in the paper.


| File                                 | Rows  | Description                                                                       |
| ------------------------------------ | ----- | --------------------------------------------------------------------------------- |
| `data/msv_df.csv`                    | 120   | One row per video — metadata, topic labels, engagement stats, and mean human PMSV |
| `data/msv_df_by_participant.csv`     | 1,010 | One row per participant × video rating — demographics and all 17 PMSV item scores |
| `data/samples/10_percent_sample.csv` | 1,010 | Sample manifest used for all inference runs                                       |
| `data/samples/few_shot_examples.csv` | 3     | Fixed examples used in few-shot prompting                                         |


`data/results/` — model inference outputs (one CSV per condition). Each file has one row per participant × video, with columns:

```
participant_id, video_id, survey_batch_id,
age, gender, race, education, income, sen_seek,
human_perceived_msv, predicted_msv,
human_emotional, human_arousing, ..., human_unusual,   (17 cols)
ai_emotional,    ai_arousing,    ..., ai_unusual        (17 cols)
```

> **Note:** Running new inference with `main.py` requires `data/msv_df.csv` (video list) and the video files (available upon request).

---

## Reproducing Paper Results

All scripts run from the repo root. The notebook is the easiest entry point.

### Notebook

Open `analysis/paper_figures.ipynb` in JupyterLab — it reproduces all figures and Table 1 inline. Works whether launched from the repo root or from `analysis/`.

### Individual scripts


| Script                          | Output                                            | Paper                        |
| ------------------------------- | ------------------------------------------------- | ---------------------------- |
| `python analysis/figs/fig_2.py` | `figures/pmsv_corr_agmt.pdf`                      | Fig 2: ICC grid              |
| `python analysis/figs/fig_3.py` | `figures/pmsv_hists.pdf`                          | Fig 3: PMSV distributions    |
| `python analysis/figs/fig_4.py` | `figures/pmsv_age.pdf`, `figures/pmsv_gender.pdf` | Fig 4: Age and gender        |
| `python analysis/figs/fig_5.py` | `figures/pmsv_dist_shuffle_gemini/qwen.pdf`       | Fig 5: Zero-shot vs shuffled |
| `python analysis/figs/tab_1.py` | stdout                                            | Table 1: ICC by subgroup     |


All scripts use the canonical CSVs in `data/results/` by default. Run any script with `--help` to see path override options.

---

## Citation

```bibtex
@inproceedings{shrestha2026pmsv,
  title     = {Multimodal Large Language Models as Synthetic Participants in Video-Based Studies: An Evaluation},
  author    = {Shrestha, Prabal and Jiang, Bohan and Xue, Haoning and Liu, Huan and Zhou, Xinyi},
  booktitle = {...},
  year      = {2026}
}
```

