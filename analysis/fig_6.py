"""
Paper Figure 6: Video-level Pearson ρ heatmaps — Gemini and Qwen (zero-shot).

Outputs → figures/pmsv_corr_agmt_video_gemini.pdf
          figures/pmsv_corr_agmt_video_qwen.pdf

Run from repo root:
    python analysis/fig_6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from profile_ablation import _draw_corr_only, _load_model_data, MODEL_SPECS


def main() -> None:
    for model in ("gemini", "qwen"):
        data = _load_model_data(MODEL_SPECS[model])
        out = _draw_corr_only(model, data)
        print(f"[fig6] → {out}")


if __name__ == "__main__":
    main()
