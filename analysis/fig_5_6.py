"""
Paper Figure 5: Distribution of Qwen-predicted PMSVs with real vs shuffled
participant profiles (zero-shot).

Paper Figure 6: Video-level Pearson ρ and weighted Cohen's κ
(human vs Gemini PMSVs, zero-shot).

Outputs → figures/pmsv_dist_real_vs_shuffle_qwen.pdf
          figures/pmsv_corr_agmt_video_gemini.pdf

Run from repo root:
    python analysis/fig_5_6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from profile_ablation import main

if __name__ == "__main__":
    main()
