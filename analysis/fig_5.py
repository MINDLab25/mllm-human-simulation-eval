"""
Paper Figure 5: Zero-shot vs 3-run shuffled ensemble PMSV distribution.

Outputs → figures/pmsv_dist_real_vs_shuffle_gemini.pdf
          figures/pmsv_dist_real_vs_shuffle_qwen.pdf

Run from repo root:
    python analysis/fig_5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from profile_ablation import _draw_dist_gemini_ensemble, _draw_dist_qwen_ensemble


if __name__ == "__main__":
    for fn in (_draw_dist_gemini_ensemble, _draw_dist_qwen_ensemble):
        out = fn()
        print(f"[fig5] → {out}")
