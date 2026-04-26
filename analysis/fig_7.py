"""
Paper Figure 7: Residual 2D-density heatmaps (human vs model, zero-shot).

Outputs → figures/pmsv_corr_agmt_residual_gemini.pdf
          figures/pmsv_corr_agmt_residual_qwen.pdf

Run from repo root:
    python analysis/fig_7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from profile_ablation import _draw_residual_only


def main() -> None:
    for model in ("gemini", "qwen"):
        out = _draw_residual_only(model)
        print(f"[fig7] → {out}")


if __name__ == "__main__":
    main()
