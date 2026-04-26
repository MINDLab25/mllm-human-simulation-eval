"""
Paper Figure 2: Participant-level ρ and weighted Cohen's κ grid
(Gemini / Qwen × Zero-shot / Few-shot / CoT).

Output → figures/pmsv_corr_agmt.pdf

Run from repo root:
    python analysis/fig_2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figures import parse_args, gen_fig2


def main() -> None:
    args = parse_args()
    gen_fig2(args, Path("figures") / "pmsv_corr_agmt.pdf")


if __name__ == "__main__":
    main()
