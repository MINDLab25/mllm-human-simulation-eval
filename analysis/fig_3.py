"""
Paper Figure 3: Participant-level PMSV distributions — Human vs Gemini / Qwen
across zero-shot, few-shot, and CoT conditions.

Output → figures/pmsv_hists.pdf

Run from repo root:
    python analysis/fig_3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figures import parse_args, gen_fig3


def main() -> None:
    args = parse_args()
    gen_fig3(args, Path("figures") / "pmsv_hists.pdf")


if __name__ == "__main__":
    main()
