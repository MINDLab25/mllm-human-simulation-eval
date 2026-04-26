"""
Paper Figure 4: PMSV by demographic group (zero-shot).
Age (median split at 39) and gender boxplots for Human / Gemini / Qwen.

Outputs → figures/pmsv_age.pdf
          figures/pmsv_gender.pdf

Run from repo root:
    python analysis/fig_4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figures import parse_args, gen_fig4


def main() -> None:
    args = parse_args()
    gen_fig4(
        args,
        Path("figures") / "pmsv_age.pdf",
        Path("figures") / "pmsv_gender.pdf",
    )


if __name__ == "__main__":
    main()
