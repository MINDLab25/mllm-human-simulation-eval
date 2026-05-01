"""
Paper Figure 3: Participant-level PMSV distributions — Human vs Gemini / Qwen
across zero-shot, few-shot, and CoT conditions.

Output → figures/pmsv_hists.pdf

Run from repo root:
    python analysis/figs/fig_3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent.parent))
from figures import FS, GEMINI_COLOR, LW, QWEN_COLOR, load_participant_agg, parse_args

HUMAN_COLOR = "#bec5cc"
GEMINI_DARK = "#4fb5ef"
QWEN_DARK   = "#a875d6"


def generate(args, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    bins   = np.arange(0.75, 7.5, 0.5)
    x_grid = np.linspace(1, 7, 300)
    lw     = LW["regression"] * 0.8

    STYLES = [
        {"linestyle": "-",          "linewidth": lw,       "label": "Zero-shot"},
        {"linestyle": (0, (6, 2)),  "linewidth": lw,       "label": "Few-shot"},
        {"linestyle": (0, (1, 1)),  "linewidth": lw * 0.8, "label": "CoT"},
    ]

    def _load(csv_path):
        df = load_participant_agg(Path(csv_path)).dropna(subset=["human_msv", "ai_msv"])
        return df["human_msv"].values, df["ai_msv"].values

    def _ms(arr):
        arr = np.asarray(arr, dtype=float)
        valid = arr[~np.isnan(arr)]
        return np.mean(valid), np.std(valid, ddof=1)

    human_g, g_zero = _load(args.csv)
    _,       g_fews = _load(args.fews_csv)
    _,       g_cot  = _load(args.cot_csv)
    human_q, q_zero = _load(args.qwen_csv)
    _,       q_fews = _load(args.qwen_fews_csv)
    _,       q_cot  = _load(args.qwen_cot_csv)

    def _draw_panel(ax, human, ai_list, color, xlabel, title):
        ax.hist(human, bins=bins, density=True,
                color=HUMAN_COLOR, alpha=0.65,
                edgecolor="white", linewidth=0.8, rwidth=0.96)
        for ai_vals, style in zip(ai_list, STYLES):
            kde = gaussian_kde(ai_vals, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), color=color, zorder=3, **style)
        ax.set_xlim(0.5, 7.5)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_ylabel("Density", fontsize=FS["axis_label"])
        ax.set_xlabel(xlabel,    fontsize=FS["axis_label"])
        ax.set_title(title,      fontsize=FS["title"], fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        h_mu,  h_sd  = _ms(human)
        zs_mu, zs_sd = _ms(ai_list[0])
        fs_mu, fs_sd = _ms(ai_list[1])
        ct_mu, ct_sd = _ms(ai_list[2])
        human_handle = mpatches.Patch(color=HUMAN_COLOR, alpha=0.65)
        line_handles = [
            mlines.Line2D([], [], color=color,
                          linestyle=s["linestyle"], linewidth=s["linewidth"])
            for s in STYLES
        ]
        ax.legend(
            [human_handle] + line_handles,
            [
                f"Human ({h_mu:.2f}±{h_sd:.2f})",
                f"ZS ({zs_mu:.2f}±{zs_sd:.2f})",
                f"FS ({fs_mu:.2f}±{fs_sd:.2f})",
                f"CoT ({ct_mu:.2f}±{ct_sd:.2f})",
            ],
            loc="upper right",
            bbox_to_anchor=(1.04, 0.95),
            borderaxespad=0.0,
            fontsize=34,
            framealpha=0.0,
            edgecolor="none",
            handlelength=1.6,
            handletextpad=0.5,
            borderpad=0.2,
            labelspacing=0.6,
        )

    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    _draw_panel(axes[0], human_g, [g_zero, g_fews, g_cot],
                GEMINI_DARK, "PMSV", "Human vs Gemini")
    _draw_panel(axes[1], human_q, [q_zero, q_fews, q_cot],
                QWEN_DARK,   "PMSV", "Human vs Qwen")
    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(top=0.95, bottom=0.08, right=0.97, left=0.1)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig3] → {out}")


def main() -> None:
    generate(parse_args(), Path("figures") / "pmsv_hists.pdf")


if __name__ == "__main__":
    main()
