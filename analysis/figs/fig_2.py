"""
Paper Figure 2: Participant-level ICC grid
(Gemini / Qwen × Zero-shot / Few-shot / CoT).

Output → figures/pmsv_corr_agmt.pdf

Run from repo root:
    python analysis/figs/fig_2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from figures import (
    FS,
    GEMINI_COLOR,
    LW,
    QWEN_COLOR,
    _icc_a1,
    _round_clip,
    load_participant_agg,
    parse_args,
)


def _icc_grid(conditions_data: list) -> plt.Figure:
    """2 × N grid: rows = [Gemini, Qwen], cols = conditions.
    `conditions_data`: list of (label, df_g, df_q, icc_hg, icc_hq).
    """
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    n_cols = len(conditions_data)
    ratings = np.arange(1, 8)

    def _mat_pct(human_rc, ai_rc):
        mat = np.zeros((7, 7), dtype=int)
        for h, a in zip(human_rc, ai_rc):
            mat[a - 1, h - 1] += 1
        return mat / mat.sum() * 100

    panels = []
    for label, df_g, df_q, icc_hg, icc_hq in conditions_data:
        gem = df_g.dropna(subset=["human_msv", "ai_msv"])
        qwn = df_q.dropna(subset=["human_msv", "ai_msv"])
        h_g = gem["human_msv"].to_numpy(dtype=float)
        a_g = gem["ai_msv"].to_numpy(dtype=float)
        h_q = qwn["human_msv"].to_numpy(dtype=float)
        a_q = qwn["ai_msv"].to_numpy(dtype=float)
        panels.append({
            "label":  label,
            "gemini": (h_g, a_g, _mat_pct(_round_clip(h_g), _round_clip(a_g)),
                       GEMINI_COLOR, icc_hg),
            "qwen":   (h_q, a_q, _mat_pct(_round_clip(h_q), _round_clip(a_q)),
                       QWEN_COLOR,   icc_hq),
        })

    fig = plt.figure(figsize=(4.4 * n_cols + 0.8, 10.5))
    gs   = GridSpec(2, n_cols, figure=fig, hspace=0.06, wspace=0.08)
    axes = np.empty((2, n_cols), dtype=object)
    for r in range(2):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    last_im = [None, None]
    for col, panel in enumerate(panels):
        for row, model_key in enumerate(["gemini", "qwen"]):
            ax = axes[row, col]
            human, ai, mat, color, kappa = panel[model_key]

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom", ["#f7f7f7", color], N=256
            )
            im = ax.imshow(mat, cmap=cmap, aspect="equal",
                           vmin=0, vmax=15, origin="lower",
                           extent=[-0.5, 6.5, -0.5, 6.5])
            last_im[row] = im

            ax.plot([-0.5, 6.5], [-0.5, 6.5],
                    color="#aaaaaa", linestyle="--",
                    linewidth=LW["regression"] * 0.45, zorder=5)
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels(ratings)
            ax.set_yticklabels(ratings)
            ax.tick_params(axis="both", labelsize=FS["tick"])
            if row != 1:
                ax.tick_params(axis="x", labelbottom=False)
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel("Human" if row == 1 else "", fontsize=FS["axis_label"], labelpad=8)
            ax.set_ylabel("")
            ax.set_title(f"ICC = {kappa:.2f}", fontsize=FS["annotation"],
                         fontweight="bold", pad=4)

    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.88)

    for row in range(2):
        row_pos = [axes[row, c].get_position() for c in range(n_cols)]
        ax_y0, ax_y1 = row_pos[0].y0, row_pos[0].y1
        cbar_ax = fig.add_axes([row_pos[-1].x1 + 0.012, ax_y0, 0.012, ax_y1 - ax_y0])
        cbar = fig.colorbar(last_im[row], cax=cbar_ax)
        cbar.set_ticks([0, 5, 10, 15])
        cbar.set_ticklabels(["0", "5", "10", "15"])
        cbar.ax.tick_params(labelsize=FS["tick"])
        cbar.set_label("% of participants", fontsize=FS["annotation"])

    for row, row_label in enumerate(["Gemini", "Qwen"]):
        row_axes = [axes[row, c].get_position() for c in range(n_cols)]
        y_mid   = (row_axes[0].y0 + row_axes[0].y1) / 2.0
        fig.text(row_axes[0].x0 - 0.045, y_mid, row_label,
                 ha="center", va="center", fontsize=FS["axis_label"], rotation=90)

    for col, panel in enumerate(panels):
        bbox = axes[0, col].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2.0, 0.965, panel["label"],
                 ha="center", va="top", fontsize=FS["title"])

    return fig


def generate(args, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    conditions = []
    for label, gcsv, qcsv in [
        ("Zero-shot", args.csv,      args.qwen_csv),
        ("Few-shot",  args.fews_csv, args.qwen_fews_csv),
        ("CoT",       args.cot_csv,  args.qwen_cot_csv),
    ]:
        gp, qp = Path(gcsv), Path(qcsv)
        if gp.exists() and qp.exists():
            conditions.append((label, gp, qp))

    if not conditions:
        import sys as _sys
        print("[fig2] No valid conditions found.", file=_sys.stderr)
        return

    grid_data = []
    for label, gcsv, qcsv in conditions:
        df_g = load_participant_agg(gcsv).dropna(subset=["human_msv", "ai_msv"])
        df_q = load_participant_agg(qcsv).dropna(subset=["human_msv", "ai_msv"])
        grid_data.append((
            label, df_g, df_q,
            _icc_a1(df_g["human_msv"].values, df_g["ai_msv"].values),
            _icc_a1(df_q["human_msv"].values, df_q["ai_msv"].values),
        ))

    fig = _icc_grid(grid_data)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig2] → {out}")


def main() -> None:
    generate(parse_args(), Path("figures") / "pmsv_corr_agmt.pdf")


if __name__ == "__main__":
    main()
