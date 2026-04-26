"""Generate participant-level kappa/corr heatmap + PMSV distribution
(real profile vs shuffled profile) figures.

For each selected model, three figures are produced:
  * combined 1x2 panel (heatmap + distribution)
  * standalone heatmap
  * standalone distribution

Run:
    python kappa_corr_dist_participant.py --model qwen
    python kappa_corr_dist_participant.py --model gemini
    python kappa_corr_dist_participant.py --model both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

import pandas as pd

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent))

from figures import (
    FS,
    GEMINI_COLOR,
    LW,
    QWEN_COLOR,
    _ci_band,
    _round_clip,
    _weighted_kappa,
    load_participant_agg,
    load_rating_level,
)


def load_video_agg(csv_path: Path) -> pd.DataFrame:
    """One row per video: mean human & AI MSV across all participants."""
    df = load_rating_level(csv_path)
    out = (
        df.dropna(subset=["human_msv", "ai_msv"])
        .groupby("video_id", as_index=False)[["human_msv", "ai_msv"]]
        .mean()
    )
    return out


REG_COLOR = "#ff8087"  # heatmap regression line (red)
SHUFFLED_COLOR = "#375f7d"
PROFILE_DASH = "-"  # solid line for profile
SHUFFLED_DASH = "--"
PROFILE_LW_SCALE = 1.35  # profile line thicker than default
SHUFFLED_LW_SCALE = 0.65
GEMINI_DARK = "#4fb5ef"
QWEN_DARK = "#a875d6"

MODEL_SPECS = {
    "qwen": {
        "profile": Path("data/results/qwen_zero_shot.csv"),
        "shuffled": Path("data/results/qwen_shuffled.csv"),
        "color": QWEN_COLOR,
        "line_color": QWEN_DARK,
        "label": "Qwen",
        "out": Path("figures/pmsv_dist_real_vs_shuffle_qwen.pdf"),
        "out_corr": Path("figures/pmsv_corr_agmt_video_qwen.pdf"),
        "out_dist": Path("figures/pmsv_dist_real_vs_shuffle_qwen.pdf"),
    },
    "gemini": {
        "profile": Path("data/results/gemini_zero_shot.csv"),
        "shuffled": Path("data/results/gemini_shuffled.csv"),
        "color": GEMINI_COLOR,
        "line_color": GEMINI_DARK,
        "label": "Gemini",
        "out": Path("figures/pmsv_corr_agmt_video_gemini.pdf"),
        "out_corr": Path("figures/pmsv_corr_agmt_video_gemini.pdf"),
        "out_dist": Path("figures/pmsv_dist_real_vs_shuffle_gemini.pdf"),
    },
}

# ── layout constants (manual axes placement for exact alignment) ────────────
FIGSIZE = (15, 7.0)
PANEL_W = 0.34
PANEL_H = 0.76
LEFT_X = 0.07
CBAR_GAP = 0.008
CBAR_W = 0.012
# Gap between colorbar (and its label) and the right panel's left edge.
# Increase this value to push the density plot further right.
INTER_GAP = 0.20
BOTTOM_Y = 0.14

# Standalone single-panel figure sizes (match combined-panel proportions).
SINGLE_CORR_FIGSIZE = (6.5, 6.0)
SINGLE_DIST_FIGSIZE = (7.5, 7.0)

# Font sizes


def _mean_sd(arr: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _load_model_data(spec: dict) -> dict:
    """Heatmap stats are computed at video level; distribution uses participant level."""
    # Video-level for the heatmap / kappa / correlation.
    df_vid = load_video_agg(spec["profile"])

    # Participant-level for the right-hand distribution plot.
    df_prof = load_participant_agg(spec["profile"]).dropna(
        subset=["human_msv", "ai_msv"]
    )
    df_shuf = load_participant_agg(spec["shuffled"]).dropna(
        subset=["human_msv", "ai_msv"]
    )

    human = df_vid["human_msv"].to_numpy(dtype=float)
    ai = df_vid["ai_msv"].to_numpy(dtype=float)
    h_rc = _round_clip(human)
    a_rc = _round_clip(ai)
    kappa = _weighted_kappa(h_rc, a_rc)

    mat = np.zeros((7, 7), dtype=int)
    for h, a in zip(h_rc, a_rc):
        mat[a - 1, h - 1] += 1
    mat_pct = mat / mat.sum() * 100

    slope, intercept, r, p, _ = stats.linregress(human, ai)
    x_line = np.linspace(1, 7, 300)
    y_fit, ci_lo, ci_hi = _ci_band(human, ai, x_line)

    prof_vals = df_prof["ai_msv"].to_numpy(dtype=float)
    shuf_vals = df_shuf["ai_msv"].to_numpy(dtype=float)

    return dict(
        human=human,
        ai=ai,
        kappa=kappa,
        mat_pct=mat_pct,
        r=r,
        p=p,
        x_line=x_line,
        y_fit=y_fit,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        prof_vals=prof_vals,
        shuf_vals=shuf_vals,
    )


def _draw_corr_panel(fig, ax_k, cbar_ax, spec: dict, data: dict) -> None:
    """Draw the kappa/corr heatmap + CI band + title annotation on (ax_k, cbar_ax)."""
    ratings = np.arange(1, 8)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", ["#f7f7f7", spec["color"]], N=256
    )
    im = ax_k.imshow(
        data["mat_pct"],
        cmap=cmap,
        aspect="auto",
        vmin=0,
        origin="lower",
        extent=[-0.5, 6.5, -0.5, 6.5],
    )

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=FS["tick"])
    cbar.set_label("% of videos", fontsize=FS["annotation"])

    ax_k.fill_between(
        data["x_line"] - 1,
        data["ci_lo"] - 1,
        data["ci_hi"] - 1,
        color=REG_COLOR,
        alpha=0.25,
        linewidth=0,
        zorder=4,
    )
    ax_k.plot(
        data["x_line"] - 1,
        data["y_fit"] - 1,
        color=REG_COLOR,
        linewidth=LW["regression"] * 0.7,
        zorder=5,
    )

    annot = f"ρ = {data['r']:.2f}"

    ax_k.set_xlim(-0.5, 6.5)
    ax_k.set_ylim(-0.5, 6.5)
    ax_k.set_xticks(range(7))
    ax_k.set_yticks(range(7))
    ax_k.set_xticklabels(ratings)
    ax_k.set_yticklabels(ratings)
    ax_k.tick_params(axis="both", labelsize=FS["tick"])
    ax_k.set_xlabel("Human", fontsize=FS["axis_label"])
    ax_k.set_ylabel(spec["label"], fontsize=FS["axis_label"])
    ax_k.set_title(annot, fontsize=FS["annotation"], fontweight="bold", pad=8)


def _draw_dist_panel(ax_d, spec: dict, data: dict) -> None:
    """Draw the real-vs-shuffled KDE plot on ax_d (with staggered legends)."""
    prof_vals = data["prof_vals"]
    shuf_vals = data["shuf_vals"]

    x_grid = np.linspace(1, 7, 300)
    lw = LW["regression"] * 0.8
    kde_prof = gaussian_kde(prof_vals, bw_method="scott")
    kde_shuf = gaussian_kde(shuf_vals, bw_method="scott")

    ax_d.plot(
        x_grid,
        kde_prof(x_grid),
        color=spec["line_color"],
        linestyle=PROFILE_DASH,
        linewidth=lw * PROFILE_LW_SCALE,
        zorder=3,
    )
    ax_d.plot(
        x_grid,
        kde_shuf(x_grid),
        color=SHUFFLED_COLOR,
        linestyle=SHUFFLED_DASH,
        linewidth=lw * SHUFFLED_LW_SCALE,
        zorder=4,
    )

    ax_d.set_xlim(0.5, 7.5)
    ax_d.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax_d.set_ylim(0, 1.0)
    ax_d.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.tick_params(labelsize=FS["tick"])
    ax_d.set_ylabel("Density", fontsize=FS["axis_label"])
    ax_d.set_xlabel("PMSV", fontsize=FS["axis_label"])
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.set_title(spec["label"], fontsize=FS["title"], fontweight="bold", pad=8)

    p_mu, p_sd = _mean_sd(prof_vals)
    s_mu, s_sd = _mean_sd(shuf_vals)
    prof_handle = mlines.Line2D(
        [],
        [],
        color=spec["line_color"],
        linestyle=PROFILE_DASH,
        linewidth=lw * PROFILE_LW_SCALE,
    )
    shuf_handle = mlines.Line2D(
        [],
        [],
        color=SHUFFLED_COLOR,
        linestyle=SHUFFLED_DASH,
        linewidth=lw * SHUFFLED_LW_SCALE,
    )
    leg_kwargs = dict(
        borderaxespad=0.0,
        fontsize=FS["annotation"],
        framealpha=0.0,
        edgecolor="none",
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.0,
        labelspacing=0.3,
    )
    legend_left_x = 0.1
    leg_prof = ax_d.legend(
        [prof_handle],
        [f"Real ({p_mu:.2f}±{p_sd:.2f})"],
        loc="lower left",
        bbox_to_anchor=(legend_left_x, 0.88),
        **leg_kwargs,
    )
    ax_d.add_artist(leg_prof)
    leg_shuf = ax_d.legend(
        [shuf_handle],
        [f"Shuffled ({s_mu:.2f}±{s_sd:.2f})"],
        loc="upper left",
        bbox_to_anchor=(legend_left_x, 0.875),
        **leg_kwargs,
    )
    # Stash on the figure so _save can include them in tight-bbox calculations.
    extras = list(getattr(ax_d.figure, "_extra_legend_artists", []))
    extras.extend([leg_prof, leg_shuf])
    ax_d.figure._extra_legend_artists = extras


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    extras = getattr(fig, "_extra_legend_artists", None) or None
    fig.savefig(out, bbox_inches="tight", dpi=150, bbox_extra_artists=extras)
    plt.close(fig)


def _draw_combined(model: str, data: dict) -> Path:
    spec = MODEL_SPECS[model]
    fig = plt.figure(figsize=FIGSIZE)

    cbar_x = LEFT_X + PANEL_W + CBAR_GAP
    right_x = cbar_x + CBAR_W + INTER_GAP

    ax_k = fig.add_axes([LEFT_X, BOTTOM_Y, PANEL_W, PANEL_H])
    cbar_ax = fig.add_axes([cbar_x, BOTTOM_Y, CBAR_W, PANEL_H])
    ax_d = fig.add_axes([right_x, BOTTOM_Y, PANEL_W, PANEL_H])

    _draw_corr_panel(fig, ax_k, cbar_ax, spec, data)
    _draw_dist_panel(ax_d, spec, data)

    out = spec["out"]
    _save(fig, out)
    return out


def _draw_corr_only(model: str, data: dict) -> Path:
    spec = MODEL_SPECS[model]
    fig = plt.figure(figsize=SINGLE_CORR_FIGSIZE)

    panel_w = 0.72
    cbar_gap = 0.018
    cbar_w = 0.025
    left_x = 0.16
    bottom_y = BOTTOM_Y
    panel_h = PANEL_H

    ax_k = fig.add_axes([left_x, bottom_y, panel_w, panel_h])
    cbar_ax = fig.add_axes([left_x + panel_w + cbar_gap, bottom_y, cbar_w, panel_h])
    _draw_corr_panel(fig, ax_k, cbar_ax, spec, data)

    out = spec["out_corr"]
    _save(fig, out)
    return out


def _draw_dist_only(model: str, data: dict) -> Path:
    spec = MODEL_SPECS[model]
    fig = plt.figure(figsize=SINGLE_DIST_FIGSIZE)

    panel_w = 0.78
    left_x = 0.16
    bottom_y = BOTTOM_Y
    panel_h = PANEL_H

    ax_d = fig.add_axes([left_x, bottom_y, panel_w, panel_h])
    _draw_dist_panel(ax_d, spec, data)

    out = spec["out_dist"]
    _save(fig, out)
    return out


def _draw_corr_sidebyside() -> Path:
    """Draw Gemini and Qwen video-level heatmaps side by side → figures/pmsv_corr_agmt_video.pdf"""
    from matplotlib.gridspec import GridSpec

    g_data = _load_model_data(MODEL_SPECS["gemini"])
    q_data = _load_model_data(MODEL_SPECS["qwen"])

    fig = plt.figure(figsize=(15.8, 7.0))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 0.05, 0.18, 1, 0.05], wspace=0.08)
    ax_g = fig.add_subplot(gs[0, 0])
    cbar_g = fig.add_subplot(gs[0, 1])
    _gap = fig.add_subplot(gs[0, 2])
    _gap.set_visible(False)
    ax_q = fig.add_subplot(gs[0, 3])
    cbar_q = fig.add_subplot(gs[0, 4])

    _draw_corr_panel(fig, ax_g, cbar_g, MODEL_SPECS["gemini"], g_data)
    cbar_g.set_ylabel("")
    _draw_corr_panel(fig, ax_q, cbar_q, MODEL_SPECS["qwen"], q_data)
    ax_q.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14)
    out = Path("figures/pmsv_corr_agmt_video.pdf")
    _save(fig, out)
    print(
        f"[corr_sidebyside] gemini r={g_data['r']:.3f} κ={g_data['kappa']:.3f} | "
        f"qwen r={q_data['r']:.3f} κ={q_data['kappa']:.3f}\n"
        f"    -> {out}"
    )
    return out


def _ensemble_participant_msv(paths: list[Path]) -> pd.DataFrame:
    """Average predicted_msv across multiple runs per (participant_id, video_id),
    then aggregate to participant level (mean across videos)."""
    dfs = [
        pd.read_csv(p)[["participant_id", "video_id", "predicted_msv"]] for p in paths
    ]
    merged = dfs[0].rename(columns={"predicted_msv": "pred_0"})
    for i, df in enumerate(dfs[1:], 1):
        merged = merged.merge(
            df.rename(columns={"predicted_msv": f"pred_{i}"}),
            on=["participant_id", "video_id"],
            how="inner",
        )
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged["predicted_msv"] = merged[pred_cols].mean(axis=1)
    return merged.groupby("participant_id")["predicted_msv"].mean().reset_index()


def _draw_dist_gemini_ensemble() -> Path:
    """Fig 7: Gemini zero-shot vs 3-run shuffled ensemble KDE distribution."""
    spec = MODEL_SPECS["gemini"]

    # zero-shot participant-level ai_msv
    df_prof = load_participant_agg(spec["profile"]).dropna(subset=["ai_msv"])
    prof_vals = df_prof["ai_msv"].to_numpy(dtype=float)

    # ensemble shuffled participant-level predicted_msv
    shuf_paths = [
        Path("data/results/gemini_shuffled.csv"),
        Path("data/results/gemini_shuffled_run_2.csv"),
        Path("data/results/gemini_shuffled_run_3.csv"),
    ]
    df_shuf = _ensemble_participant_msv(shuf_paths).dropna(subset=["predicted_msv"])
    shuf_vals = df_shuf["predicted_msv"].to_numpy(dtype=float)

    data = dict(prof_vals=prof_vals, shuf_vals=shuf_vals)

    fig = plt.figure(figsize=SINGLE_DIST_FIGSIZE)
    panel_w = 0.78
    left_x = 0.16
    ax_d = fig.add_axes([left_x, BOTTOM_Y, panel_w, PANEL_H])
    _draw_dist_panel(ax_d, spec, data)

    out = Path("figures/pmsv_dist_real_vs_shuffle_gemini.pdf")
    _save(fig, out)
    return out


def _draw_dist_qwen_ensemble() -> Path:
    """Fig 5: Qwen zero-shot vs 3-run shuffled ensemble KDE distribution."""
    spec = MODEL_SPECS["qwen"]

    df_prof = load_participant_agg(spec["profile"]).dropna(subset=["ai_msv"])
    prof_vals = df_prof["ai_msv"].to_numpy(dtype=float)

    shuf_paths = [
        Path("data/results/qwen_shuffled_run_1.csv"),
        Path("data/results/qwen_shuffled_run_2.csv"),
        Path("data/results/qwen_shuffled_run_3.csv"),
    ]
    df_shuf = _ensemble_participant_msv(shuf_paths).dropna(subset=["predicted_msv"])
    shuf_vals = df_shuf["predicted_msv"].to_numpy(dtype=float)

    data = dict(prof_vals=prof_vals, shuf_vals=shuf_vals)

    fig = plt.figure(figsize=SINGLE_DIST_FIGSIZE)
    panel_w = 0.78
    left_x  = 0.16
    ax_d = fig.add_axes([left_x, BOTTOM_Y, panel_w, PANEL_H])
    _draw_dist_panel(ax_d, spec, data)

    out = Path("figures/pmsv_dist_real_vs_shuffle_qwen.pdf")
    _save(fig, out)
    return out


def _compute_residuals(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (eps_human, eps_model): per-video-mean-subtracted residuals."""
    df = pd.read_csv(csv_path).dropna(subset=["human_perceived_msv", "predicted_msv"])
    vid = (
        df.groupby("video_id")
        .agg(
            h_bar=("human_perceived_msv", "mean"),
            a_bar=("predicted_msv", "mean"),
        )
        .reset_index()
    )
    m = df.merge(vid, on="video_id")
    return (m["human_perceived_msv"] - m["h_bar"]).to_numpy(), (
        m["predicted_msv"] - m["a_bar"]
    ).to_numpy()


def _load_residual_data(spec: dict) -> dict:
    eps_h, eps_a = _compute_residuals(spec["profile"])
    r, _ = stats.pearsonr(eps_h, eps_a)

    rng = 4.0
    edges = np.linspace(-rng, rng, 14)
    H, _, _ = np.histogram2d(eps_h, eps_a, bins=[edges, edges])
    H_pct = H.T / H.sum() * 100

    x_line = np.linspace(eps_h.min(), eps_h.max(), 300)
    y_fit, ci_lo, ci_hi = _ci_band(eps_h, eps_a, x_line)

    return dict(
        r=r,
        H_pct=H_pct,
        edges=edges,
        x_line=x_line,
        y_fit=y_fit,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )


def _draw_residual_panel(fig, ax_k, cbar_ax, spec: dict, data: dict) -> None:
    """Draw the residual 2D-density heatmap, matching _draw_corr_panel style."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", ["#f7f7f7", spec["color"]], N=256
    )
    edges = data["edges"]
    im = ax_k.imshow(
        data["H_pct"],
        extent=[edges[0], edges[-1], edges[0], edges[-1]],
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=0,
    )

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 5, 10])
    cbar.ax.tick_params(labelsize=FS["tick"])
    cbar.set_label("% of ratings", fontsize=FS["annotation"])

    ax_k.fill_between(
        data["x_line"],
        data["ci_lo"],
        data["ci_hi"],
        color=REG_COLOR,
        alpha=0.25,
        linewidth=0,
        zorder=4,
    )
    ax_k.plot(
        data["x_line"],
        data["y_fit"],
        color=REG_COLOR,
        linewidth=LW["regression"] * 0.7,
        zorder=5,
    )

    ax_k.set_xticks([-4, -2, 0, 2, 4])
    ax_k.set_yticks([-4, -2, 0, 2, 4])
    ax_k.tick_params(axis="both", labelsize=FS["tick"])
    ax_k.set_xlabel("Human", fontsize=FS["axis_label"])
    ax_k.set_ylabel(spec["label"], fontsize=FS["axis_label"])
    ax_k.set_title(
        f"ρ = {data['r']:.2f}", fontsize=FS["annotation"], fontweight="bold", pad=8
    )
    ax_k.set_xlim(edges[0], edges[-1])
    ax_k.set_ylim(edges[0], edges[-1])


def _draw_residual_only(model: str) -> Path:
    spec = MODEL_SPECS[model]
    data = _load_residual_data(spec)

    fig = plt.figure(figsize=SINGLE_CORR_FIGSIZE)
    panel_w = 0.72
    cbar_gap = 0.018
    cbar_w = 0.025
    left_x = 0.16

    ax_k = fig.add_axes([left_x, BOTTOM_Y, panel_w, PANEL_H])
    cbar_ax = fig.add_axes([left_x + panel_w + cbar_gap, BOTTOM_Y, cbar_w, PANEL_H])
    _draw_residual_panel(fig, ax_k, cbar_ax, spec, data)

    out = Path(f"figures/pmsv_corr_agmt_residual_{model}.pdf")
    _save(fig, out)
    return out


def _draw(model: str) -> None:
    spec = MODEL_SPECS[model]
    data = _load_model_data(spec)
    combined = _draw_combined(model, data)
    corr_only = _draw_corr_only(model, data)
    residual = _draw_residual_only(model)
    print(
        f"[kappa_corr_dist] {model}: r={data['r']:.3f}  κ={data['kappa']:.3f}\n"
        f"    combined -> {combined}\n"
        f"    heatmap  -> {corr_only}\n"
        f"    residual -> {residual}"
    )
    if model != "gemini":
        dist_only = _draw_dist_only(model, data)
        print(f"    density  -> {dist_only}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Participant-level kappa/corr heatmap + profile-vs-shuffled distribution."
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "gemini", "both"],
        default="qwen",
        help="Which model to plot (default: qwen).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "both":
        _draw("qwen")
        _draw("gemini")
    else:
        _draw(args.model)


if __name__ == "__main__":
    main()
