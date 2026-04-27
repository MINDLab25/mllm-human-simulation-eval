"""figures.py — shared helpers for generating paper figures 2–4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Helvetica",
    "Helvetica Neue",
    "Arial",
    "DejaVu Sans",
]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Helvetica"
matplotlib.rcParams["mathtext.it"] = "Helvetica:italic"
matplotlib.rcParams["mathtext.bf"] = "Helvetica:bold"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FS = {
    "annotation": 35,
    "tick": 32,
    "axis_label": 37,
    "y_label": 35,
    "title": 38,
    "grid_title": 35,
}
LW = {
    "regression": 8,
    "identity": 3,
}

GEMINI_COLOR = "#8fd8fa"
QWEN_COLOR = "#cb98e7"
MALE_COLOR = "#399ca7"
FEMALE_COLOR = "#fe8a5b"

ITEM_IDS = [
    "emotional",
    "arousing",
    "involving",
    "exciting",
    "powerful_impact",
    "stimulating",
    "strong_visual",
    "strong_soundeffect",
    "dramatic",
    "graphic",
    "creative",
    "goosebump",
    "intense",
    "strong_soundtrack",
    "novel",
    "unique",
    "unusual",
]

AGE_BINS = [0, 38, 200_000]
AGE_LABELS = ["Under 39", "39 or older"]

EDUCATION_MAP = {
    1: "Below Bachelor's",
    2: "Below Bachelor's",
    3: "Below Bachelor's",
    4: "Below Bachelor's",
    5: "Bachelor's or higher",
    6: "Bachelor's or higher",
}
INCOME_MAP = {
    1: "Less than $50,000",
    2: "Less than $50,000",
    3: "$50,000 – $99,999",
    4: "$50,000 – $99,999",
    5: "$100,000 or more",
    6: "$100,000 or more",
}


# ── Data loading ──────────────────────────────────────────────────────────────


def load_participant_agg(csv_path: Path) -> pd.DataFrame:
    """One row per participant — mean across all their videos."""
    df = pd.read_csv(csv_path)
    agg: dict[str, str] = {}
    for item in ITEM_IDS:
        if f"human_{item}" in df.columns:
            agg[f"human_{item}"] = "mean"
        if f"ai_{item}" in df.columns:
            agg[f"ai_{item}"] = "mean"
    if "human_perceived_msv" in df.columns:
        agg["human_perceived_msv"] = "mean"
    if "predicted_msv" in df.columns:
        agg["predicted_msv"] = "mean"
    demo = {
        c: "first"
        for c in ["age", "gender", "race", "sen_seek", "education", "income"]
        if c in df.columns
    }
    out = df.groupby("participant_id").agg({**agg, **demo}).reset_index()
    out = out.rename(
        columns={"human_perceived_msv": "human_msv", "predicted_msv": "ai_msv"}
    )
    if "age" in out.columns:
        out["age_group"] = pd.cut(
            out["age"].clip(upper=120), bins=AGE_BINS, labels=AGE_LABELS
        )
    if "education" in out.columns:
        out["edu_label"] = out["education"].apply(
            lambda x: EDUCATION_MAP.get(int(x), str(x)) if pd.notna(x) else None
        )
    if "income" in out.columns:
        out["income_label"] = out["income"].apply(
            lambda x: INCOME_MAP.get(int(x), str(x)) if pd.notna(x) else None
        )
    if "race" in out.columns:
        out["race"] = out["race"].replace("Black or African American", "Black")
    if "sen_seek" in out.columns:
        out["ss_group"] = pd.cut(
            out["sen_seek"],
            bins=[0.99, 2.5, 5.01],
            labels=["Low SS", "High SS"],
        )
    return out


def load_rating_level(csv_path: Path) -> pd.DataFrame:
    """Load raw (participant × video) ratings — no aggregation."""
    df = pd.read_csv(csv_path)
    keep = ["participant_id", "video_id", "human_perceived_msv", "predicted_msv"] + [
        c
        for c in df.columns
        if (c.startswith("human_") or c.startswith("ai_"))
        and c not in ("human_perceived_msv",)
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out = out.rename(
        columns={"human_perceived_msv": "human_msv", "predicted_msv": "ai_msv"}
    )
    return out.dropna(subset=["human_msv", "ai_msv"])


# ── Shared stats helpers ──────────────────────────────────────────────────────


def _ci_band(human, ai, x_fit):
    """95% CI band for a regression line."""
    slope, intercept, *_ = stats.linregress(human, ai)
    y_fit = slope * x_fit + intercept
    n = len(human)
    x_mean = np.mean(human)
    ss_xx = np.sum((human - x_mean) ** 2)
    ss_res = np.sum((ai - (slope * human + intercept)) ** 2)
    mse = ss_res / (n - 2)
    se_fit = np.sqrt(mse * (1 / n + (x_fit - x_mean) ** 2 / ss_xx))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    return y_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit


def _round_clip(vals: np.ndarray) -> np.ndarray:
    return np.clip(np.round(vals).astype(int), 1, 7)


def _weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    return cohen_kappa_score(a, b, weights="quadratic")


def _icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(A,1): two-way random, absolute agreement, single measure (k=2)."""
    n = len(x)
    m = (x + y) / 2
    mu = m.mean()
    SSB = 2 * np.sum((m - mu) ** 2)
    SSW = np.sum((x - m) ** 2 + (y - m) ** 2)
    SSC = n * ((x.mean() - mu) ** 2 + (y.mean() - mu) ** 2)
    SSE = SSW - SSC
    MSB = SSB / (n - 1)
    MSW = SSW / n
    MSC = SSC
    MSE = SSE / (n - 1)
    denom = MSB + MSW + 2 * (MSC - MSE) / n
    return float((MSB - MSW) / denom) if denom != 0 else float("nan")


# ── Figure 2: participant-level ρ / κ grid ────────────────────────────────────


def _kappa_corr_grid_participant(conditions_data: list) -> plt.Figure:
    """2 × N grid: rows = [Gemini, Qwen], cols = conditions.
    `conditions_data`: list of (label, df_g, df_q, kappa_hg, kappa_hq).
    """
    import matplotlib.colors as mcolors

    n_cols = len(conditions_data)
    ratings = np.arange(1, 8)
    REG_COLOR = "#ff8087"

    def _mat_pct(human_rc, ai_rc):
        mat = np.zeros((7, 7), dtype=int)
        for h, a in zip(human_rc, ai_rc):
            mat[a - 1, h - 1] += 1
        return mat / mat.sum() * 100

    panels = []
    for label, df_g, df_q, kappa_hg, kappa_hq in conditions_data:
        gem = df_g.dropna(subset=["human_msv", "ai_msv"])
        qwn = df_q.dropna(subset=["human_msv", "ai_msv"])
        h_g = gem["human_msv"].to_numpy(dtype=float)
        a_g = gem["ai_msv"].to_numpy(dtype=float)
        h_q = qwn["human_msv"].to_numpy(dtype=float)
        a_q = qwn["ai_msv"].to_numpy(dtype=float)
        panels.append(
            {
                "label": label,
                "gemini": (
                    h_g,
                    a_g,
                    _mat_pct(_round_clip(h_g), _round_clip(a_g)),
                    GEMINI_COLOR,
                    "Gemini",
                    kappa_hg,
                ),
                "qwen": (
                    h_q,
                    a_q,
                    _mat_pct(_round_clip(h_q), _round_clip(a_q)),
                    QWEN_COLOR,
                    "Qwen",
                    kappa_hq,
                ),
            }
        )

    shared_vmax = max(max(p["gemini"][2].max(), p["qwen"][2].max()) for p in panels)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(5.8 * n_cols + 0.8, 11))
    gs = GridSpec(
        2,
        n_cols + 1,
        figure=fig,
        width_ratios=[1] * n_cols + [0.05],
        hspace=0.45,
        wspace=0.42,
    )
    axes = np.empty((2, n_cols), dtype=object)
    for r in range(2):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cbar_axs = [fig.add_subplot(gs[r, n_cols]) for r in range(2)]

    last_im = [None, None]
    for col, panel in enumerate(panels):
        for row, model_key in enumerate(["gemini", "qwen"]):
            ax = axes[row, col]
            human, ai, mat, color, model_label, kappa = panel[model_key]

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom", ["#f7f7f7", color], N=256
            )
            im = ax.imshow(
                mat,
                cmap=cmap,
                aspect="equal",
                vmin=0,
                vmax=shared_vmax,
                origin="lower",
                extent=[-0.5, 6.5, -0.5, 6.5],
            )
            last_im[row] = im

            annot = f"ICC = {kappa:.2f}"
            if len(human) >= 2:
                slope, intercept, r, p, _ = stats.linregress(human, ai)
                x_line = np.linspace(1, 7, 300)
                y_fit, ci_lo, ci_hi = _ci_band(human, ai, x_line)
                ax.fill_between(
                    x_line - 1,
                    ci_lo - 1,
                    ci_hi - 1,
                    color=REG_COLOR,
                    alpha=0.25,
                    linewidth=0,
                    zorder=4,
                )
                ax.plot(
                    x_line - 1,
                    y_fit - 1,
                    color=REG_COLOR,
                    linewidth=LW["regression"] * 0.55,
                    zorder=5,
                )
                annot = f"ρ = {r:.2f}, ICC = {kappa:.2f}"

            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels(ratings)
            ax.set_yticklabels(ratings)
            ax.tick_params(axis="both", labelsize=FS["tick"])
            ax.set_xlabel(
                "Human" if row == 1 else "", fontsize=FS["axis_label"], labelpad=10
            )
            if col == 0:
                ax.set_ylabel(model_label, fontsize=FS["axis_label"], labelpad=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(annot, fontsize=FS["annotation"], fontweight="bold", pad=10)

    for row, cbar_ax in enumerate(cbar_axs):
        cbar = fig.colorbar(last_im[row], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=FS["tick"])
        cbar.set_label("% of participants", fontsize=FS["annotation"])

    fig.subplots_adjust(top=0.87)

    for col, panel in enumerate(panels):
        bbox = axes[0, col].get_position()
        x_center = (bbox.x0 + bbox.x1) / 2.0
        fig.text(
            x_center, 0.965, panel["label"], ha="center", va="top", fontsize=FS["title"]
        )

    return fig


def gen_fig2(args: argparse.Namespace, out: Path) -> None:
    """Figure 2 → figures/pmsv_corr_agmt.pdf"""
    out.parent.mkdir(parents=True, exist_ok=True)

    conditions = []
    for label, gcsv, qcsv in [
        ("Zero-shot", args.csv, args.qwen_csv),
        ("Few-shot", args.fews_csv, args.qwen_fews_csv),
        ("CoT", args.cot_csv, args.qwen_cot_csv),
    ]:
        gp, qp = Path(gcsv), Path(qcsv)
        if gp.exists() and qp.exists():
            conditions.append((label, gp, qp))

    if not conditions:
        print("[fig2] No valid conditions found.", file=sys.stderr)
        return

    grid_data = []
    for label, gcsv, qcsv in conditions:
        df_g = load_participant_agg(gcsv).dropna(subset=["human_msv", "ai_msv"])
        df_q = load_participant_agg(qcsv).dropna(subset=["human_msv", "ai_msv"])
        icc_hg = _icc_a1(df_g["human_msv"].values, df_g["ai_msv"].values)
        icc_hq = _icc_a1(df_q["human_msv"].values, df_q["ai_msv"].values)
        grid_data.append((label, df_g, df_q, icc_hg, icc_hq))

    fig = _kappa_corr_grid_participant(grid_data)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig2] → {out}")


# ── Figure 3: PMSV distribution (participant-level, all conditions) ───────────


def gen_fig3(args: argparse.Namespace, out: Path) -> None:
    """Figure 3 → figures/pmsv_hists.pdf"""
    from scipy.stats import gaussian_kde
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    out.parent.mkdir(parents=True, exist_ok=True)

    HUMAN_COLOR_H = "#bec5cc"
    GEMINI_DARK = "#4fb5ef"
    QWEN_DARK = "#a875d6"
    bins = np.arange(0.75, 7.5, 0.5)
    x_grid = np.linspace(1, 7, 300)
    lw = LW["regression"] * 0.8

    STYLES = [
        {"linestyle": "-", "linewidth": lw, "label": "Zero-shot"},
        {"linestyle": (0, (6, 2)), "linewidth": lw, "label": "Few-shot"},
        {"linestyle": (0, (1, 1)), "linewidth": lw * 0.8, "label": "CoT"},
    ]

    def _load(csv_path):
        df = load_participant_agg(Path(csv_path)).dropna(subset=["human_msv", "ai_msv"])
        return df["human_msv"].values, df["ai_msv"].values

    def _ms(arr):
        arr = np.asarray(arr, dtype=float)
        return np.mean(arr[~np.isnan(arr)]), np.std(arr[~np.isnan(arr)], ddof=1)

    human_g, g_zero = _load(args.csv)
    _, g_fews = _load(args.fews_csv)
    _, g_cot = _load(args.cot_csv)
    human_q, q_zero = _load(args.qwen_csv)
    _, q_fews = _load(args.qwen_fews_csv)
    _, q_cot = _load(args.qwen_cot_csv)

    def _draw_panel(ax, human, ai_list, color, xlabel, title):
        ax.hist(
            human,
            bins=bins,
            density=True,
            color=HUMAN_COLOR_H,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.8,
            rwidth=0.96,
        )
        for ai_vals, style in zip(ai_list, STYLES):
            kde = gaussian_kde(ai_vals, bw_method="scott")
            ax.plot(x_grid, kde(x_grid), color=color, zorder=3, **style)
        ax.set_xlim(0.5, 7.5)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_ylabel("Density", fontsize=FS["axis_label"])
        ax.set_xlabel(xlabel, fontsize=FS["axis_label"])
        ax.set_title(title, fontsize=FS["title"], fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        h_mu, h_sd = _ms(human)
        zs_mu, zs_sd = _ms(ai_list[0])
        fs_mu, fs_sd = _ms(ai_list[1])
        ct_mu, ct_sd = _ms(ai_list[2])
        human_handle = mpatches.Patch(color=HUMAN_COLOR_H, alpha=0.65)
        line_handles = [
            mlines.Line2D(
                [], [], color=color, linestyle=s["linestyle"], linewidth=s["linewidth"]
            )
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
    _draw_panel(
        axes[0],
        human_g,
        [g_zero, g_fews, g_cot],
        GEMINI_DARK,
        "PMSV",
        "Human vs Gemini",
    )
    _draw_panel(
        axes[1], human_q, [q_zero, q_fews, q_cot], QWEN_DARK, "PMSV", "Human vs Qwen"
    )
    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(top=0.95, bottom=0.08, right=0.97, left=0.1)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig3] → {out}")


# ── Figure 4: PMSV by demographic group (boxplots) ───────────────────────────


def gen_fig4(args: argparse.Namespace, out_age: Path, out_gender: Path) -> None:
    """Figure 4 → figures/pmsv_age.pdf + figures/pmsv_gender.pdf"""
    from scipy.stats import ttest_ind

    for out in (out_age, out_gender):
        out.parent.mkdir(parents=True, exist_ok=True)

    def _fmt_sig(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _sig_bracket(ax, y_top, sig, fs, lw_br):
        tick_h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
        ax.plot([1, 2], [y_top, y_top], color="black", linewidth=lw_br, clip_on=False)
        ax.plot(
            [1, 1],
            [y_top - tick_h, y_top],
            color="black",
            linewidth=lw_br,
            clip_on=False,
        )
        ax.plot(
            [2, 2],
            [y_top - tick_h, y_top],
            color="black",
            linewidth=lw_br,
            clip_on=False,
        )
        ax.text(
            1.5,
            y_top + tick_h * 0.4,
            sig,
            ha="center",
            va="bottom",
            fontsize=fs,
            color="black",
        )

    def _make_fig(
        gemini_csv,
        qwen_csv,
        group_col,
        group1,
        group2,
        label1,
        label2,
        color1,
        color2,
        preprocess,
        bracket_height_factor,
        out_path,
    ):
        lw = LW["regression"] * 0.8
        lw_br = lw * 0.35

        def _load(csv_path):
            df = pd.read_csv(csv_path).dropna(
                subset=["human_perceived_msv", "predicted_msv"]
            )
            if preprocess is not None:
                df = preprocess(df)
            return df[df[group_col].isin([group1, group2])].copy()

        df_g = _load(gemini_csv)
        df_q = _load(qwen_csv)
        panels = [
            ("Human", df_g, "human_perceived_msv"),
            ("Gemini", df_g, "predicted_msv"),
            ("Qwen", df_q, "predicted_msv"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(11, 5))
        for i, (ax, (col_title, df, ycol)) in enumerate(zip(axes, panels)):
            v1 = df[df[group_col] == group1][ycol].values
            v2 = df[df[group_col] == group2][ycol].values
            bp = ax.boxplot(
                [v1, v2],
                positions=[1, 2],
                widths=0.55,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=3.0),
                whiskerprops=dict(linewidth=2.0),
                capprops=dict(linewidth=2.0),
                flierprops=dict(marker="o", markersize=4, alpha=0.3, linewidth=0),
                boxprops=dict(linewidth=2.0),
            )
            for patch, c in zip(bp["boxes"], [color1, color2]):
                patch.set_facecolor(c)
                patch.set_alpha(0.8)
            for flier, c in zip(bp["fliers"], [color1, color2]):
                flier.set(markerfacecolor=c, markeredgecolor="none")
            _, p = ttest_ind(v1, v2, equal_var=False)
            y_br = max(v1.max(), v2.max()) * bracket_height_factor
            ax.set_ylim(0.9, max(y_br * 1.15, 7.5))
            ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
            _sig_bracket(ax, y_br, _fmt_sig(p), 26, lw_br)
            ax.set_title(col_title, fontsize=28, fontweight="bold", pad=10)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([label1, label2], fontsize=27)
            ax.tick_params(axis="y", labelsize=23)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i == 0:
                ax.set_ylabel("PMSV", fontsize=27)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

        fig.tight_layout(w_pad=5.0)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[fig4] → {out_path}")

    def _prep_age(df):
        age = pd.to_numeric(df["age"], errors="coerce").clip(18, 100)
        df = df.copy()
        df["age_group"] = age.apply(
            lambda a: (
                "Younger (≤38)"
                if pd.notna(a) and a <= 38
                else ("Older (≥39)" if pd.notna(a) else None)
            )
        )
        return df

    _make_fig(
        args.csv,
        args.qwen_csv,
        group_col="age_group",
        group1="Younger (≤38)",
        group2="Older (≥39)",
        label1="< 39",
        label2="≥ 39",
        color1=MALE_COLOR,
        color2=FEMALE_COLOR,
        preprocess=_prep_age,
        bracket_height_factor=1.08,
        out_path=out_age,
    )
    _make_fig(
        args.csv,
        args.qwen_csv,
        group_col="gender",
        group1="Male",
        group2="Female",
        label1="Male",
        label2="Female",
        color1=MALE_COLOR,
        color2=FEMALE_COLOR,
        preprocess=None,
        bracket_height_factor=1.08,
        out_path=out_gender,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures 2–4.")
    parser.add_argument(
        "--csv", type=Path, default=Path("data/results/gemini_zero_shot.csv")
    )
    parser.add_argument(
        "--qwen-csv", type=Path, default=Path("data/results/qwen_zero_shot.csv")
    )
    parser.add_argument(
        "--fews-csv", type=Path, default=Path("data/results/gemini_few_shot.csv")
    )
    parser.add_argument(
        "--cot-csv", type=Path, default=Path("data/results/gemini_cot.csv")
    )
    parser.add_argument(
        "--qwen-cot-csv", type=Path, default=Path("data/results/qwen_cot.csv")
    )
    parser.add_argument(
        "--qwen-fews-csv", type=Path, default=Path("data/results/qwen_few_shot.csv")
    )
    return parser.parse_args()
