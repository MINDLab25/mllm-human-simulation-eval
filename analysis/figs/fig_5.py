"""
Paper Figure 5: PMSV distribution — Zero-shot vs Shuffled-profile (participant level).
Boxplots with paired t-test for Gemini and Qwen.

Outputs:
    figures/pmsv_dist_shuffle_gemini.pdf
    figures/pmsv_dist_shuffle_qwen.pdf

Also prints LMM mean-shift and variance tests.

Run from repo root:
    python analysis/figs/fig_5.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Helvetica"
matplotlib.rcParams["mathtext.it"] = "Helvetica:italic"
matplotlib.rcParams["mathtext.bf"] = "Helvetica:bold"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.formula.api import mixedlm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from figures import GEMINI_COLOR, QWEN_COLOR

GEMINI_DARK   = "#4fb5ef"
QWEN_DARK     = "#a875d6"

MODEL_SPECS = {
    "gemini": {"line_color": GEMINI_COLOR, "label": "Gemini"},
    "qwen":   {"line_color": QWEN_COLOR,   "label": "Qwen"},
}


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["predicted_msv"])
    return df[["participant_id", "video_id", "predicted_msv"]]


def load_human_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["human_perceived_msv"])
    return df[["participant_id", "human_perceived_msv"]]


def load_ensemble(paths: list[Path]) -> pd.DataFrame:
    frames = [load_ratings(p) for p in paths if len(load_ratings(p)) > 0]
    if not frames:
        raise ValueError(f"No valid shuffled runs found in {paths}")
    merged = frames[0].rename(columns={"predicted_msv": "pred_0"})
    for i, df in enumerate(frames[1:], 1):
        merged = merged.merge(
            df.rename(columns={"predicted_msv": f"pred_{i}"}),
            on=["participant_id", "video_id"],
            how="inner",
        )
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged["predicted_msv"] = merged[pred_cols].mean(axis=1)
    return merged[["participant_id", "video_id", "predicted_msv"]]


# ── LMM tests ─────────────────────────────────────────────────────────────────

def lmm_mean_test(df_zs: pd.DataFrame, df_shuf: pd.DataFrame) -> dict:
    common = set(df_zs["participant_id"]) & set(df_shuf["participant_id"])
    df_zs  = df_zs[df_zs["participant_id"].isin(common)]
    df_shuf = df_shuf[df_shuf["participant_id"].isin(common)]
    long = pd.concat([
        df_zs[["participant_id", "predicted_msv"]].assign(source=0),
        df_shuf[["participant_id", "predicted_msv"]].assign(source=1),
    ], ignore_index=True).rename(columns={"predicted_msv": "rating"})
    result = mixedlm("rating ~ source", long, groups=long["participant_id"]).fit(
        reml=True, method="powell"
    )
    return {
        "zs_mean":     float(df_zs["predicted_msv"].mean()),
        "shuf_mean":   float(df_shuf["predicted_msv"].mean()),
        "zs_sd":       float(df_zs["predicted_msv"].std()),
        "shuf_sd":     float(df_shuf["predicted_msv"].std()),
        "coef_source": float(result.params["source"]),
        "pval_source": float(result.pvalues["source"]),
    }


def lmm_variance_test(df_zs: pd.DataFrame, df_shuf: pd.DataFrame) -> dict:
    common = set(df_zs["participant_id"]) & set(df_shuf["participant_id"])
    df_zs  = df_zs[df_zs["participant_id"].isin(common)]
    df_shuf = df_shuf[df_shuf["participant_id"].isin(common)]
    zs_mean   = df_zs["predicted_msv"].mean()
    shuf_mean = df_shuf["predicted_msv"].mean()
    long = pd.concat([
        df_zs[["participant_id"]].assign(
            source=0, sq_dev=(df_zs["predicted_msv"].values - zs_mean) ** 2
        ),
        df_shuf[["participant_id"]].assign(
            source=1, sq_dev=(df_shuf["predicted_msv"].values - shuf_mean) ** 2
        ),
    ], ignore_index=True)
    result = mixedlm("sq_dev ~ source", long, groups=long["participant_id"]).fit(
        reml=True, method="powell"
    )
    return {"coef_source": float(result.params["source"]),
            "pval_source": float(result.pvalues["source"])}


def _sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── Figure ─────────────────────────────────────────────────────────────────────

def draw_boxplot(
    model: str,
    human_df: pd.DataFrame,
    df_zs: pd.DataFrame,
    df_shuf: pd.DataFrame,
    out: Path,
) -> None:
    spec   = MODEL_SPECS[model]
    color  = spec["line_color"]
    _shuf_colors = {"gemini": "#7baacb", "qwen": "#5577aa"}
    shuf_c = _shuf_colors.get(model, "#375f7d")

    human_p = human_df.groupby("participant_id")["human_perceived_msv"].mean()
    zs_p    = df_zs.groupby("participant_id")["predicted_msv"].mean()
    shuf_p  = df_shuf.groupby("participant_id")["predicted_msv"].mean()

    common = zs_p.index.intersection(shuf_p.index)
    t_stat, p_val = ttest_rel(zs_p.loc[common].values, shuf_p.loc[common].values)
    sig = _sig(p_val)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    parts = ax.violinplot(
        [human_p.values, zs_p.values, shuf_p.values],
        positions=[1, 1.7, 2.4], widths=0.55,
        showmedians=True, showextrema=True,
    )
    styles = [
        dict(facecolor="#cccccc", alpha=0.85),
        dict(facecolor=color,     alpha=0.80),
        dict(facecolor=color,     alpha=0.35),
    ]
    for body, s in zip(parts["bodies"], styles):
        body.set_facecolor(s["facecolor"])
        body.set_edgecolor("none")
        body.set_alpha(s["alpha"])
        body.set_zorder(3)
    for key in ("cmins", "cmaxes", "cbars"):
        parts[key].set_color("#444444"); parts[key].set_linewidth(2.0); parts[key].set_zorder(4)
    parts["cmedians"].set_color("#444444"); parts["cmedians"].set_linewidth(3.0); parts["cmedians"].set_zorder(4)

    y_br  = max(zs_p.values.max(), shuf_p.values.max()) * 1.08
    y_top = max(y_br * 1.15, 7.5)
    tick_h = (y_top - 0.5) * 0.03
    ax.set_ylim(0.5, y_top)
    ax.plot([1.7, 2.4], [y_br, y_br], color="black", linewidth=2.24, clip_on=False, zorder=5)
    ax.plot([1.7, 1.7], [y_br - tick_h, y_br], color="black", linewidth=2.24, clip_on=False, zorder=5)
    ax.plot([2.4, 2.4], [y_br - tick_h, y_br], color="black", linewidth=2.24, clip_on=False, zorder=5)
    ax.text(2.05, y_br + tick_h * 0.4, sig, ha="center", va="bottom", fontsize=38, color="black", zorder=5)

    ax.set_xticks([1, 1.7, 2.4])
    ax.set_xticklabels(["Human", "Real", "Shuffled"], fontsize=38)
    ax.set_xlim(0.5, 2.9)
    ax.set_ylabel("PMSV", fontsize=38)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.tick_params(axis="y", labelsize=38)
    ax.set_title(spec["label"], fontsize=38, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out)
    print(f"  → {out}  [t-test p={p_val:.4f} {sig}, n={len(common)}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    gemini_zs   = load_ratings(Path("data/results/gemini_zero_shot.csv"))
    gemini_shuf = load_ensemble([
        Path("data/results/gemini_shuffled.csv"),
        Path("data/results/gemini_shuffled_run_2.csv"),
        Path("data/results/gemini_shuffled_run_3.csv"),
    ])
    qwen_zs   = load_ratings(Path("data/results/qwen_zero_shot.csv"))
    qwen_shuf = load_ensemble([
        Path("data/results/qwen_shuffled_run_1.csv"),
        Path("data/results/qwen_shuffled_run_2.csv"),
        Path("data/results/qwen_shuffled_run_3.csv"),
    ])

    print("LMM: Zero-shot vs Shuffled")
    print(f"\n{'MLLM':<8} {'ZS mean':>8} {'Sh mean':>8} {'ZS SD':>7} {'Sh SD':>7} "
          f"{'MeanShift':>10} {'p(mean)':>10} {'VarCoef':>8} {'p(var)':>10}")
    print("-" * 80)
    for label, df_zs, df_shuf in [("Gemini", gemini_zs, gemini_shuf),
                                    ("Qwen",   qwen_zs,   qwen_shuf)]:
        m = lmm_mean_test(df_zs, df_shuf)
        v = lmm_variance_test(df_zs, df_shuf)
        print(f"{label:<8} {m['zs_mean']:>8.3f} {m['shuf_mean']:>8.3f} "
              f"{m['zs_sd']:>7.3f} {m['shuf_sd']:>7.3f} "
              f"{m['coef_source']:>+10.3f} {m['pval_source']:>8.2e}{_sig(m['pval_source']):>2}  "
              f"{v['coef_source']:>+8.3f} {v['pval_source']:>8.2e}{_sig(v['pval_source']):>2}")

    human_df = load_human_df(Path("data/results/gemini_zero_shot.csv"))
    draw_boxplot("gemini", human_df, gemini_zs, gemini_shuf,
                 Path("figures/pmsv_dist_shuffle_gemini.pdf"))
    draw_boxplot("qwen", human_df, qwen_zs, qwen_shuf,
                 Path("figures/pmsv_dist_shuffle_qwen.pdf"))


if __name__ == "__main__":
    main()
