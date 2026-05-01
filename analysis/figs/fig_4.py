"""
Paper Figure 4: PMSV by demographic group (zero-shot).
Age (median split at 39) and gender boxplots for Human / Gemini / Qwen.

Outputs → figures/pmsv_age.pdf
          figures/pmsv_gender.pdf

Run from repo root:
    python analysis/figs/fig_4.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.formula.api import mixedlm

sys.path.insert(0, str(Path(__file__).parent.parent))
from figures import FEMALE_COLOR, LW, MALE_COLOR, load_participant_agg, parse_args


def _fmt_sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _sig_bracket(ax, y_top: float, sig: str, fs: int, lw_br: float) -> None:
    tick_h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([1, 2], [y_top, y_top], color="black", linewidth=lw_br, clip_on=False)
    ax.plot([1, 1], [y_top - tick_h, y_top], color="black", linewidth=lw_br, clip_on=False)
    ax.plot([2, 2], [y_top - tick_h, y_top], color="black", linewidth=lw_br, clip_on=False)
    ax.text(1.5, y_top + tick_h * 0.4, sig,
            ha="center", va="bottom", fontsize=fs, color="black")


def _make_panel(gemini_csv, qwen_csv, group_col, group1, group2,
                label1, label2, color1, color2, preprocess,
                bracket_height_factor, out_path) -> None:
    lw    = LW["regression"] * 0.8
    lw_br = lw * 0.35

    def _load(csv_path):
        df = pd.read_csv(csv_path).dropna(
            subset=["human_perceived_msv", "predicted_msv"])
        if preprocess is not None:
            df = preprocess(df)
        return df[df[group_col].isin([group1, group2])].copy()

    df_g = _load(gemini_csv)
    df_q = _load(qwen_csv)

    fig, axes = plt.subplots(1, 3, figsize=(11, 5))
    for i, (ax, col_title, df, ycol) in enumerate(zip(
        axes,
        ["Human", "Gemini", "Qwen"],
        [df_g, df_g, df_q],
        ["human_perceived_msv", "predicted_msv", "predicted_msv"],
    )):
        v1 = df[df[group_col] == group1][ycol].values
        v2 = df[df[group_col] == group2][ycol].values
        bp = ax.boxplot(
            [v1, v2], positions=[1, 2], widths=0.55, patch_artist=True,
            medianprops=dict(color="white", linewidth=3.0),
            whiskerprops=dict(linewidth=2.0),
            capprops=dict(linewidth=2.0),
            flierprops=dict(marker="o", markersize=4, alpha=0.3, linewidth=0),
            boxprops=dict(linewidth=2.0),
        )
        for patch, c in zip(bp["boxes"], [color1, color2]):
            patch.set_facecolor(c); patch.set_alpha(0.8)
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

    def _interaction_p(df, ai_col):
        long = pd.concat([
            df[["participant_id", group_col, "human_perceived_msv"]].rename(
                columns={"human_perceived_msv": "rating"}).assign(source=0),
            df[["participant_id", group_col, ai_col]].rename(
                columns={ai_col: "rating"}).assign(source=1),
        ], ignore_index=True)
        long["group_bin"] = (long[group_col] == group2).astype(int)
        long = long.dropna(subset=["rating"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mixedlm("rating ~ source * group_bin", long,
                          groups=long["participant_id"]).fit(reml=True, method="powell")
        return float(res.pvalues["source:group_bin"])

    p_g = _interaction_p(df_g, "predicted_msv")
    p_q = _interaction_p(df_q, "predicted_msv")

    fig.tight_layout(w_pad=5.0)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[fig4] → {out_path}  "
          f"[interaction Gemini p={p_g:.4f}, Qwen p={p_q:.4f}]")


def generate(args, out_age: Path, out_gender: Path) -> None:
    for out in (out_age, out_gender):
        out.parent.mkdir(parents=True, exist_ok=True)

    def _prep_age(df):
        import pandas as _pd
        age = _pd.to_numeric(df["age"], errors="coerce").clip(18, 100)
        df = df.copy()
        df["age_group"] = age.apply(
            lambda a: ("Younger (≤38)" if _pd.notna(a) and a <= 38
                       else ("Older (≥39)" if _pd.notna(a) else None))
        )
        return df

    _make_panel(
        args.csv, args.qwen_csv,
        group_col="age_group",
        group1="Younger (≤38)", group2="Older (≥39)",
        label1="< 39",          label2="≥ 39",
        color1=MALE_COLOR,      color2=FEMALE_COLOR,
        preprocess=_prep_age,
        bracket_height_factor=1.08,
        out_path=out_age,
    )
    _make_panel(
        args.csv, args.qwen_csv,
        group_col="gender",
        group1="Male",       group2="Female",
        label1="Male",       label2="Female",
        color1=MALE_COLOR,   color2=FEMALE_COLOR,
        preprocess=None,
        bracket_height_factor=1.08,
        out_path=out_gender,
    )


def main() -> None:
    args = parse_args()
    generate(args, Path("figures") / "pmsv_age.pdf", Path("figures") / "pmsv_gender.pdf")


if __name__ == "__main__":
    main()
