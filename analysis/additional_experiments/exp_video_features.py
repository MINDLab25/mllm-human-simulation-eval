"""
Video features analysis: can non-LLM baselines predict individual PMSV ratings?

Reproduces the RF analysis in the paper appendix (Appendix: Video features).
Adapted from Xue et al. (2026): 20 video features (3 audio + 17 visual).

Conditions tested:
  1. Global Mean             — predict training mean for every test row
  2. Video Mean              — per-video training mean (leave-participants-out)
  3. Video ID — RF           — one-hot video ID as features
  4. Video ID + Profile — RF — video ID + participant profile features
  5. Video Features — RF     — 20 video features
  6. Video Features + Profile — RF — 20 video features + participant profile

Evaluation: 100 repeated participant-level 80/20 train/test splits.
Key claim: adding profiles to video features increased R² by +0.026 and
reduced RMSE by −0.019 (p<.01; paired t-tests).

Run from repo root:
    python analysis/additional_experiments/exp_video_features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

PARTICIPANT_DATA = Path("data/msv_df_by_participant.csv")
VIDEO_DATA       = Path("data/msv_df.csv")
N_SPLITS  = 100
TEST_FRAC = 0.20
SEED      = 42

# 20 video features from Xue et al. (2026)
VIDEO_FEAT_COLS = [
    "loudness", "tempo", "spec_centroid",                # audio (3)
    "brightness", "warmness", "saturation", "entropy",   # visual static (4)
    "shotc_ps", "shotd_ps",                              # visual dynamic (2)
    "face_size", "face_age_avg", "face_female",          # face static
    "face_anger_avg", "face_disgust_avg", "face_fear_avg",
    "face_happiness_avg", "face_sadness_avg", "face_surprise_avg",  # emotions (9)
    "facec_ps", "faced_ps",                              # face dynamic (2)
]
PROFILE_NUM_COLS = ["age", "sen_seek"]
PROFILE_CAT_COLS = ["gender_clean", "race_simple", "education", "income"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    part = pd.read_csv(PARTICIPANT_DATA).dropna(subset=["perceived_msv"])
    part["age"] = part["age"].clip(upper=100)
    part["race_simple"] = part["race"].apply(lambda r: (
        "White" if pd.notna(r) and "," not in str(r) and "White" in str(r) else
        "Black" if pd.notna(r) and "," not in str(r) and "Black" in str(r) else
        "Asian" if pd.notna(r) and "," not in str(r) and "Asian" in str(r) else
        "Other"
    ))
    part["gender_clean"] = part["gender"].apply(
        lambda x: x if x in ("Male", "Female") else "Other"
    )
    # Treat education and income as categorical strings
    part["education"] = part["education"].fillna(0).astype(int).astype(str)
    part["income"]    = part["income"].fillna(0).astype(int).astype(str)

    vid = pd.read_csv(VIDEO_DATA)[["video_id"] + VIDEO_FEAT_COLS]
    return part.merge(vid, on="video_id", how="left").reset_index(drop=True)


# ── Participant split ─────────────────────────────────────────────────────────

def participant_split(participants: np.ndarray, seed: int):
    unique_pids = np.array(sorted(set(participants)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    n_test = max(1, int(len(unique_pids) * TEST_FRAC))
    test_pids = set(unique_pids[:n_test])
    test_mask = np.array([p in test_pids for p in participants])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


# ── Feature builders (fit on train, transform both) ───────────────────────────

def profile_X(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray):
    num = df[PROFILE_NUM_COLS].fillna(df[PROFILE_NUM_COLS].median()).to_numpy(float)
    cat = df[PROFILE_CAT_COLS].astype(str).to_numpy()
    sc  = StandardScaler(); sc.fit(num[tr])
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore"); enc.fit(cat[tr])
    X = np.hstack([sc.transform(num), enc.transform(cat)])
    return X[tr], X[te]


def video_feat_X(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray):
    V = df[VIDEO_FEAT_COLS].to_numpy(float)
    medians = np.nanmedian(V[tr], axis=0)
    for j in range(V.shape[1]):
        V[np.isnan(V[:, j]), j] = medians[j]
    sc = StandardScaler(); sc.fit(V[tr])
    V = sc.transform(V)
    return V[tr], V[te]


def video_id_X(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    vids = df[["video_id"]].to_numpy()
    enc.fit(vids[tr])
    X = enc.transform(vids)
    return X[tr], X[te]


def video_mean_pred(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
                    y: np.ndarray) -> np.ndarray:
    vmeans = df.iloc[tr].groupby("video_id")["perceived_msv"].mean().to_dict()
    grand  = y[tr].mean()
    return df["video_id"].map(vmeans).fillna(grand).to_numpy(float)[te]


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho = pearsonr(y_true, y_pred)[0] if y_pred.std() > 1e-9 else float("nan")
    return {
        "r":    rho,
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

COND_NAMES = [
    "1. Global Mean",
    "2. Video Mean",
    "3. Video ID — RF",
    "4. Video ID + Profile — RF",
    "5. Video Features — RF",
    "6. Video Features + Profile — RF",
]

def main() -> None:
    df = load()
    y = df["perceived_msv"].to_numpy(float)
    participants = df["participant_id"].to_numpy()
    print(f"N rows: {len(df)}  |  participants: {df.participant_id.nunique()}  |  videos: {df.video_id.nunique()}\n")

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(0, 1_000_000, size=N_SPLITS)
    records: dict[str, list[dict]] = {c: [] for c in COND_NAMES}

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    for i, seed in enumerate(seeds):
        tr, te = participant_split(participants, int(seed))

        # 1. Global Mean
        records["1. Global Mean"].append(evaluate(y[te], np.full(len(te), y[tr].mean())))

        # 2. Video Mean
        records["2. Video Mean"].append(evaluate(y[te], video_mean_pred(df, tr, te, y)))

        # 3. Video ID — RF
        vid_tr, vid_te = video_id_X(df, tr, te)
        rf.fit(vid_tr, y[tr])
        records["3. Video ID — RF"].append(evaluate(y[te], rf.predict(vid_te)))

        # 4. Video ID + Profile — RF
        pro_tr, pro_te = profile_X(df, tr, te)
        rf.fit(np.hstack([vid_tr, pro_tr]), y[tr])
        records["4. Video ID + Profile — RF"].append(evaluate(y[te], rf.predict(np.hstack([vid_te, pro_te]))))

        # 5. Video Features — RF
        vf_tr, vf_te = video_feat_X(df, tr, te)
        rf.fit(vf_tr, y[tr])
        records["5. Video Features — RF"].append(evaluate(y[te], rf.predict(vf_te)))

        # 6. Video Features + Profile — RF
        pro_tr2, pro_te2 = profile_X(df, tr, te)
        rf.fit(np.hstack([vf_tr, pro_tr2]), y[tr])
        records["6. Video Features + Profile — RF"].append(evaluate(y[te], rf.predict(np.hstack([vf_te, pro_te2]))))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_SPLITS} splits done …")

    # Print summary
    metrics = ["r", "R2", "MAE", "RMSE"]
    header = f"{'Condition':<35}" + "".join(f"{m:>10}" for m in metrics)
    print(f"\n{header}")
    print("─" * (35 + 10 * len(metrics)))
    for name in COND_NAMES:
        row = f"{name:<35}"
        for m in metrics:
            vals = np.array([s[m] for s in records[name]])
            ci   = 1.96 * vals.std() / np.sqrt(len(vals))
            row += f" {np.nanmean(vals):>6.4f}±{ci:.3f}"
        print(row)

    # Delta test: condition 6 vs 5
    def _arr(cond, m):
        return np.array([s[m] for s in records[cond]])

    for metric in ["R2", "RMSE"]:
        a = _arr("5. Video Features — RF", metric)
        b = _arr("6. Video Features + Profile — RF", metric)
        delta = b - a
        _, p_t = ttest_rel(b, a)
        _, p_w = wilcoxon(b, a)
        ci = 1.96 * delta.std() / np.sqrt(len(delta))
        sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
        print(f"\nΔ(6−5) {metric}: {delta.mean():+.4f} ±{ci:.4f}  "
              f"t-test p={p_t:.4f} {sig}  Wilcoxon p={p_w:.4f}")


if __name__ == "__main__":
    main()
