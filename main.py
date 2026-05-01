"""
main.py — end-to-end PMSV inference runner.

Default mode is --sync (concurrent, incremental saves, resumable).
Use --batch for the Gemini Batch API (cheaper but no live progress).

Usage
-----
    # Sync (default) — use latest sample
    python main.py

    # Sync — draw a fresh sample first
    python main.py --new-sample

    # Sync — run on a specific saved sample
    python main.py --sample data/samples/10_percent_sample.csv

    # Sync — full dataset, 20 workers
    python main.py --full --workers 20

    # Sync — resume an interrupted run
    python main.py --resume outputs/results_20260312_124807.csv

    # Batch API mode
    python main.py --batch

    # Few-shot (3 fixed examples per prompt)
    python main.py --few-shot
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from pmsv_synth.config import (
    MAX_CONCURRENT,
    MSV_BY_PARTICIPANT_PATH,
    MSV_DF_PATH,
    OUTPUTS_DIR,
)
from pmsv_synth.data.export import results_to_comparison_csv, results_to_csv
from pmsv_synth.data.sampler import (
    _resolve_video_path,
    create_sample,
    get_participants_for_sample,
    load_full_dataset,
    load_latest_sample,
    load_sample,
)

HUMAN_ITEM_COLS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PMSV inference using Gemini (sync by default)."
    )

    # ── Sample source ───────────────────────────────────────────────────────
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--new-sample",
        action="store_true",
        help="Draw a fresh random sample and run inference on it.",
    )
    sample_group.add_argument(
        "--use-latest",
        action="store_true",
        help="(default) Load the most recently saved sample.",
    )
    sample_group.add_argument(
        "--sample",
        type=Path,
        metavar="PATH",
        help="Load a specific saved sample CSV.",
    )
    sample_group.add_argument(
        "--full",
        action="store_true",
        help="Run inference on the entire dataset (~1,200 videos).",
    )

    # ── Inference mode ──────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sync",
        action="store_true",
        default=False,
        help="(default) Concurrent sync mode — incremental saves, resumable.",
    )
    mode_group.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Gemini Batch API mode (async, cheaper, no live progress).",
    )

    # ── Provider ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        choices=["gemini", "qwen3-local"],
        default="gemini",
        help=(
            "Inference provider: 'gemini' (default) uses the Gemini API; "
            "'qwen3-local' runs Qwen3-Omni-30B-A3B-GPTQ-4bit on the local GPU "
            "(requires the pmsv-qwen3 conda env)."
        ),
    )

    # ── Sync-specific options ───────────────────────────────────────────────
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_CONCURRENT,
        metavar="N",
        help=f"Concurrent videos in sync mode (default: {MAX_CONCURRENT}).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="CSV",
        help=(
            "Resume an interrupted sync run. Pass the partial results CSV; "
            "already-processed (participant_id, video_id) pairs are skipped."
        ),
    )

    # ── Prompt options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--few-shot",
        action="store_true",
        default=False,
        help="Enable few-shot mode: prepend 3 fixed calibration examples to every prompt.",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        default=False,
        help=(
            "Chain-of-thought mode: the model reasons step-by-step before "
            "giving ratings. Reasoning is saved in the cot_reasoning column."
        ),
    )
    parser.add_argument(
        "--survey-order",
        action="store_true",
        default=False,
        help=(
            "Present items in the original Qualtrics anchor directions "
            "(some items reversed, e.g. Unique ←→ Common). "
            "Reverse-scoring is applied automatically after inference. "
            "Default: all items normalized to 1 = low, 7 = high sensation."
        ),
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        default=False,
        help=(
            "Omit the demographic profile from all prompts. "
            "The model rates videos as a generic respondent with no persona "
            "conditioning (content-only baseline)."
        ),
    )
    parser.add_argument(
        "--participants",
        type=Path,
        metavar="CSV",
        default=None,
        help=(
            "Override participants CSV (same schema as "
            "data/msv_df_by_participant.csv). Useful for running with "
            "shuffled demographic profiles produced by shuffle_profiles.py."
        ),
    )

    return parser.parse_args()


def _load_done_pairs(csv_path: Path) -> set[tuple[int, int]]:
    """Return the set of (participant_id, video_id) pairs already in *csv_path*."""
    df = pd.read_csv(csv_path, usecols=["participant_id", "video_id"])
    return set(zip(df["participant_id"].astype(int), df["video_id"].astype(int)))


def main() -> None:
    args = parse_args()
    use_batch = args.batch  # sync is the default when neither flag is given

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Load sample ───────────────────────────────────────────────────────
    if args.full:
        sample = load_full_dataset()
    elif args.new_sample:
        sample = create_sample()
    elif args.sample:
        sample = load_sample(args.sample)
    else:
        sample = load_latest_sample()

    # ── no-profile: one API call per video, one output row per video ──────────
    # Skip participant join entirely — the sample already has video-level human
    # averages (perceived_msv, and all 17 item columns) so we use it directly.
    if args.no_profile:
        participants = sample.copy()
        # participant_id is meaningless here; use video_id as a stand-in so the
        # rest of the pipeline (done_pairs, CSV schema) doesn't change.
        participants["participant_id"] = participants["video_id"]
        # Ensure the demographic columns expected by sync.py are present (as NaN)
        for col in ["age", "gender", "race", "education", "income", "sen_seek"]:
            if col not in participants.columns:
                participants[col] = None
        print(f"[main] No-profile mode: {len(participants)} videos — 1 API call per video.")
    else:
        participants = get_participants_for_sample(
            sample, participants_path=args.participants
        )
        n_videos = participants["video_id"].nunique()
        src_label = args.participants.name if args.participants else "msv_df_by_participant.csv"
        print(
            f"[main] {len(participants)} participant-video pairs across {n_videos} videos "
            f"(profiles from {src_label})."
        )

    # ── 2a. Sync mode (default) ──────────────────────────────────────────────
    if not use_batch:
        if args.provider == "qwen3-local":
            from pmsv_synth.inference.qwen3_local import run_sync
        else:
            from pmsv_synth.inference.gemini import run_sync

        output_csv = OUTPUTS_DIR / f"results_{timestamp}.csv"
        done_pairs: set[tuple[int, int]] = set()

        if args.resume and args.resume.exists():
            existing = pd.read_csv(args.resume)
            # Only treat rows with a successful prediction as done so that
            # failed rows (predicted_msv is NaN) are retried on resume.
            successful = existing[existing["predicted_msv"].notna()]
            done_pairs = {
                (int(r["participant_id"]), int(r["video_id"]))
                for _, r in successful.iterrows()
            }
            output_csv = args.resume
            print(
                f"[main] Resuming {args.resume.name} — "
                f"{len(done_pairs)} pairs already done."
            )

        survey_df = None
        video_path_map = None
        if args.few_shot:
            print("[main] Few-shot mode: 3 fixed examples per prompt (multimodal).")
            survey_df = pd.read_csv(MSV_BY_PARTICIPANT_PATH)
            msv_df = pd.read_csv(MSV_DF_PATH)
            video_path_map = {}
            for _, vrow in msv_df.iterrows():
                try:
                    video_path_map[int(vrow["video_id"])] = _resolve_video_path(
                        vrow["survey_batch_id"]
                    )
                except FileNotFoundError:
                    pass
            print(f"[main] Resolved paths for {len(video_path_map)} videos.")

        if args.cot:
            print("[main] Chain-of-thought mode enabled.")

        if args.no_profile:
            print("[main] No-profile mode: demographic profile omitted from all prompts.")

        if args.survey_order:
            print(
                "[main] Survey-order mode: reversed items presented as-is; reverse-scoring applied post-inference."
            )

        results = run_sync(
            participants,
            HUMAN_ITEM_COLS,
            output_csv=output_csv,
            done_pairs=done_pairs,
            max_concurrent=args.workers,
            n_shots=int(args.few_shot),
            survey_df=survey_df,
            video_path_map=video_path_map,
            use_cot=args.cot,
            survey_order=args.survey_order,
            no_profile=args.no_profile,
        )

        n_success = sum(1 for r in results if r.get("predicted_msv") is not None)
        print(f"\n[main] Done. {n_success}/{len(results)} successful predictions.")
        print(f"       Results CSV → {output_csv}")

        # Export comparison CSV
        try:
            comparison_path = results_to_comparison_csv(output_csv)
            print(f"       Comparison  → {comparison_path}")
        except Exception as exc:
            print(f"[main] Comparison CSV failed: {exc}")

        return

    # ── 2b. Batch API mode ───────────────────────────────────────────────────
    from pmsv_synth.inference.gemini import (
        get_client,
        build_inline_requests,
        delete_files,
        parse_batch_results,
        poll_batch_job,
        submit_batch_job,
        upload_videos,
    )

    client = get_client()

    if args.resume:
        # Resume existing batch job by name
        job_name = str(args.resume)
        print(f"[main] Resuming batch job: {job_name}")
        job = poll_batch_job(client, job_name)
        file_names: list[str] = []
    else:
        survey_df = pd.read_csv(MSV_BY_PARTICIPANT_PATH) if args.few_shot else None
        if args.few_shot:
            print("[main] Few-shot mode: 3 fixed examples per prompt.")

        uri_map, file_names = upload_videos(participants, client=client)
        inline_requests = build_inline_requests(
            participants,
            uri_map,
            n_shots=int(args.few_shot),
            survey_df=survey_df,
            survey_order=args.survey_order,
        )
        job = submit_batch_job(
            inline_requests,
            client=client,
            display_name=f"pmsv-inference-{timestamp}",
        )
        job = poll_batch_job(client, job.name)

    results = parse_batch_results(
        job, participants, HUMAN_ITEM_COLS, survey_order=args.survey_order
    )

    output_json = OUTPUTS_DIR / f"results_{timestamp}.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    n_success = sum(1 for r in results if r["predicted_msv"] is not None)
    csv_path = results_to_csv(output_json)
    comparison_path = results_to_comparison_csv(output_json)

    print(
        f"\n[main] Done. {n_success}/{len(results)} successful predictions."
        f"\n       JSON        → {output_json}"
        f"\n       Results CSV → {csv_path}"
        f"\n       Comparison  → {comparison_path}"
    )

    if file_names:
        delete_files(client, file_names)


if __name__ == "__main__":
    main()
