"""Run and aggregate the professor-requested multi-seed UBMEC study.

Examples (Colab/Linux):
  python run_ubmec_multiseed_study.py validation --ablation
  python run_ubmec_multiseed_study.py final --ablation
  python run_ubmec_multiseed_study.py aggregate --ablation

Run validation first, inspect the CSVs, and only then run the final stage.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_SEEDS = (42, 123, 2026, 3407, 777)
EXPERIMENT_VERSION = "professor-feedback-v7"


def load_analysis_dependencies():
    """Load plotting dependencies only for aggregation.

    Validation can therefore start in a fresh Colab runtime; the main
    experiment installs these packages before the later aggregate stage.
    """
    global plt, np, pd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validation", "final", "aggregate"))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--compact-profile",
        default="core",
        choices=("core", "full"),
        help=(
            "Predeclared 5-layer proposed profile used for every seed. "
            "Choose it using validation/pilot evidence, never test results."
        ),
    )
    parser.add_argument(
        "--tiny-profile",
        default="tiny_progressive",
        choices=("tiny_progressive", "tiny_balanced", "tiny_feature"),
        help=(
            "Predeclared 1-layer proposed profile used for every seed. "
            "Choose it using validation/pilot evidence, never test results."
        ),
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name(
            "ubmec_fixed_size_breakpoint_experiment.py"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ubmec_professor_multiseed_outputs"),
    )
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument(
        "--ablation-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run the ablation study only for these seeds (default: all seeds).",
    )
    parser.add_argument(
        "--shared-teacher-seed",
        type=int,
        default=None,
        help=(
            "Reuse the fine-tuned teacher of this seed for every other seed "
            "(saves ~40 min per seed on a T4; the teacher becomes a fixed reference)."
        ),
    )
    parser.add_argument(
        "--ablation-mode",
        default="cumulative",
        choices=("cumulative", "leave_one_out"),
        help="cumulative = Table 8 stages; leave_one_out = Table 9 rows.",
    )
    parser.add_argument(
        "--unseen-holdout",
        action="store_true",
        help="Use only when this split has genuinely never been inspected.",
    )
    return parser.parse_args()


def seed_runs_ablation(args, seed):
    if not args.ablation:
        return False
    return args.ablation_seeds is None or seed in args.ablation_seeds


def read_metadata(path):
    return json.loads(path.read_text(encoding="utf-8"))


def matching_run_directories(
    output_root,
    seed,
    split_seed,
    ablation,
    compact_profile,
    tiny_profile,
):
    matches = []
    for metadata_path in (output_root / "live").glob("*/run_metadata.json"):
        metadata = read_metadata(metadata_path)
        if (
            metadata.get("experiment_version") == EXPERIMENT_VERSION
            and metadata.get("training_seed") == seed
            and metadata.get("split_seed") == split_seed
            and bool(metadata.get("run_ablation_study")) == ablation
            and metadata.get("compact_proposed_profiles")
            == [compact_profile]
            and metadata.get("tiny_proposed_profiles") == [tiny_profile]
        ):
            matches.append(metadata_path.parent)
    return matches


def invoke_experiment(args, seed, final_test):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "UBMEC_RUN_MODE": "live",
            "UBMEC_TRAINING_SEED": str(seed),
            "UBMEC_SPLIT_SEED": str(args.split_seed),
            "UBMEC_OUTPUT_DIR": str(args.output_root.resolve()),
            "UBMEC_RUN_FINAL_TEST": "true" if final_test else "false",
            "UBMEC_RUN_ABLATION": (
                "true" if seed_runs_ablation(args, seed) else "false"
            ),
            "UBMEC_ABLATION_MODE": args.ablation_mode,
            "UBMEC_COMPACT_PROFILE": args.compact_profile,
            "UBMEC_TINY_PROFILE": args.tiny_profile,
            "UBMEC_PRIOR_TEST_EXPOSED": (
                "false" if args.unseen_holdout else "true"
            ),
        }
    )
    if (
        args.shared_teacher_seed is not None
        and seed != args.shared_teacher_seed
    ):
        teacher_runs = matching_run_directories(
            args.output_root,
            args.shared_teacher_seed,
            args.split_seed,
            seed_runs_ablation(args, args.shared_teacher_seed),
            args.compact_profile,
            args.tiny_profile,
        )
        teacher_files = [
            run_dir / "teacher_banglabert.pt"
            for run_dir in teacher_runs
            if (run_dir / "teacher_banglabert.pt.complete").exists()
        ]
        if not teacher_files:
            raise RuntimeError(
                "No completed teacher checkpoint found for shared teacher "
                f"seed {args.shared_teacher_seed}; run that seed first."
            )
        env["UBMEC_SHARED_TEACHER"] = str(teacher_files[0].resolve())
    print(
        f"\nRunning seed={seed}, stage="
        f"{'final' if final_test else 'validation'}, "
        f"ablation={seed_runs_ablation(args, seed)}"
    )
    subprocess.check_call(
        [sys.executable, str(args.script.resolve())],
        cwd=str(args.script.resolve().parent),
        env=env,
    )


def run_stage(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_test = args.stage == "final"
    protocol_plan = {
        "experiment_version": EXPERIMENT_VERSION,
        "training_seeds": list(args.seeds),
        "fixed_split_seed": args.split_seed,
        "compact_profile": args.compact_profile,
        "tiny_profile": args.tiny_profile,
        "ablation": args.ablation,
        "ablation_seeds": args.ablation_seeds,
        "shared_teacher_seed": args.shared_teacher_seed,
        "test_evidence": (
            "confirmatory_unseen_holdout"
            if args.unseen_holdout
            else "exploratory_previously_observed_holdout"
        ),
    }
    protocol_plan_path = args.output_root / "multiseed_protocol_plan.json"
    # Keys that define the scientific protocol and must never change inside
    # one output root. Seed lists, ablation scope and teacher sharing may be
    # extended across sessions, so they are merged into the saved plan.
    fixed_keys = (
        "experiment_version",
        "fixed_split_seed",
        "compact_profile",
        "tiny_profile",
        "test_evidence",
    )
    if protocol_plan_path.exists():
        existing_plan = read_metadata(protocol_plan_path)
        if existing_plan.get("experiment_version") != EXPERIMENT_VERSION:
            # A plan from an older protocol version: its run directories are
            # ignored by matching_run_directories, so archive it and start anew.
            archived = protocol_plan_path.with_name(
                "multiseed_protocol_plan_"
                f"{existing_plan.get('experiment_version', 'unknown')}.json"
            )
            protocol_plan_path.replace(archived)
            print(f"Archived old protocol plan to {archived}")
            existing_plan = None
    if protocol_plan_path.exists() and existing_plan is not None:
        for key in fixed_keys:
            if existing_plan.get(key) != protocol_plan[key]:
                raise RuntimeError(
                    "The saved multi-seed protocol differs from this command "
                    f"({key}). Use a new --output-root for a different protocol."
                )
        protocol_plan["training_seeds"] = sorted(
            set(existing_plan.get("training_seeds", [])) | set(args.seeds)
        )
        protocol_plan["session_history"] = existing_plan.get(
            "session_history", []
        ) + [
            {
                "stage": args.stage,
                "seeds": list(args.seeds),
                "ablation": args.ablation,
                "ablation_seeds": args.ablation_seeds,
                "shared_teacher_seed": args.shared_teacher_seed,
            }
        ]
    protocol_plan_path.write_text(
        json.dumps(protocol_plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if final_test:
        missing_locks = []
        for seed in args.seeds:
            matches = matching_run_directories(
                args.output_root,
                seed,
                args.split_seed,
                seed_runs_ablation(args, seed),
                args.compact_profile,
                args.tiny_profile,
            )
            if len(matches) != 1 or not (
                matches[0] / "selection_lock_manifest.json"
            ).exists():
                missing_locks.append(seed)
        if missing_locks:
            raise RuntimeError(
                "Final testing has not started. Validation locks are missing "
                f"for seeds {missing_locks}; complete validation for every "
                "declared seed first."
            )

    for seed in args.seeds:
        matches = matching_run_directories(
            args.output_root,
            seed,
            args.split_seed,
            seed_runs_ablation(args, seed),
            args.compact_profile,
            args.tiny_profile,
        )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple matching run directories found for seed {seed}: {matches}"
            )
        run_dir = matches[0] if matches else None
        final_result = (
            run_dir / "fixed_size_breakpoint_results.csv"
            if run_dir is not None
            else None
        )
        selection_lock = (
            run_dir / "selection_lock_manifest.json"
            if run_dir is not None
            else None
        )
        if final_test and final_result is not None and final_result.exists():
            print(f"Skipping completed final run for seed {seed}: {run_dir}")
            continue
        if not final_test and selection_lock is not None and selection_lock.exists():
            print(f"Skipping completed validation run for seed {seed}: {run_dir}")
            continue
        if final_test and (selection_lock is None or not selection_lock.exists()):
            raise RuntimeError(
                f"Seed {seed} has no validation lock. Run the validation stage first."
            )
        invoke_experiment(args, seed, final_test)


def collect_runs(args):
    run_records = []
    for seed in args.seeds:
        matches = matching_run_directories(
            args.output_root,
            seed,
            args.split_seed,
            seed_runs_ablation(args, seed),
            args.compact_profile,
            args.tiny_profile,
        )
        completed = [
            path
            for path in matches
            if (path / "fixed_size_breakpoint_results.csv").exists()
        ]
        if len(completed) != 1:
            raise RuntimeError(
                f"Expected one completed final run for seed {seed}; found {completed}"
            )
        run_records.append((seed, completed[0]))
    if len(run_records) < 3:
        raise RuntimeError("At least three completed seeds are required.")
    return run_records


def aggregate_main_results(run_records, aggregate_dir):
    frames = []
    for seed, run_dir in run_records:
        frame = pd.read_csv(run_dir / "fixed_size_breakpoint_results.csv")
        frame.insert(0, "Training Seed", seed)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(aggregate_dir / "multiseed_raw_results.csv", index=False)

    group_columns = [
        "Approach",
        "Comparison Range (MB)",
        "Architecture",
        "Layers",
        "KD Applied",
        "Quantization Applied",
        "Weight Precision",
        "File Format",
    ]
    summary = (
        raw.groupby(group_columns, dropna=False)
        .agg(
            Runs=("Training Seed", "nunique"),
            **{
                "Saved Weight Size Mean (MB)": (
                    "Saved Weight File Size (MB)",
                    "mean",
                ),
                "Saved Weight Size Std (MB)": (
                    "Saved Weight File Size (MB)",
                    "std",
                ),
                "Accuracy Mean (%)": ("Accuracy (%)", "mean"),
                "Accuracy Std (%)": ("Accuracy (%)", "std"),
                "Macro F1 Mean (%)": ("Macro F1 (%)", "mean"),
                "Macro F1 Std (%)": ("Macro F1 (%)", "std"),
            },
        )
        .reset_index()
    )
    summary.to_csv(
        aggregate_dir / "multiseed_mean_std_summary.csv", index=False
    )

    paired = raw[
        raw["Approach"].isin(
            ("Standard Logit-KD Control", "Proposed Method")
        )
    ].pivot_table(
        index=["Training Seed", "Comparison Range (MB)"],
        columns="Approach",
        values=["Accuracy (%)", "Macro F1 (%)"],
        aggfunc="first",
    )
    paired.columns = [f"{metric} {method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    for metric in ("Accuracy (%)", "Macro F1 (%)"):
        paired[f"{metric} Proposed-minus-Standard (point)"] = (
            paired[f"{metric} Proposed Method"]
            - paired[f"{metric} Standard Logit-KD Control"]
        )
    paired.to_csv(
        aggregate_dir / "paired_proposed_vs_standard_by_seed.csv", index=False
    )
    gain_columns = [
        column for column in paired.columns if "Proposed-minus-Standard" in column
    ]
    gain_summary = paired.groupby("Comparison Range (MB)")[gain_columns].agg(
        ["count", "mean", "std"]
    )
    gain_summary.to_csv(
        aggregate_dir / "paired_gain_mean_std.csv"
    )
    return raw, summary


def aggregate_ablations(run_records, aggregate_dir):
    frames = []
    for seed, run_dir in run_records:
        path = run_dir / "ablation_test_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["Training Seed"] = seed
        frames.append(frame)
    if not frames:
        return
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(aggregate_dir / "ablation_multiseed_raw.csv", index=False)
    summary = (
        raw.groupby(
            ["Architecture", "Ablation Key", "Ablation Step"],
            dropna=False,
        )
        .agg(
            Runs=("Training Seed", "nunique"),
            **{
                "Accuracy Mean (%)": ("Accuracy (%)", "mean"),
                "Accuracy Std (%)": ("Accuracy (%)", "std"),
                "Macro F1 Mean (%)": ("Macro F1 (%)", "mean"),
                "Macro F1 Std (%)": ("Macro F1 (%)", "std"),
            },
        )
        .reset_index()
    )
    summary.to_csv(
        aggregate_dir / "ablation_multiseed_mean_std.csv", index=False
    )


def plot_summary(summary, aggregate_dir):
    styles = {
        "12-layer BanglaBERT Teacher": ("#2563eb", "s"),
        "Direct Training Control (No KD)": ("#64748b", "^"),
        "Standard Logit-KD Control": ("#ea580c", "D"),
        "Proposed Method": ("#15803d", "o"),
    }
    for metric, stem in (
        ("Accuracy", "accuracy_multiseed_mean_std"),
        ("Macro F1", "macro_f1_multiseed_mean_std"),
    ):
        fig, axis = plt.subplots(figsize=(10, 6))
        for (approach, architecture), rows in summary.groupby(
            ["Approach", "Architecture"]
        ):
            rows = rows.sort_values("Saved Weight Size Mean (MB)")
            color, marker = styles[approach]
            label = f"{approach} ({int(rows.iloc[0]['Layers'])} layers)"
            axis.errorbar(
                rows["Saved Weight Size Mean (MB)"],
                rows[f"{metric} Mean (%)"],
                yerr=rows[f"{metric} Std (%)"].fillna(0),
                color=color,
                marker=marker,
                linestyle="-" if len(rows) > 1 else "None",
                linewidth=2,
                capsize=4,
                label=label,
            )
        axis.set_xlabel("Measured Saved Weight File Size (MB)")
        axis.set_ylabel(f"{metric} (%)")
        axis.set_title(f"{metric} vs. Model Size (Mean +/- SD Across Seeds)")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(aggregate_dir / f"{stem}.png", dpi=300)
        fig.savefig(aggregate_dir / f"{stem}.pdf")
        plt.close(fig)


def aggregate(args):
    run_records = collect_runs(args)
    aggregate_dir = args.output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    _, summary = aggregate_main_results(run_records, aggregate_dir)
    aggregate_ablations(run_records, aggregate_dir)
    plot_summary(summary, aggregate_dir)
    print(f"Aggregated {len(run_records)} seeds into: {aggregate_dir}")


def main():
    args = parse_args()
    args.script = args.script.resolve()
    args.output_root = args.output_root.resolve()
    if args.stage == "aggregate":
        load_analysis_dependencies()
        aggregate(args)
    else:
        run_stage(args)


if __name__ == "__main__":
    main()
