"""Example runner for a shared split, forest search, and final evaluation.

Adapt this flow to your experimental design. Results stay in memory; choose how
to save them and record the settings needed to reproduce your study.
"""

from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path
from time import perf_counter
from typing import Any
import pandas as pd
from data_loading import DATASETS, load_and_split, prepare_data, prepare_final_data, load_dataset
from hyperband import optimise_hyperband
from random_forest import final_test_evaluation, make_evaluator
from random_search import optimise_random_search
from smbo import optimise_smbo
from tabular_foundation import run_foundation_model
import csv

# Update this if the provided largest dataset is replaced.
FOUNDATION_DATASET = "covertype"

# n_trials is the example evaluation budget for each of Random Search and SMBO.
# Choose budgets and a Hyperband schedule that support your justified comparison.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_samples": 2_500,
        "min_trees": 3,
        "max_trees": 27,
        "n_trials": 9,
    },
    "course": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 81,
        "n_trials": 16,
    },
    "full": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 243,
        "n_trials": 24,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="breast-w", choices=[*DATASETS, "all"])
    parser.add_argument("--profile", default="smoke", choices=PROFILES)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["default", "random", "smbo", "hyperband"],
        choices=["default", "random", "smbo", "hyperband", "foundation"],
    )
    parser.add_argument( "--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/main"))
    return parser.parse_args()


def run_dataset(name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return example in-memory results; their structure is yours to adapt."""

    if args.methods == ["foundation"] and name != FOUNDATION_DATASET:
        if args.dataset == "all":
            return []
        raise ValueError(f"Foundation-only runs require --dataset {FOUNDATION_DATASET}")

    profile = PROFILES[args.profile]
    splits = load_and_split(name, args.cache_dir, profile["max_samples"], args.split_seed)
    results = []
    forest_methods = {"default", "random", "smbo", "hyperband"}.intersection(args.methods)
    if forest_methods:
        X_train, X_valid = prepare_data(splits)
        evaluator = make_evaluator(
            X_train, splits.y_train, X_valid, splits.y_valid
        )

        def timed_evaluator(config, n_trees, seed):
            result = evaluator(config, n_trees, seed)
            result["cumulative_seconds"] = perf_counter() - search_start
            return result

        final_arrays = prepare_final_data(splits)
        min_trees = int(profile["min_trees"])
        max_trees = int(profile["max_trees"])

        for method in ("default", "random", "smbo", "hyperband"):
            if method not in forest_methods:
                continue
            search_start = perf_counter()
            if method == "default":
                config = {}  # Library defaults, with the common tree count.
                history = [timed_evaluator(config, max_trees, args.seed)]
            elif method == "random":
                config, history = optimise_random_search(
                    timed_evaluator, profile["n_trials"], max_trees, args.seed
                )
            elif method == "smbo":
                config, history = optimise_smbo(
                    timed_evaluator, profile["n_trials"], max_trees, args.seed
                )
            else:
                config, history = optimise_hyperband(
                    timed_evaluator, min_trees, max_trees, args.seed
                )
            search_seconds = perf_counter() - search_start
            final_result = final_test_evaluation(
                config, max_trees, args.seed, *final_arrays
            )
            print(f"{name} / {method} / seed {args.seed}: {final_result}", flush=True)
            results.append({
                "dataset": name,
                "method": method,
                "seed": args.seed,
                "configuration": config,
                "history": history,
                "search_seconds": search_seconds,
                "final_result": final_result,
            })

    if "foundation" in args.methods and name == FOUNDATION_DATASET:
        result = run_foundation_model(splits, seed=args.seed)
        print(f"{name} / foundation / seed {args.seed}: {result}", flush=True)
        results.append({
            "dataset": name,
            "method": "foundation",
            "seed": args.seed,
            "result": result,
        })
    return results

def save_results(results, output_dir, max_trees):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    progress = []

    for run in results:
        dataset = run["dataset"]
        method = run["method"]
        seed = run["seed"]

        if method == "foundation":
            result = run["result"]
            summary.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "test_balanced_accuracy": result["metrics"]["balanced_accuracy"],
                "search_seconds": 0.0,
                "final_seconds": result["total_seconds"],
                "total_seconds": result["total_seconds"],
            })
            continue

        final = run["final_result"]
        search_time = run["search_seconds"]
        final_time = final["elapsed_sec"]

        summary.append({
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "test_balanced_accuracy": final["metrics"]["balanced_accuracy"],
            "search_seconds": search_time,
            "final_seconds": final_time,
            "total_seconds": search_time + final_time,
        })

        if method == "default":
            continue

        best = float("-inf")

        for trial in run["history"]:
            if trial["n_trees"] == max_trees:
                best = max(best, trial["objective"])
            progress.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "cumulative_seconds": trial["cumulative_seconds"],
                "best_objective": ( best if best != float("-inf") else "" ),
            })

    summary_columns = ["dataset", "method", "seed", "test_balanced_accuracy", "search_seconds", "final_seconds", "total_seconds"]
    progress_columns = ["dataset", "method", "seed", "cumulative_seconds", "best_objective"]

    for filename, rows, columns in [
        ("summary.csv", summary, summary_columns),
        ("progress.csv", progress, progress_columns),
    ]:
        with open(output_dir / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

def inspect_data():
    for name in DATASETS:
        X, y = load_dataset(name)
        splits = load_and_split(name)

        print(f"\n=== {name} ===")

        print("Samples:", X.shape[0])
        print("Features:", X.shape[1])

        print("Numerical features:",
              len(X.select_dtypes(include=["number", "bool"]).columns))

        print("Categorical features:",
              len(X.select_dtypes(exclude=["number", "bool"]).columns))

        print("Missing values:", X.isna().sum().sum())

        print("Classes:", len(np.unique(y)))

        print("Class distribution:")
        print(pd.Series(y).value_counts(normalize=True))

        print(splits.X_train.shape)
        print(splits.X_valid.shape)
        print(splits.X_test.shape)

def main() -> None:
    """Run the selected examples; add result saving before the main study."""

    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_results = []
    output_dir = args.results_dir
    max_trees = int(PROFILES[args.profile]["max_trees"])
    for name in names:
        for seed in args.seeds:
            args.seed = seed
            print(f"Running {name} with seed {args.seed}", flush=True)
            results = run_dataset(name, args)
            all_results.extend(results)
            save_results(all_results, output_dir, max_trees)
            # TODO: save results in a format of your choice, along with the settings
            # needed to reproduce the run. Retain enough information for your plots
            # and tables. This example only prints final results; it saves no files.


if __name__ == "__main__":

    main()
