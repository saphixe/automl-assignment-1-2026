
import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv("results/main/summary.csv")
progress = pd.read_csv("results/main/progress.csv")

forest = summary[summary["method"] != "foundation"]

table1 = forest.pivot_table(
    index="dataset",
    columns="method",
    values="test_balanced_accuracy",
    aggfunc=["mean", "std"]
)

print(table1.round(4))

table1.to_csv("results/main/table1.csv")

# ----------------------------------------------------------------------------------------------------------------------


for dataset in progress["dataset"].unique():

    fig, ax = plt.subplots(figsize=(7, 4))

    data = progress[progress["dataset"] == dataset]

    for method in ["random", "smbo", "hyperband"]:

        method_data = data[data["method"] == method]

        for seed, run in method_data.groupby("seed"):

            run = run.sort_values("cumulative_seconds")
            run = run.dropna(subset=["best_objective"])

            ax.step(
                run["cumulative_seconds"],
                run["best_objective"],
                where="post",
                label=method if seed == method_data["seed"].iloc[0] else None,
                alpha=0.7
            )

    ax.set_xlabel("Cumulative search time (seconds)")
    ax.set_ylabel("Best validation balanced accuracy")
    ax.set_title(dataset)
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"results/main/progress_{dataset}.pdf")
    plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------


foundation = pd.read_csv("results/foundation/summary.csv")

covertype_forests = summary[
    summary["dataset"] == "covertype"
]

combined = pd.concat(
    [covertype_forests, foundation],
    ignore_index=True
)

table3 = combined.groupby("method").agg(
    test_balanced_accuracy=("test_balanced_accuracy", "mean"),
    computational_cost_seconds=("total_seconds", "mean")
)

print(table3.round(3))

table3.to_csv("results/main/table3.csv")


