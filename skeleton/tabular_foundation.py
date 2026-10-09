"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import balanced_accuracy_score
from time import perf_counter
from data_loading import DataSplits, prepare_final_data
from tabpfn import TabPFNClassifier


def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    """
    X_train, y_train, X_test, y_test = prepare_final_data(splits)

    full_train_size = len(y_train)

    model = TabPFNClassifier(device="auto", random_state=seed, )

    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = perf_counter() - start

    metrics = {"balanced_accuracy": float(balanced_accuracy_score(y_test, predictions))}

    return {
        "model": "TabPFN",
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": fit_seconds + predict_seconds,
        "train_samples_used": len(y_train),
        "train_samples_available": full_train_size,
        "test_samples": len(y_test),
    }
