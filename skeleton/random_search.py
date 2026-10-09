"""Optional interface for Random Search in the common forest space.

Implement this loop, connect a package, or use another organisation. Choose how
to retain the results needed to analyse search progress and computational effort.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from random_forest import Config, Evaluator, sample_configuration


def optimise_random_search(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: randomly sample and evaluate up to n_trials configurations.

    Use the shared search space and train each forest with n_trees trees.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """
    rng = np.random.default_rng(seed)

    best_config = None         # start with empty config
    best_score = -float("inf") # trying to maximise score
    history = []               # keep tract of trials

    for trial in range(n_trials):
        config = sample_configuration(rng)          # generate a randon config
        result = evaluator(config, n_trees, seed)   # evaluate and produce result
        history.append(result)

        if result["objective"] > best_score:        # check config performance
            best_score = result["objective"]
            best_config = config

    return best_config, history
