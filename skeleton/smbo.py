"""Optional SMBO interface for the shared forest search space.

Implement the method, connect a suitable package, or replace this interface.
Choose and explain the method's settings and how you retain search results.
"""

from __future__ import annotations

from typing import Any
import optuna
from random_forest import Config, Evaluator, SEARCH_SPACE


def optimise_smbo(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: use SMBO to choose configurations based on previous evaluations.

    Use the shared search space and train each forest with n_trees trees.
    Use up to n_trials evaluations, including any initial evaluations.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    history = []

    def SMBO_trial(trial):
        config = {}

        for parameter, choices in SEARCH_SPACE.items():
            # Sampling is suggested by Optuna using info from earlier trial evaluations
            config[parameter] = trial.suggest_categorical(parameter, list(choices))

        result = evaluator(config, n_trees, seed)

        record = {
            **result,
            "trial": trial.number,
        }

        history.append(record)

        # returned performance value is used to determine which configuration to suggest next
        return float(result["objective"])

    study.optimize(SMBO_trial, n_trials=n_trials, n_jobs=1)

    # select the best configuration found
    best_config = dict(study.best_trial.params)

    return best_config, history