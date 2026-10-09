"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any
import math
import numpy as np
from random_forest import Config, Evaluator, sample_configuration


def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any]:
    """TODO: implement or configure multiple successive-halving brackets.

    Use the shared search space. Start brackets with different numbers of
    configurations and trees per forest, between min_trees and max_trees.
    At each stage, keep the better configurations and give them more trees,
    keeping other settings fixed. Use reduction_factor for the decrease in
    configuration count and increase in trees.
    Compare validation objectives consistently: respect whether higher or lower
    values are better. Explain your schedule, refitting or warm starts, and
    how validation results determine the final selection.
    Return the selected configuration and results needed for your analysis.
    """

    rng = np.random.default_rng(seed)

    # number of brackets
    bracket_max = math.floor(math.log(max_trees / min_trees, reduction_factor) + 1e-12)

    history = []
    candidates = []

    # process each bracket starting from the biggest
    for bracket in range(bracket_max, -1, -1):
        # the starting amount of configurations
        n_config = math.ceil((bracket_max + 1) / (bracket + 1) * reduction_factor ** bracket)
        # the starting amount of trees
        initial_trees = max_trees / (reduction_factor ** bracket)
        # gather the configurations
        configurations = [sample_configuration(rng) for _ in range(n_config)]

        # run successive halving rounds
        for round in range(bracket + 1):
            # determine the number of trees for this round
            n_trees = min(max_trees, max(min_trees, math.ceil(initial_trees * reduction_factor ** round)))

            results = []

            # evaluate each config in the bracket
            for config in configurations:
                result = evaluator(config, n_trees, seed)

                record = {
                    **result,
                    "bracket": bracket,
                    "round": round,
                }

                history.append(record)
                results.append(record)

                # configurations evaluated at max trees budget are chosen for final evaluation
                if n_trees == max_trees:
                    candidates.append(record)

            # sort configurations by objective performance in descending order
            results.sort(key=lambda item: item["objective"], reverse=True)

            # eliminates configs if the final rounds isn't reached yet
            if round < bracket:
                # calculate how many configurations to keep
                n_keep = max(1, math.floor(len(results) / reduction_factor))
                # keeps the top n_keep configurations for the next round
                configurations = [item["configuration"] for item in results[:n_keep]]

            # configurations are maintained but a new random forest is made which increases compute

    # find best performing configuration
    best_result = max(candidates, key=lambda item: item["objective"])

    return best_result["configuration"], history

