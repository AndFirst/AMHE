import argparse
import csv
import os
import time

import cocoex
import cocopp
import numpy as np

from algorithms.CMAES import CMAES
from algorithms.MAES import MAES
from ipop.BudgetIPOP import BudgetIPOP
from ipop.OptimizerAlgorithmWrapper import OptimizerAlgorithmWrapper
from ipop.PeriodicRestartIPOP import PeriodicRestartIPOP
from ipop.StagnationIPOP import StagnationIPOP
from ipop.ThresholdIPOP import ThresholdIPOP


def run_single_experiment(
    wrapper_class, wrapper_params, optimizer_class, problem, max_iterations
):
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    mean_range = (lower, upper)
    sigma_range = (0.1, 1.0)

    initial_sigma = np.random.uniform(0.1, 1.0)
    initial_mean = np.random.uniform(lower, upper)

    wrapper_init_args = {
        "optimizer_class": optimizer_class,
        "objective_function": problem,
        "initial_mean": initial_mean,
        "initial_sigma": initial_sigma,
        "population_size": None,
        "dim": problem.dimension,
        **{
            k: v
            for k, v in wrapper_params.items()
            if k not in ["name", "exclude_ranges"]
        },
    }

    if not wrapper_params.get("exclude_ranges", False):
        wrapper_init_args["mean_range"] = mean_range
        wrapper_init_args["sigma_range"] = sigma_range

    wrapper = wrapper_class(**wrapper_init_args)

    start_time = time.process_time()
    best_solution, best_fitness, *_ = wrapper.run(max_iterations=max_iterations)
    end_time = time.process_time()

    return best_solution, best_fitness, round(end_time - start_time, 8)


def construct_output_folder(wrapper_name, optimizer_name, params):
    if wrapper_name == "dummy":
        return f"{optimizer_name}"
    postfix = "_".join(
        [f"{v}" for k, v in params.items() if k not in ["name"]]
    )
    return f"{wrapper_name[0]}_{optimizer_name[0]}_{postfix}"


def append_result_to_csv(output_path, row_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.isfile(output_path)
    with open(output_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main(args):
    suite_name = "bbob"
    max_sweeps = 3

    base_optimizers = {
        "maes": MAES,
        "cmaes": CMAES,
    }

    wrappers_per_optimizer = {
        "maes": [
            (OptimizerAlgorithmWrapper, {"name": "dummy"}),
            (
                StagnationIPOP,
                {
                    "name": "stagnation_restart",
                    "stagnation_limit": 15,
                    "pop_increase_factor": 2,
                },
            ),
            (
                StagnationIPOP,
                {
                    "name": "stagnation_restart",
                    "stagnation_limit": 30,
                    "pop_increase_factor": 2,
                },
            ),
            (
                StagnationIPOP,
                {
                    "name": "stagnation_restart",
                    "stagnation_limit": 30,
                    "pop_increase_factor": 2,
                    "exclude_ranges": True,
                },
            ),
            (
                PeriodicRestartIPOP,
                {
                    "name": "periodic_restart",
                    "restart_interval": 50,
                    "pop_increase_factor": 2,
                },
            ),
            (
                PeriodicRestartIPOP,
                {
                    "name": "periodic_restart",
                    "restart_interval": 100,
                    "pop_increase_factor": 2,
                },
            ),
            (
                BudgetIPOP,
                {
                    "name": "budget_reset",
                    "restart_budget": 10000,
                    "pop_increase_factor": 2,
                },
            ),
            (
                ThresholdIPOP,
                {
                    "name": "threshold_reset",
                    "evaluation_window": 10,
                    "improvement_threshold": 1e-3,
                    "pop_increase_factor": 2,
                },
            ),
            (
                ThresholdIPOP,
                {
                    "name": "threshold_reset",
                    "evaluation_window": 20,
                    "improvement_threshold": 1e-3,
                    "pop_increase_factor": 2,
                },
            ),
            (
                ThresholdIPOP,
                {
                    "name": "threshold_reset",
                    "evaluation_window": 10,
                    "improvement_threshold": 1e-6,
                    "pop_increase_factor": 2,
                },
            ),
            (
                ThresholdIPOP,
                {
                    "name": "threshold_reset",
                    "evaluation_window": 20,
                    "improvement_threshold": 1e-6,
                    "pop_increase_factor": 2,
                },
            ),
        ],
        "cmaes": [
            (OptimizerAlgorithmWrapper, {"name": "dummy"}),
            (
                StagnationIPOP,
                {
                    "name": "stagnation_restart",
                    "stagnation_limit": 30,
                    "pop_increase_factor": 2,
                    "exclude_ranges": True,
                },
            ),
        ],
    }

    result_folders = []

    for opt_name, optimizer_class in base_optimizers.items():
        for wrapper_class, wrapper_params in wrappers_per_optimizer[opt_name]:
            suite = cocoex.Suite(suite_name, "", "")
            output_folder = construct_output_folder(
                wrapper_params["name"], opt_name, wrapper_params
            )
            result_folders.append(os.path.join("exdata", output_folder))
            csv_output_path = os.path.join("csv_results", f"{output_folder}.csv")

            observer = cocoex.Observer(suite_name, f"result_folder: {output_folder}")
            minimal_print = cocoex.utilities.MiniPrint()

            for sweep in range(1, max_sweeps + 1):
                for problem in suite:
                    problem.observe_with(observer)
                    problem(problem.dimension * [0])

                    best_solution, best_fitness, time_taken = run_single_experiment(
                        wrapper_class, wrapper_params, optimizer_class, problem, args.e
                    )

                    minimal_print(problem, f"{output_folder} time: {time_taken}s")

                    result_row = {
                        "optimizer": opt_name,
                        "wrapper": wrapper_params["name"],
                        "problem_id": problem.id,
                        "dimension": problem.dimension,
                        "best_fitness": best_fitness,
                        "evaluations": problem.evaluations,
                        "sweep": sweep,
                        "time_one_iteration": time_taken,
                    }

                    for k, v in wrapper_params.items():
                        if k not in ["name", "exclude_ranges"]:
                            result_row[f"param_{k}"] = v

                    append_result_to_csv(csv_output_path, result_row)

    # cocopp.main(result_folders) # usun do tylko zapisywania wyników


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimization with various wrappers"
    )
    parser.add_argument(
        "-e", type=int, help="Number of optimization iterations", default=5000
    )
    parser.add_argument("-s", type=int, help="Random seed", default=10000)
    args = parser.parse_args()

    np.random.seed(args.s)
    main(args)
