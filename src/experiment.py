import argparse
import time

import cocoex
import cocopp
import numpy as np

from cmaes import CMAES
from maes import MAES
from maes_ipop import MAES_IPOP


def main(args):
    optimizers = {"maes": MAES, "maes_ipop": MAES_IPOP, "cmaes": CMAES}

    suite_name = "bbob"
    budget_multiplier = 20
    initial_sigma = 0.5

    for opt_name in optimizers:
        suite = cocoex.Suite(suite_name, "", "")
        output_folder = opt_name
        observer = cocoex.Observer(suite_name, f"result_folder: {output_folder}")
        repeater = cocoex.ExperimentRepeater(budget_multiplier)
        minimal_print = cocoex.utilities.MiniPrint()
        while not repeater.done():
            for problem in suite:
                if repeater.done(problem):
                    continue
                problem.observe_with(observer)
                problem(problem.dimension * [0])
                repeater.track(problem)

                # Inicjalizacja optymalizatora
                initial_mean = np.zeros(problem.dimension)
                optimizer_class = optimizers[opt_name]
                if opt_name == "maes_ipop":
                    optimizer = optimizer_class(
                        objective_function=problem,
                        initial_mean=initial_mean,
                        initial_sigma=initial_sigma,
                        stagnation_limit=50,
                        pop_increase_factor=2,
                    )
                else:
                    optimizer = optimizer_class(
                        objective_function=problem,
                        initial_mean=initial_mean,
                        initial_sigma=initial_sigma,
                    )

                start_time = time.process_time()
                best_solution, best_fitness, *_ = optimizer.run(max_iterations=args.e)
                end_time = time.process_time()
                time_taken = round(end_time - start_time, 3)

                problem(best_solution)
                repeater.track(problem)
                minimal_print(problem, f"{opt_name} time: {time_taken}s")
                # do pliku


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MAES, MAES_IPOP, and CMAES optimizers"
    )
    parser.add_argument(
        "-e", type=int, help="Number of optmization iterations", default=500
    )
    parser.add_argument("-s", type=int, help="Random seed", default=1)
    args = parser.parse_args()

    # # Ustawienie ziarna losowego
    np.random.seed(args.s)
    main(args)
    cocopp.main(
        [
            "exdata/maes",
            "exdata/maes_ipop",
            "exdata/cmaes",
        ]
    )
