# #!/usr/bin/env python
# """A short and simple example experiment with restarts mainly for reading.

# For code to run in practice, check out the ``example_experiment_complete.py``.

# The code is fully functional but (mainly) emphasises on readability. Hence,
# it produces only rudimentary progress messages and does not provide batch
# distribution or timing prints.

# To apply the code to a different solver, `fmin` must be re-assigned or
# re-defined accordingly, and the below code may need to be edited at the two
# places marked with "### input".

# For example, using `cma.fmin2` instead of `scipy.optimize.fmin` can be done
# like::

#     import cma
#     def fmin(fun, x0, **kwargs):
#         res = cma.fmin2(fun, x0, 2, {'verbose': -9} if not kwargs.get('disp')
#                                     else None)
#         return res[0]

# See also: https://coco-platform.org/getting-started#experiment
# """

# __author__ = "Nikolaus Hansen"
# __copyright__ = "public domain"

# import cocoex  # experimentation module
# import cocopp  # post-processing module (not strictly necessary)
# import scipy  # to define the solver to be benchmarked

# ### input: define suite and solver (see also "input" below where fmin is called)
# suite_name = "bbob"
# fmin = scipy.optimize.fmin  # optimizer to be benchmarked
# budget_multiplier = 1  # increase to 3, 10, 30, ... x dimension

# ### prepare
# suite = cocoex.Suite(
#     suite_name, "", ""
# )  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
# output_folder = "{}_of_{}_{}D_on_{}".format(
#     fmin.__name__, fmin.__module__ or "", int(budget_multiplier + 0.499), suite_name
# )
# observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
# repeater = cocoex.ExperimentRepeater(budget_multiplier)  # x dimension
# minimal_print = cocoex.utilities.MiniPrint()

# ### go
# while not repeater.done():  # while budget is left and successes are few
#     for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
#         if repeater.done(problem):
#             continue
#         problem.observe_with(observer)  # generate data for cocopp
#         problem(problem.dimension * [0])  # for better comparability

#         ### input: the next three lines need to be adapted to the specific fmin
#         xopt = fmin(
#             problem, repeater.initial_solution_proposal(problem), disp=False
#         )  # could depend on budget_multiplier
#         problem(xopt)  # make sure the returned solution is evaluated

#         repeater.track(problem)  # track evaluations and final_target_hit
#         minimal_print(problem)  # show progress

# ### post-process data
# dsl = cocopp.main(
#     observer.result_folder + " bfgs!"
# )  # re-run folders look like "...-001" etc


import shutil

import cocoex
import cocopp
import numpy as np

from cmaes import CMAES  # Twój kod CMAES
from maes_ipop import MAES_IPOP  # Twój kod MAES_IPOP


def run_algorithm_on_suite(algorithm_class, suite_name="bbob", output_folder="results"):
    # Inicjalizacja zestawu testowego BBOB
    suite = cocoex.Suite(suite_name, "", "dimensions:2,10")  # Wybierz wymiary
    observer = cocoex.Observer(suite_name, f"result_folder: {output_folder}")

    for problem in suite:
        # Połącz problem z obserwatorem
        problem.observe_with(observer)

        # Parametry początkowe
        dim = problem.dimension
        initial_mean = np.zeros(dim)  # Punkt początkowy w zerze (możesz dostosować)
        initial_sigma = 0.5  # Początkowy rozmiar kroku
        max_budget = 50 * dim  # Maksymalna liczba ewaluacji (dostosuj)

        # Inicjalizacja algorytmu
        algo = algorithm_class(
            objective_function=problem,  # Funkcja celu z COCO
            initial_mean=initial_mean,
            initial_sigma=initial_sigma,
            dim=dim,
        )

        # Uruchom algorytm
        algo.run(max_iterations=max_budget)


shutil.rmtree("exdata", ignore_errors=True)
shutil.rmtree("ppdata", ignore_errors=True)
# Uruchom MAES_IPOP
maes_result_folder = "results_MAES_IPOP"
run_algorithm_on_suite(MAES_IPOP, suite_name="bbob", output_folder=maes_result_folder)

# Uruchom CMAES
cmaes_result_folder = "results_CMAES"
run_algorithm_on_suite(CMAES, suite_name="bbob", output_folder=cmaes_result_folder)

# ### post-process data
dsl = cocopp.main(
    f"exdata/{maes_result_folder} exdata/{cmaes_result_folder} bfgs!"
)  # re-run folders look like "...-001" etc
