import numpy as np

from interfaces.optimizerWrapper import OptimizerWrapper

class OptimizerAlgorithmWrapper(OptimizerWrapper):
    def __init__(self, optimizer_class, objective_function,
                 initial_mean, initial_sigma,
                 mean_range=None, sigma_range=None,
                 population_size=None, dim=None):
        """
        Dummy wrapper for optimizer_class without any restart logic.

        Args:
            optimizer_class: reference to MAES class
            objective_function: function to optimize
            initial_mean: starting point for optimization
            initial_sigma: starting sigma value
            mean_range: (low, high) tuple for mean resampling (optional, not used)
            sigma_range: (low, high) tuple for sigma resampling (optional, not used)
            population_size: size of the population
            dim: dimensionality of the problem
        """
        self.optimizer_class = optimizer_class
        self.f = objective_function
        self.initial_mean = np.array(initial_mean)
        self.initial_sigma = initial_sigma
        self.mean_range = mean_range  # Not used, placeholder for compatibility
        self.sigma_range = sigma_range  # Not used, placeholder for compatibility
        self.population_size = population_size
        self.dim = dim if dim is not None else len(initial_mean)

        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = self.optimizer_class(
            initial_mean=self.initial_mean,
            initial_sigma=self.initial_sigma,
            population_size=self.population_size,
            dim=self.dim
        )

    def run(self, max_iterations=1000, fitness_threshold=-np.inf):
        for _ in range(max_iterations):
            solutions, z = self.optimizer.ask()
            fitnesses = np.array([self.optimizer.f(s) for s in solutions])
            self.optimizer.tell(solutions, z, fitnesses)

            if self.optimizer.best_fitness <= fitness_threshold:
                break

        return self.optimizer.best_solution, self.optimizer.best_fitness, 0