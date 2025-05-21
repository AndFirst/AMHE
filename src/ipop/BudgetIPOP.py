import numpy as np

from interfaces.optimizerWrapper import OptimizerWrapper

class BudgetIPOP(OptimizerWrapper):
    def __init__(self, optimizer_class, objective_function,
                 initial_mean, initial_sigma,
                 mean_range=None, sigma_range=None,
                 pop_increase_factor=2, population_size=None, dim=None,
                 restart_budget=1000):
        """
        IPOP-style restart when fixed number of fitness evaluations is exceeded (restart_budget).

        Args:
            optimizer_class: reference to MAES_IPOP class
            objective_function: function to optimize
            initial_mean: starting point for optimization
            initial_sigma: starting sigma value
            mean_range: tuple (low, high) for sampling new initial_mean on restart
            sigma_range: tuple (low, high) for sampling new sigma on restart
            pop_increase_factor: factor by which to increase population size on restart
            population_size: initial population size
            dim: dimensionality of the problem
            restart_budget: number of evaluations before restart
        """
        self.optimizer_class = optimizer_class
        self.f = objective_function
        self.initial_mean = np.array(initial_mean)
        self.initial_sigma = initial_sigma
        self.mean_range = mean_range
        self.sigma_range = sigma_range
        self.pop_increase_factor = pop_increase_factor
        self.population_size = population_size
        self.dim = dim if dim is not None else len(initial_mean)
        self.restart_budget = restart_budget

        self.population_multiplier = 1
        self.restart_count = 0

        self._init_optimizer()
        self.eval_counter = 0

    def _sample_new_start(self):
        mean = self.initial_mean
        sigma = self.initial_sigma

        if self.mean_range is not None:
            mean = np.random.uniform(self.mean_range[0], self.mean_range[1], self.dim)

        if self.sigma_range is not None:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

        return mean, sigma

    def _init_optimizer(self):
        mean, sigma = self._sample_new_start()
        pop_size = int(self.population_size * self.population_multiplier) if self.population_size else None

        self.optimizer = self.optimizer_class(
            initial_mean=mean,
            initial_sigma=sigma,
            population_size=pop_size,
            dim=self.dim
        )

    def run(self, max_iterations=1000, fitness_threshold=-np.inf):
        for _ in range(max_iterations):
            solutions, z = self.optimizer.ask()
            fitnesses = np.array([self.optimizer.f(s) for s in solutions])
            self.eval_counter += len(fitnesses)
            self.optimizer.tell(solutions, z, fitnesses)

            if self.optimizer.best_fitness <= fitness_threshold:
                break

            if self.eval_counter >= self.restart_budget:
                self.population_multiplier *= self.pop_increase_factor
                self.restart_count += 1
                self._init_optimizer()
                self.eval_counter = 0

        return self.optimizer.best_solution, self.optimizer.best_fitness, self.restart_count