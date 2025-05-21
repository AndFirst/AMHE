import numpy as np

class ThresholdIPOP:
    def __init__(self, optimizer_class, objective_function,
                 initial_mean, initial_sigma,
                 mean_range=None, sigma_range=None,
                 improvement_threshold=1e-6, 
                 evaluation_window=10,
                 pop_increase_factor=2,
                 population_size=None, dim=None):
        """
        IPOP restart when average improvement in fitness over a window falls below threshold.

        Args:
            optimizer_class: reference to MAES_IPOP class
            objective_function: function to optimize
            initial_mean: starting point for optimization
            initial_sigma: starting sigma value
            mean_range: tuple (low, high) for sampling new initial_mean on restart
            sigma_range: tuple (low, high) for sampling new sigma on restart
            improvement_threshold: minimum avg improvement to avoid restart
            evaluation_window: number of recent iterations to evaluate improvement
            pop_increase_factor: factor by which to increase population size
            population_size: initial population size
            dim: dimensionality of the problem
        """
        self.optimizer_class = optimizer_class
        self.f = objective_function
        self.initial_mean = np.array(initial_mean)
        self.initial_sigma = initial_sigma
        self.mean_range = mean_range
        self.sigma_range = sigma_range
        self.improvement_threshold = improvement_threshold
        self.evaluation_window = evaluation_window
        self.pop_increase_factor = pop_increase_factor
        self.population_size = population_size
        self.dim = dim if dim is not None else len(initial_mean)

        self.population_multiplier = 1
        self.restart_count = 0
        self.recent_fitness = []

        self._init_optimizer()

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
        self.recent_fitness = []

    def run(self, max_iterations=1000, fitness_threshold=-np.inf):
        for _ in range(max_iterations):
            solutions, z = self.optimizer.ask()
            fitnesses = np.array([self.optimizer.f(s) for s in solutions])
            self.optimizer.tell(solutions, z, fitnesses)

            self.recent_fitness.append(self.optimizer.best_fitness)
            if len(self.recent_fitness) > self.evaluation_window:
                self.recent_fitness.pop(0)
                improvements = np.diff(self.recent_fitness)
                avg_improvement = -np.mean(improvements)  # negative because we minimize
                if avg_improvement < self.improvement_threshold:
                    self.population_multiplier *= self.pop_increase_factor
                    self.restart_count += 1
                    self._init_optimizer()

            if self.optimizer.best_fitness <= fitness_threshold:
                break

        return self.optimizer.best_solution, self.optimizer.best_fitness, self.restart_count