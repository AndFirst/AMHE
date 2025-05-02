import numpy as np


class CMAES:
    def __init__(
        self,
        objective_function,
        initial_mean,
        initial_sigma,
        population_size=None,
        dim=None,
    ):
        """Initialize CMA-ES algorithm.

        Args:
            objective_function: Function to minimize, takes numpy array as input
            initial_mean: Initial mean vector (numpy array)
            initial_sigma: Initial step size (float)
            population_size: Number of samples per generation (optional)
            dim: Problem dimension (optional, inferred from initial_mean if not provided)
        """
        self.f = objective_function
        self.mean = np.array(initial_mean)
        self.sigma = initial_sigma
        self.dim = dim if dim is not None else len(initial_mean)
        self.lambda_ = (
            population_size
            if population_size is not None
            else 4 + int(3 * np.log(self.dim))
        )

        # Strategy parameters
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)

        # Learning rates
        self.cc = (4 + self.mu_eff / self.dim) / (
            self.dim + 4 + 2 * self.mu_eff / self.dim
        )
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2
            * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((self.dim + 2) ** 2 + self.mu_eff),
        )
        self.damps = (
            1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        )

        # Evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)

        # Covariance matrix and its square root
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)

        self.iteration = 0
        self.best_solution = None
        self.best_fitness = np.inf

    def ask(self):
        """Generate new population of solutions."""
        # Sample population
        z = np.random.randn(self.lambda_, self.dim)
        y = z @ np.diag(self.D) @ self.B.T
        samples = self.mean + self.sigma * y

        return samples

    def tell(self, solutions, fitnesses):
        """Update CMA-ES parameters based on evaluated solutions."""
        # Sort solutions by fitness
        indices = np.argsort(fitnesses)
        solutions = solutions[indices]
        fitnesses = fitnesses[indices]

        # Update best solution
        if fitnesses[0] < self.best_fitness:
            self.best_fitness = fitnesses[0]
            self.best_solution = solutions[0].copy()

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(solutions[: self.mu] * self.weights[:, np.newaxis], axis=0)

        # Update evolution paths
        y_w = (
            np.sum(
                (solutions[: self.mu] - old_mean) * self.weights[:, np.newaxis], axis=0
            )
            / self.sigma
        )
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mu_eff
        ) * (y_w @ self.B @ np.diag(1 / self.D))
        hsig = np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - self.cs) ** (2 * self.iteration)
        ) < 1.4 + 2 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(
            self.cc * (2 - self.cc) * self.mu_eff
        ) * y_w

        # Update covariance matrix
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.sum(
            [
                self.weights[i]
                * np.outer(
                    (solutions[i] - old_mean) / self.sigma,
                    (solutions[i] - old_mean) / self.sigma,
                )
                for i in range(self.mu)
            ],
            axis=0,
        )
        self.C = (
            (1 - self.c1 - self.cmu) * self.C + self.c1 * rank_one + self.cmu * rank_mu
        )

        # Update step size
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1)
        )

        # Update B and D
        try:
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.abs(self.D))
            self.B = self.B[:, np.argsort(self.D)]
            self.D = self.D[np.argsort(self.D)]
        except np.linalg.LinAlgError:
            pass  # Keep old B and D if decomposition fails

        self.iteration += 1

    def run(self, max_iterations=1000, fitness_threshold=-np.inf):
        """Run optimization until max iterations or fitness threshold is reached."""
        for _ in range(max_iterations):
            solutions = self.ask()
            fitnesses = np.array([self.f(s) for s in solutions])
            self.tell(solutions, fitnesses)

            if self.best_fitness <= fitness_threshold:
                break

        return self.best_solution, self.best_fitness
