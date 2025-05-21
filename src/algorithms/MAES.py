import numpy as np

from interfaces.optimizer import Optimizer

class MAES(Optimizer):
    def __init__(
        self,
        initial_mean,
        initial_sigma,
        population_size=None,
        dim=None,
    ):
        """Initialize MA-ES algorithm.

        Args:
            initial_mean: Initial mean vector (numpy array)
            initial_sigma: Initial step size (float)
            population_size: Number of samples per generation (optional)
            dim: Problem dimension (optional, inferred from initial_mean if not provided)
        """
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

        # Evolution path
        self.ps = np.zeros(self.dim)

        # Transformation matrix
        self.M = np.eye(self.dim)

        self.iteration = 0
        self.best_solution = None
        self.best_fitness = np.inf

    def ask(self):
        """Generate new population of solutions."""
        # Sample population
        pop = np.zeros((self.lambda_, self.dim))
        z = np.random.normal(0, 1, (self.lambda_, self.dim))
        for i in range(self.lambda_):
            d = np.dot(self.M, z[i])
            pop[i] = self.mean + self.sigma * d

        return pop, z

    def tell(self, solutions, z, fitnesses):
        """Update MA-ES parameters based on evaluated solutions."""
        # Sort solutions by fitness
        indices = np.argsort(fitnesses)
        solutions = solutions[indices]
        z = z[indices]
        fitnesses = fitnesses[indices]

        # Update best solution
        if fitnesses[0] < self.best_fitness:
            self.best_fitness = fitnesses[0]
            self.best_solution = solutions[0].copy()

        # Update mean
        d = np.array([np.dot(self.M, z[i]) for i in range(self.lambda_)])
        self.mean += self.sigma * np.sum(
            self.weights[i] * d[indices[i]] for i in range(self.mu)
        )

        # Update evolution path
        zz = np.sum(self.weights[i] * z[indices[i]] for i in range(self.mu))
        c = np.sqrt(self.cs * (2 - self.cs) * self.mu_eff)
        self.ps = (1 - self.cs) * self.ps + c * zz

        # Update transformation matrix
        one = np.eye(self.dim)
        part1 = one
        part2 = self.c1 / 2 * (np.outer(self.ps, self.ps) - one)
        part3 = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            part3 += self.weights[i] * np.outer(z[indices[i]], z[indices[i]])
        part3 = self.cmu / 2 * (part3 - one)
        self.M = np.dot(self.M, part1 + part2 + part3)

        # Update step size
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1)
        )

        self.iteration += 1