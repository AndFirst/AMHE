from abc import ABC, abstractmethod


class OptimizerWrapper(ABC):
    @abstractmethod
    def run(self, max_iterations: int = 1000, fitness_threshold: float = -float('inf')) -> tuple:
        """Runs the optimization process.

        Returns:
            best_solution: the best solution found
            best_fitness: the fitness of the best solution
            restart_count: number of restarts performed
        """
        pass
