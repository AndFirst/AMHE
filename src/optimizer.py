from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, f, max_iter=1000):
        """
        Initialize the optimizer with a function to optimize and a maximum number of iterations.

        :param f: The function to optimize.
        :param max_iter: The maximum number of iterations.
        """
        self.f = f
        self.max_iter = max_iter

    def run(self):
        """
        Run the optimization process.

        :return: The result of the optimization.
        """
        for _ in range(self.max_iter):
            self.step()
        return self.results

    @abstractmethod
    def step(self): ...

    @property
    @abstractmethod
    def results(self): ...
