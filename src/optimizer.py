from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def run(self):
        pass
