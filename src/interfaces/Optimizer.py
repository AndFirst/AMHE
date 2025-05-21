from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Generates new candidate solutions.
        
        Returns:
            solutions: np.ndarray of shape (λ, dim)
            z: np.ndarray of shape (λ, dim), the latent noise vectors
        """
        pass

    @abstractmethod
    def tell(self, solutions: np.ndarray, z: np.ndarray, fitnesses: np.ndarray):
        """Updates internal state based on evaluated solutions."""
        pass