from abc import ABC, abstractmethod

import numpy as np


class FabricationModel(ABC):
    @abstractmethod
    def apply(self, design: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, design: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass
