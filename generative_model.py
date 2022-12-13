"""Generative Model."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


class Bernoulli:
    """Bernoulli distribution."""

    def __init__(self, p: float):
        """Initialize."""
        self.p = p

    def prob(self, X: np.ndarray):
        """Compute probability of data under distribution.

        X is expected to be N-by-ndims.
        """
        return self.p ** X * (1 - self.p) ** (1 - X)

    def draw(self, size: int):
        """Draw samples from distribution."""
        return np.random.binomial(1, self.p, size)
