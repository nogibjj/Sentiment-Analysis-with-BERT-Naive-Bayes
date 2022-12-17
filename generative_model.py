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
        return np.prod(self.p**X * (1 - self.p) ** (1 - X), 1)

    def draw(self, size: int):
        """Draw samples from distribution."""
        return np.random.binomial(1, self.p, size)


class BernoulliNB:
    """Bernoulli Naive Bayes classifier."""

    def __init__(self):
        """Initialize."""
        self.distributions: dict[Any, Bernoulli] = dict()
        self.xlims = None
        self.ylims = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train."""
        for label in np.unique(y):
            X_class = X[y == label, :]
            dist = Bernoulli(np.mean(X_class, 0))
            self.distributions[label] = dist
        self._xlims = (min(X[:, 0]), max(X[:, 0]))
        self._ylims = (min(X[:, 1]), max(X[:, 1]))

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply."""
        ps = {label: dist.prob(X) for label, dist in self.distributions.items()}
        labels = list(ps.keys())
        return np.array(labels)[np.argmax(np.array(list(ps.values())), 0)]

    def plot(self):
        """Plot."""
        x1 = np.linspace(self._xlims[0], self._xlims[1], 100)
        x2 = np.linspace(self._ylims[0], self._ylims[1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        ps = {label: dist.prob(X) for label, dist in self.distributions.items()}
        labels = list(ps.keys())
        for i, label in enumerate(labels):
            plt.contour(X1, X2, ps[label].reshape(X1.shape), levels=[0.5])
        plt.xlim(self._xlims)
        plt.ylim(self._ylims)

    def draw(self, size: int):
        """Draw samples from distribution."""
        Xs = []
        ys = []
        for label, dist in self.distributions.items():
            Xs.append(dist.draw(size))
            ys.append(np.repeat(label, size))
        return np.vstack(Xs), np.hstack(ys)
