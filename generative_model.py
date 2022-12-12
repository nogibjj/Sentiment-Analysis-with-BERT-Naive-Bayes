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
        print(X)
        x_transpose = X.transpose()
        print(X.shape)
        print(self.p.shape)
        print(x_transpose.shape)
        print(self.p)
        print(x_transpose)
        decrement_p = [
            [element if element == 0 else 1 - element for element in row]
            for row in self.p
        ]
        decrement_x = [[row[0][0], row[1]] for row in x_transpose]
        print(decrement_x)
        decrement_x = [
            [element if element == 0 else 1 - element for element in row]
            for row in decrement_x
        ]
        print(decrement_p.shape)
        print(decrement_x.shape)
        return self.p @ x_transpose + decrement_p @ decrement_x

    def draw(self, size: int):
        """Draw samples from distribution."""
        return np.random.binomial(1, self.p, size)


class NaiveBayesClassifier:
    """Naive Bayes classifier."""

    def __init__(self):
        """Initialize."""
        self.distributions: dict[Any, Bernoulli] = dict()

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train."""
        for label in np.unique(y):
            X_class = X[y == label, :]
            dist = Bernoulli(np.mean(X_class, 0))
            self.distributions[label] = dist

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply."""
        ps = {label: dist.prob(X) for label, dist in self.distributions.items()}
        labels = list(ps.keys())
        return np.array([labels[i] for i in np.argmax(ps, 0)])

    def draw(self, size: int):
        """Draw samples from distribution."""
        labels = list(self.distributions.keys())
        Xs = np.zeros((size, self.distributions[labels[0]].p.size))
        ys = np.zeros(size)
        for i in range(size):
            label = np.random.choice(labels)
            Xs[i, :] = self.distributions[label].draw(1)
            ys[i] = label
        return Xs, ys

    def plot(self):
        """Plot."""
        labels = list(self.distributions.keys())
        for label in labels:
            dist = self.distributions[label]
            plt.plot([0, 1], [dist.p, dist.p], label=label)
        plt.legend()
        plt.show()

    def __repr__(self):
        """Representation."""
        return f"NaiveBayesClassifier({self.distributions})"

    def accuracy_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        return np.mean(self.apply(X) == y)
