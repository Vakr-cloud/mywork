"""Изотропный источник света (Задача 2)."""
import numpy as np


class Source:
    """Изотропный точечный источник света в сцинтилляторе.

    Y    : выход фотонов, фотонов/МэВ
    seed : начальное значение генератора случайных чисел
    """

    def __init__(self, Y: float = 10_000, seed: int | None = None):
        if Y <= 0:
            raise ValueError("Выход фотонов Y должен быть положительным")
        self.Y = Y
        self.rng = np.random.default_rng(seed)

    def mean_photons(self, E: float) -> float:
        return self.Y * E

    def n_photons(self, E: float, fluctuate: bool = True) -> int:
        if E < 0:
            raise ValueError("Энергия E не может быть отрицательной")
        mean = self.mean_photons(E)
        return int(self.rng.poisson(mean)) if fluctuate else int(round(mean))

    def random_directions(self, n: int) -> np.ndarray:
        phi = self.rng.uniform(0.0, 2.0 * np.pi, n)
        cos_theta = self.rng.uniform(-1.0, 1.0, n)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        return np.stack([sin_theta * np.cos(phi),
                         sin_theta * np.sin(phi),
                         cos_theta], axis=1)

    def generate(self, E: float, fluctuate: bool = True) -> np.ndarray:
        return self.random_directions(self.n_photons(E, fluctuate))

    def __repr__(self):
        return f"Source(Y={self.Y:_} фотонов/МэВ)"
