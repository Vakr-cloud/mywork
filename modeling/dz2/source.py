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

    # ---------- Задача 5: ленивая генерация ----------
    def photons(self, E: float):
        """Генератор: по одному фотону (единичный вектор формы (3,)) за обращение.

        Число фотонов n ~ Poisson(E*Y) разыгрывается при первом обращении к генератору;
        после n-го фотона генератор завершается (StopIteration).
        """
        n = self.n_photons(E, fluctuate=True)
        for _ in range(n):
            yield self.random_directions(1)[0]

    def photon_batches(self, E: float, batch_size: int = 100):
        """Генератор: по batch_size фотонов за обращение (массив (batch_size, 3)).

        Последняя порция может быть короче (остаток n % batch_size).
        Память O(batch_size) независимо от энергии события.
        """
        if batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")
        remaining = self.n_photons(E, fluctuate=True)
        while remaining > 0:
            k = min(batch_size, remaining)
            yield self.random_directions(k)
            remaining -= k

    def __repr__(self):
        return f"Source(Y={self.Y:_} фотонов/МэВ)"
