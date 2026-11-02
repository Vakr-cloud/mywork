"""Сферический жидкосцинтилляционный детектор (Задачи 1 и 3)."""
import numpy as np


class Detector:
    """Сферический детектор радиуса R с N ФЭУ на внутренней поверхности.

    Единицы длины — метры.

    Атрибуты
    --------
    R, N, r, a : радиус, число ФЭУ, радиус фотокатода, среднее число
                 фотоэлектронов на один фотон, попавший на фотокатод
    alpha      : угловой радиус шапки на сфере, закрытой одним ФЭУ
    pmt_dirs   : (N, 3) единичные векторы осей ФЭУ
    pmt_pos    : (N, 3) координаты ФЭУ (точки пересечения оси со сферой)
    """

    def __init__(self, R: float = 2.0, N: int = 40, r: float = 0.10,
                 a: float = 0.5, seed: int | None = None):
        if not 0 < r < R:
            raise ValueError("Нужно 0 < r < R")
        if N < 1:
            raise ValueError("Нужно N >= 1")
        if a < 0:
            raise ValueError("Параметр a должен быть неотрицательным")
        self.R, self.N, self.r, self.a = R, N, r, a
        self.alpha = np.arcsin(r / R)
        self.pmt_dirs = self._fibonacci_sphere(N)   # (N, 3)
        self.pmt_pos = R * self.pmt_dirs            # (N, 3)
        self.rng = np.random.default_rng(seed)

    # ---------- геометрия (Задача 1) ----------
    @staticmethod
    def _fibonacci_sphere(N: int) -> np.ndarray:
        """N почти равномерно распределённых единичных векторов (спираль Фибоначчи)."""
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        rho = np.sqrt(1.0 - z**2)
        return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)

    @property
    def coverage(self) -> float:
        """Доля площади сферы, закрытая фотокатодами (аналитически)."""
        return self.N * (1.0 - np.cos(self.alpha)) / 2.0

    def cathode_rims(self, n_seg: int = 32) -> np.ndarray:
        """Точки обода фотокатодов, форма (N, n_seg, 3)."""
        n = self.pmt_dirs
        helper = np.eye(3)[np.argmin(np.abs(n), axis=1)]
        e1 = np.cross(n, helper)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(n, e1)
        centers = self.R * np.cos(self.alpha) * n
        phi = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        return (centers[:, None, :]
                + self.r * (np.cos(phi)[None, :, None] * e1[:, None, :]
                            + np.sin(phi)[None, :, None] * e2[:, None, :]))

    def pmt_index(self, points, tol: float = 1e-6):
        """Номер ФЭУ, которому принадлежит точка на поверхности (или -1)."""
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        if pts.shape[1] != 3:
            raise ValueError("Точки должны иметь форму (3,) или (M, 3)")
        if not np.allclose(np.linalg.norm(pts, axis=1), self.R, rtol=tol, atol=tol):
            raise ValueError("Точка не лежит на поверхности детектора |p| = R")

        cos_ang = pts @ self.pmt_dirs.T / self.R              # (M, N)
        nearest = np.argmax(cos_ang, axis=1)
        best = cos_ang[np.arange(len(pts)), nearest]
        hit = best >= np.cos(self.alpha) - 1e-12
        idx = np.where(hit, nearest, -1)
        return int(idx[0]) if single else idx

    def is_on_pmt(self, points, tol: float = 1e-6):
        """True, если точка принадлежит какому-либо ФЭУ."""
        return self.pmt_index(points, tol) >= 0

    # ---------- распространение и детектирование света (Задача 3) ----------
    def intersect_sphere(self, origin, directions) -> np.ndarray:
        """Точки, в которых лучи origin + t*n (t > 0) выходят на поверхность детектора.

        origin     : (3,) — общий источник, либо (M, 3) — свой для каждого луча
        directions : (M, 3) единичные векторы
        Возвращает массив (M, 3).

        Решаем |o + t n|^2 = R^2:  t^2 + 2 (o.n) t + (|o|^2 - R^2) = 0.
        Источник внутри сферы => свободный член < 0 => ровно один положительный корень
        t = -(o.n) + sqrt((o.n)^2 - (|o|^2 - R^2)).
        """
        o = np.asarray(origin, dtype=float)
        d = np.asarray(directions, dtype=float)
        if np.any(np.sum(o * o, axis=-1) >= self.R**2):
            raise ValueError("Источник должен находиться внутри детектора")
        b = np.sum(o * d, axis=-1)                            # (M,)
        c = np.sum(o * o, axis=-1) - self.R**2                # () или (M,)
        t = -b + np.sqrt(b * b - c)
        return o + t[:, None] * d

    def trace(self, origin, directions) -> np.ndarray:
        """Номер ФЭУ, в который попадает каждый фотон (или -1 — поглотился на стенке).

        Рассеяния и поглощения в сцинтилляторе нет: фотон летит по прямой до границы.
        Обод фотокатода лежит на сфере, а шапка над фотокатодом — выпуклая область,
        которую луч изнутри может покинуть только через фотокатод, поэтому
        «пересёк фотокатод» <=> «вышел на сферу внутри шапки ФЭУ».
        """
        return self.pmt_index(self.intersect_sphere(origin, directions))

    def photoelectrons(self, n_photons=1):
        """Случайное число фотоэлектронов от n_photons фотонов, попавших на фотокатод.

        Один фотон даёт Poisson(a) фотоэлектронов; сумма n независимых Poisson(a)
        есть Poisson(a n), поэтому для массива n_photons (по ФЭУ) розыгрыш векторизован.
        """
        return self.rng.poisson(self.a * np.asarray(n_photons))

    def detect(self, origin, directions):
        """Отклик детектора на набор фотонов.

        Возвращает (n_hits, charge), оба формы (N,):
        число фотонов, попавших на фотокатод каждого ФЭУ, и число фотоэлектронов.
        """
        idx = self.trace(origin, directions)
        n_hits = np.bincount(idx[idx >= 0], minlength=self.N)
        return n_hits, self.photoelectrons(n_hits)

    def detect_stream(self, origin, batches):
        """Отклик детектора на поток порций фотонов (итератор массивов (k, 3)).

        Попадания накапливаются порция за порцией, фотоэлектроны разыгрываются один раз
        в конце: сумма независимых Poisson(a) по попаданиям даёт Poisson(a * n_hits).
        Возвращает (n_hits, charge), как и detect.
        """
        n_hits = np.zeros(self.N, dtype=np.int64)
        for dirs in batches:
            idx = self.trace(origin, dirs)
            n_hits += np.bincount(idx[idx >= 0], minlength=self.N)
        return n_hits, self.photoelectrons(n_hits)

    def __repr__(self):
        return (f"Detector(R={self.R} м, N={self.N}, r={self.r} м, a={self.a}; "
                f"alpha={np.degrees(self.alpha):.3f}°, покрытие={self.coverage:.2%})")
