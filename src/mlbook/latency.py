"""Синтетические данные: время отклика сервиса (мс) в зависимости от нагрузки
(тыс. запросов в секунду), гл. 3 и 5 книги 1. Зависимость квадратичная, с шумом."""
import numpy as np
import pandas as pd


def response_time(load):
    """Закономерная часть ответа φ(x), мс."""
    return 20 + 1.6 * np.asarray(load) ** 2


def make(n=12, seed=3, noise=8.0):
    """n равноотстоящих точек нагрузки на [1; 10] — выборка рис. 3.4 (при n = 12, seed = 3)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1, 10, n)
    return x, response_time(x) + rng.normal(0, noise, x.size)


def load(n=200, seed=0, noise=8.0):
    """Таблица со столбцами «нагрузка» и «время»: нагрузка случайна на [1; 10]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 10, n)
    y = response_time(x) + rng.normal(0, noise, n)
    return pd.DataFrame({"нагрузка": x.round(2), "время": y.round(1)})
