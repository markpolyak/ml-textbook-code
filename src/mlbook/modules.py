"""Синтетические данные о модулях программы (гл. 1, 7, 8 книги 1): найдут ли в модуле ошибку
после выпуска новой версии.

Признаки — как в табл. 1.1: число строк кода, цикломатическая сложность, число изменений
за месяц, есть ли автоматические тесты. Вероятность ошибки растёт с размером, сложностью и
частотой изменений и падает, если тесты есть (логистическая модель). Модули с ошибками —
около четверти выборки.
"""
import numpy as np
import pandas as pd

FEATURES = ["строк", "сложность", "изменений", "тесты"]


def make(n=500, seed=0):
    """Исходные массивы: lines, complexity, changes, tests, defect (0/1) и p — истинная вероятность."""
    rng = np.random.default_rng(seed)
    lines = np.clip(rng.lognormal(np.log(400), 0.8, n), 20, 8000).round().astype(int)
    complexity = np.maximum(1, (lines / 35 * rng.lognormal(0, 0.35, n)).round()).astype(int)
    changes = rng.poisson(rng.gamma(2.0, 2.5, n))
    tests = (rng.uniform(size=n) < 0.6).astype(int)
    h = (-0.8 + 0.8 * np.log(lines / 400) + 0.6 * np.log(complexity * 35 / lines)
         + 0.2 * (changes - 5) - 1.2 * tests)
    p = 1 / (1 + np.exp(-h))
    defect = (rng.uniform(size=n) < p).astype(int)
    return dict(lines=lines, complexity=complexity, changes=changes, tests=tests,
                defect=defect, p=p)


def load(n=500, seed=0):
    """Таблица модулей: признаки FEATURES и ответ «ошибка» (1 — ошибку нашли, 0 — нет).

    Метки 0 и 1, как принято в scikit-learn; в тексте книги классы обозначены −1 и +1.
    """
    d = make(n, seed)
    return pd.DataFrame({
        "строк": d["lines"],
        "сложность": d["complexity"],
        "изменений": d["changes"],
        "тесты": d["tests"],
        "ошибка": d["defect"],
    })
