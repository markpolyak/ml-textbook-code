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


# ---------- расширенная таблица модулей (книга 2) ----------

FEATURES_EXT = ["строк", "сложность", "изменений", "покрытие", "авторов", "возраст", "язык", "команда"]
LANGUAGES = ["Python", "Java", "C++", "Go", "JavaScript"]
N_TEAMS = 30


def make_ext(n=5000, seed=0):
    """Расширенные модули для книги 2: те же размер, сложность и изменения, что в make, и
    покрытие тестами (%; у части модулей неизвестно), число авторов за квартал, возраст (мес.),
    язык и команда (30 значений). В отличие от make, зависимость нелинейна: пороги по плотности
    ветвлений и числу авторов, частые изменения опасны только при низком покрытии, новые модули
    ошибаются чаще, у команд и языков свои поправки. Возвращает словарь массивов и p — истинную
    вероятность ошибки."""
    rng = np.random.default_rng(seed)
    lines = np.clip(rng.lognormal(np.log(400), 0.8, n), 20, 8000).round().astype(int)
    complexity = np.maximum(1, (lines / 35 * rng.lognormal(0, 0.35, n)).round()).astype(int)
    changes = rng.poisson(rng.gamma(2.0, 2.5, n))
    tested = rng.uniform(size=n) < 0.65
    coverage = np.where(tested, np.clip(rng.beta(4, 2, n) * 100, 5, 100), 0).round()
    authors = 1 + rng.poisson(np.clip(1 + changes / 3, 1, None))
    age = np.clip(rng.exponential(30, n), 0.5, 180).round(1)
    lang = rng.choice(len(LANGUAGES), n, p=[0.3, 0.25, 0.15, 0.1, 0.2])
    team = rng.integers(0, N_TEAMS, n)
    team_eff = rng.normal(0, 0.5, N_TEAMS)
    lang_eff = np.array([0.0, -0.2, 0.6, -0.5, 0.3])
    density = complexity * 35 / lines                    # 1 — обычная плотность ветвлений
    h = (-3.7 + 0.3 * np.log(lines / 400)
         + 3.0 * (density > 1.4) * (lines > 300)          # запутанный код опасен в больших модулях
         + 0.4 * np.minimum(changes, 12) * (coverage < 30)  # частые изменения без тестов
         + 2.5 * (authors >= 5)                           # много авторов
         + 2.5 * np.exp(-age / 4) + 1.5 * (age > 120)     # новые и очень старые модули
         + team_eff[team] + lang_eff[lang])
    p = 1 / (1 + np.exp(-h))
    defect = (rng.uniform(size=n) < p).astype(int)
    cov_known = np.where(rng.uniform(size=n) < 0.12, np.nan, coverage)   # покрытие не измеряли
    return dict(lines=lines, complexity=complexity, changes=changes, coverage=cov_known,
                authors=authors, age=age, lang=np.array(LANGUAGES)[lang],
                team=np.array([f"T{t + 1:02d}" for t in team]), defect=defect, p=p)


def load_ext(n=5000, seed=0):
    """Таблица расширенных модулей: признаки FEATURES_EXT (язык и команда — строки,
    в покрытии есть пропуски NaN) и ответ «ошибка» (0/1)."""
    d = make_ext(n, seed)
    return pd.DataFrame({
        "строк": d["lines"], "сложность": d["complexity"], "изменений": d["changes"],
        "покрытие": d["coverage"], "авторов": d["authors"], "возраст": d["age"],
        "язык": d["lang"], "команда": d["team"], "ошибка": d["defect"],
    })
