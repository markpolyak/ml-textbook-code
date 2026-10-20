"""Синтетические данные о квартирах (гл. 2, 4, 6 книги 1). Цены условные, не реальные.

Цена зависит от района, площади и расстояния до метро; этаж и год постройки на цену не влияют
(как в примере гл. 6). Год постройки иногда неизвестен, чаще — у старых домов.
"""
import numpy as np
import pandas as pd

DISTRICTS = ["А", "Б", "В"]
PRICE_M2 = {"А": 0.21, "Б": 0.17, "В": 0.13}   # млн руб. за м², условно


def make(n=300, seed=1):
    """Исходные массивы: district, area (м²), metro (км), price (млн руб.), floor, year.

    year — float, пропуск — nan. Первые четыре массива генерируются первыми, поэтому
    не зависят от того, какие признаки добавлены после них (рисунки книги не меняются).
    """
    rng = np.random.default_rng(seed)
    district = rng.choice(DISTRICTS, size=n, p=[0.3, 0.4, 0.3])
    area = np.clip(rng.lognormal(np.log(50), 0.32, n), 22, 150)
    metro = np.clip(rng.gamma(2.0, 0.6, n), 0.1, 6.0)          # км до метро
    k = np.array([PRICE_M2[d] for d in district])
    price = (k * area * (1 - 0.04 * metro)) * rng.lognormal(0, 0.12, n)
    floors = rng.choice([5, 9, 12, 16, 25], size=n, p=[0.25, 0.3, 0.15, 0.2, 0.1])
    floor = np.floor(rng.uniform(1, floors + 1)).astype(int)
    year = rng.integers(1955, 2025, n).astype(float)
    missing = rng.uniform(size=n) < np.where(year < 1975, 0.15, 0.03)
    year[missing] = np.nan
    return dict(district=district, area=area, metro=metro, price=price, floor=floor, year=year)


def load(n=300, seed=1):
    """Таблица квартир со столбцами «площадь», «район», «этаж», «год», «метро», «цена»."""
    d = make(n, seed)
    return pd.DataFrame({
        "площадь": d["area"].round(1),
        "район": d["district"],
        "этаж": d["floor"],
        "год": d["year"],
        "метро": d["metro"].round(2),
        "цена": d["price"].round(2),
    })
