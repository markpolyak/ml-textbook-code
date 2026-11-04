import numpy as np

from mlbook import examples, flats, latency, modules


def test_reproducible():
    for f in (flats.load, modules.load, latency.load):
        assert f().equals(f())


def test_flats():
    df = flats.load()
    assert list(df.columns) == ["площадь", "район", "этаж", "год", "метро", "цена"]
    assert set(df["район"]) == set(flats.DISTRICTS)
    assert 0 < df["год"].isna().mean() < 0.15          # пропуски есть, но их мало
    assert (df["этаж"] >= 1).all() and (df["цена"] > 0).all()


def test_flats_figures_unchanged():
    # рисунки 2.1, 2.2, 2.3, 6.2 книги построены по этим числам
    d = flats.make()
    assert np.isclose(d["area"][:3], [54.388, 38.918, 61.917], atol=1e-3).all()
    assert np.isclose(d["price"].mean(), 8.661, atol=1e-3)


def test_modules():
    df = modules.load()
    assert list(df.columns) == modules.FEATURES + ["ошибка"]
    assert set(df["ошибка"]) == {0, 1}
    assert 0.15 < df["ошибка"].mean() < 0.35


def test_modules_ext():
    df = modules.load_ext()
    assert df.equals(modules.load_ext())
    assert list(df.columns) == modules.FEATURES_EXT + ["ошибка"]
    assert set(df["язык"]) == set(modules.LANGUAGES)
    assert df["команда"].nunique() == modules.N_TEAMS
    assert 0.05 < df["покрытие"].isna().mean() < 0.2      # пропуски только в покрытии
    assert df.drop(columns="покрытие").notna().all().all()
    assert 0.2 < df["ошибка"].mean() < 0.35


def test_examples():
    assert examples.modules5().shape == (5, 5)
    assert examples.flats6()["год"].median() == 2005
    assert examples.scores10()["y"].eq(1).sum() == 4
    texts, y = examples.spam6()
    assert len(texts) == len(y) == 6
