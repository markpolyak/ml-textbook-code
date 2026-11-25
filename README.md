# Код к учебным пособиям по машинному обучению

Ноутбуки по главам и пакет `mlbook` с синтетическими данными, на которых построены примеры
книг. Сами книги — на сайте https://ml.markpolyak.ru/. Все листинги из текста книги входят в ноутбуки дословно, и CI проверяет, что они
выполняются.

## Книга 1. Линейные и вероятностные модели

| Глава | Ноутбук | |
|---|---|---|
| 1. Задача обучения по прецедентам | [ch01_problem](book1/ch01_problem.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch01_problem.ipynb) |
| 2. Данные и признаки | [ch02_data](book1/ch02_data.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch02_data.ipynb) |
| 3. Линейная регрессия и метод наименьших квадратов | [ch03_linreg](book1/ch03_linreg.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch03_linreg.ipynb) |
| 4. Градиентный спуск | [ch04_gd](book1/ch04_gd.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch04_gd.ipynb) |
| 5. Переобучение и валидация | [ch05_overfitting](book1/ch05_overfitting.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch05_overfitting.ipynb) |
| 6. Регуляризация | [ch06_regularization](book1/ch06_regularization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch06_regularization.ipynb) |
| 7. Логистическая регрессия | [ch07_logreg](book1/ch07_logreg.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch07_logreg.ipynb) |
| 8. Метрики качества классификации | [ch08_metrics](book1/ch08_metrics.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch08_metrics.ipynb) |
| 9. Наивный байесовский классификатор | [ch09_naive_bayes](book1/ch09_naive_bayes.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markpolyak/ml-textbook-code/blob/main/book1/ch09_naive_bayes.ipynb) |

В каждом ноутбуке — листинги главы на данных из `mlbook`, примеры, посчитанные в тексте
вручную (те же числа), ответы к расчётным задачам и код основных рисунков (без печатного
оформления).

## Данные: пакет `mlbook`

Все выборки синтетические и воспроизводимые (одинаковый `seed` — одинаковые данные).

| Модуль | Что | Главы |
|---|---|---|
| `mlbook.flats` | квартиры: площадь, район, этаж, год постройки (с пропусками), расстояние до метро, цена | 2, 4, 6 |
| `mlbook.modules` | модули программы: строк кода, сложность, изменений за месяц, тесты; найдут ли ошибку | 1, 7, 8 |
| `mlbook.latency` | время отклика сервиса от нагрузки (квадратичная зависимость с шумом) | 3, 5 |
| `mlbook.examples` | маленькие таблицы из текста глав | 1, 2, 8, 9 |

```python
from mlbook import flats
df = flats.load()          # pandas.DataFrame, 300 квартир
```

## Как запустить

**Colab** — кнопка в таблице; первая ячейка ноутбука ставит `mlbook`.

**Локально** (Python ≥ 3.10):

```bash
pip install -r requirements.txt
pip install -e .
jupyter lab book1/
```

## Лицензия

Код и данные — [MIT](LICENSE). Текст книг на [сайте](https://ml.markpolyak.ru/) распространяется
по другой лицензии — [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.ru).
