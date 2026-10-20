"""Проверка ноутбуков.

    python tools/notebooks.py run   [файлы]   — выполнить ноутбуки, ошибка в ячейке — ошибка проверки
    python tools/notebooks.py check [файлы]   — в репозитории ноутбуки хранятся без выходных данных
    python tools/notebooks.py strip [файлы]   — удалить выходные данные (перед коммитом)

Без списка файлов — все book*/*.ipynb. Пакет mlbook берётся из src/ (устанавливать не нужно).
"""
import os
import sys
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
KEEP_META = {"kernelspec", "language_info", "colab"}


def notebooks(args):
    return [Path(a) for a in args] or sorted(ROOT.glob("book*/*.ipynb"))


def is_clean(nb):
    return all(not c.get("outputs") and c.get("execution_count") is None
               for c in nb.cells if c.cell_type == "code")


def strip(path):
    nb = nbformat.read(path, as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
        c.metadata = {}
    nb.metadata = {k: v for k, v in nb.metadata.items() if k in KEEP_META}
    nbformat.write(nb, path)


def run(path):
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if str(ROOT / "src") not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [str(ROOT / "src"), *paths] if p)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()


def main():
    cmd, files = sys.argv[1], notebooks(sys.argv[2:])
    failed = []
    for p in files:
        if cmd == "strip":
            strip(p)
        elif cmd == "check":
            if not is_clean(nbformat.read(p, as_version=4)):
                failed.append(p)
                print(f"{p.name}: есть выходные данные — python tools/notebooks.py strip")
        elif cmd == "run":
            t = time.time()
            try:
                run(p)
                print(f"{p.name}: ок, {time.time() - t:.0f} с")
            except Exception as e:  # noqa: BLE001 — печатаем и идём дальше
                failed.append(p)
                print(f"{p.name}: ОШИБКА\n{e}")
        else:
            sys.exit(__doc__)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
