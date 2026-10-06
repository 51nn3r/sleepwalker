"""Собирает Colab-ноутбук из .py-файла с ячейками, размеченными '# %%'.

Правила:
  '# %% [markdown]'  — начало markdown-ячейки; у её строк убирается ведущий '# '.
  '# %%'             — начало ячейки с кодом.
  строки '# !...' в ячейках кода превращаются в '!...' (команды оболочки в Colab, например pip install).

Запуск: python3 build_notebook.py shapley_actor_kbhop.py   -> shapley_actor_kbhop.ipynb
        python3 build_notebook.py sleepwalker.py --preset pro -> sleepwalker_pro.ipynb (строка PRESET = "fast" → "pro")
"""

import json
import sys


def split_cells(lines):
    cells, kind, buf = [], None, []
    for line in lines:
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    return cells


def to_cell(kind, buf):
    while buf and not buf[-1].strip():          # убрать пустые строки в конце ячейки
        buf = buf[:-1]
    while buf and not buf[0].strip():
        buf = buf[1:]
    if kind == "markdown":
        text = [(l[2:] if l.startswith("# ") else l.lstrip("#")) for l in buf]
        return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text]}
    text = [("!" + l[3:] if l.startswith("# !") else l) for l in buf]
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in text]}


def build(src, dst, preset=None, key=None):
    text = open(src, encoding="utf-8").read()
    if preset:
        assert 'os.environ.get("WM_PRESET", "fast")' in text, "в исходнике нет строки PRESET"
        text = text.replace('os.environ.get("WM_PRESET", "fast")', f'os.environ.get("WM_PRESET", "{preset}")', 1)
    if key:                                      # ключ OpenRouter — только в собранный ноутбук (он не в репозитории)
        assert 'OPENROUTER_API_KEY = ""' in text, "в исходнике нет строки OPENROUTER_API_KEY"
        text = text.replace('OPENROUTER_API_KEY = ""', f'OPENROUTER_API_KEY = "{key}"', 1)
    lines = text.splitlines()
    cells = [to_cell(k, b) for k, b in split_cells(lines)]
    for i, c in enumerate(cells):                # у последней строки ячейки не должно быть '\n'
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")
        c["id"] = f"cell-{i}"                    # nbformat 4.5 требует id у каждой ячейки
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
          "cells": cells}
    json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{dst}: {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")


if __name__ == "__main__":
    args = sys.argv[1:]
    preset = None
    if "--preset" in args:
        preset = args[args.index("--preset") + 1]
        args = [a for a in args if a not in ("--preset", preset)]
    src = args[0] if args else "shapley_actor_kbhop.py"
    import os
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and os.path.exists(".env"):                  # ключ из .env (не в git) — в собранный ноутбук
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("OPENROUTER_API_KEY="):
                key = line.strip().split("=", 1)[1].strip().strip("'\"")
    build(src, src[:-3] + (f"_{preset}" if preset else "") + ".ipynb", preset, key)
