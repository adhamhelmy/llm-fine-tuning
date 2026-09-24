"""Safe, minimal-diff edits to .ipynb files, addressed by ORIGINAL cell index.

    import sys; sys.path.insert(0, '.pi/skills/notebook-editing/scripts')
    from nbedit import NB
    n = NB('unsloth/strategy_tester.ipynb')
    n.set(6, 'import json', 'new cell source')      # replace a cell (asserts it starts with 'import json')
    n.sub(9, 'old text', 'new text')                # substring replace (asserts exactly one match)
    n.delete(10, 'class Backtrader:')               # delete (asserts prefix)
    n.insert_after(2, 'print(1)')                   # new code cell ('markdown' via cell_type=)
    n.save()

Every edit asserts on the cell's ORIGINAL content, so a stale index fails loudly
instead of corrupting the notebook. Indexes never shift until save().
Output format matches the repo: indent=1, ensure_ascii=False, trailing newline.
"""

import json

SETUP_CELL = '''# Shared backtest / prompt / reward code: the `trading_rl` package in this repo
import os, sys
if os.path.isdir('../trading_rl'):   # local clone: import straight from the repo
    sys.path.insert(0, os.path.abspath('..'))
else:                                # Colab / Kaggle
    %pip install -q "trading-rl @ git+https://github.com/adhamhelmy/llm-fine-tuning.git"'''


def _src(cell):
    return ''.join(cell['source'])


def _set_src(cell, text):
    lines = text.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def save_notebook(path, nb):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


class NB:
    def __init__(self, path):
        self.path = path
        self.nb = json.load(open(path))
        self.orig = list(self.nb['cells'])
        self.drop, self.after = set(), {}

    def src(self, i):
        return _src(self.orig[i])

    def find(self, needle):
        """Original index of the single cell containing `needle`."""
        hits = [i for i, c in enumerate(self.orig) if needle in _src(c)]
        assert len(hits) == 1, f"{self.path}: {needle!r} in cells {hits}"
        return hits[0]

    def check(self, i, startswith):
        s = self.src(i)
        assert s.lstrip().startswith(startswith), f"{self.path} cell {i} starts {s[:60]!r}"
        return self.orig[i]

    def set(self, i, startswith, new):
        _set_src(self.check(i, startswith), new)

    def sub(self, i, old, new, count=1):
        s = self.src(i)
        assert s.count(old) == count, f"{self.path} cell {i}: {old[:60]!r} found {s.count(old)}x"
        _set_src(self.orig[i], s.replace(old, new))

    def delete(self, i, startswith):
        self.check(i, startswith)
        self.drop.add(i)

    def insert_after(self, i, text, cell_type='code'):
        cell = {'cell_type': cell_type, 'metadata': {}, 'source': []}
        if cell_type == 'code':
            cell.update(outputs=[], execution_count=None)
        _set_src(cell, text)
        self.after.setdefault(i, []).append(cell)

    def save(self):
        out = []
        for i, c in enumerate(self.orig):
            if i not in self.drop:
                out.append(c)
            out.extend(self.after.get(i, []))
        self.nb['cells'] = out
        save_notebook(self.path, self.nb)
        print(f"{self.path}: {len(self.orig)} -> {len(out)} cells")
