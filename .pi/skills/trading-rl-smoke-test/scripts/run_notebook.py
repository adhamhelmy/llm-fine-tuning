"""Run a CPU-only notebook's code cells headlessly with Alpaca (and optionally Ollama) stubbed.

    python run_notebook.py unsloth/strategy_tester_ollama.ipynb --ollama \
        --replace "N_SAMPLES      = 20=>N_SAMPLES      = 6"

- Runs from a temporary copy of the repo, so the real tree gets no artifacts
  (successful_strategies/, CSVs, PNGs). Pass --keep to print the path and keep it.
- `%` / `!` magic lines become `pass`, so the notebook runs against the *local* trading_rl.
- plt.show() is a no-op and IPython's display() prints the object type.
- GPU notebooks (unsloth / mlx imports) can't run this way. Lint them with the
  notebook-editing skill's nb_lint.py instead.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))

ap = argparse.ArgumentParser()
ap.add_argument('notebook', help='path relative to the repo root, e.g. unsloth/strategy_benchmark.ipynb')
ap.add_argument('--ollama', action='store_true', help='also stub the Ollama HTTP API')
ap.add_argument('--replace', action='append', default=[], metavar='OLD=>NEW',
                help='literal source substitution (e.g. shrink sample counts); repeatable')
ap.add_argument('--setup', default='', help='extra Python to run before the first cell')
ap.add_argument('--keep', action='store_true', help='keep the temp copy of the repo')
args = ap.parse_args()

work = tempfile.mkdtemp(prefix='nbrun_')
files = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=REPO,
                       capture_output=True, text=True, check=True).stdout.split('\n')
for f in files:
    if f and f != '.env' and os.path.isfile(os.path.join(REPO, f)):
        os.makedirs(os.path.dirname(os.path.join(work, f)) or work, exist_ok=True)
        shutil.copy2(os.path.join(REPO, f), os.path.join(work, f))

nb = json.load(open(os.path.join(work, args.notebook)))
cells = []
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    lines = []
    for line in ''.join(c['source']).split('\n'):
        m = re.match(r'^(\s*)([%!].*)$', line)
        lines.append(f"{m[1]}pass  # {m[2]}" if m else line)
    cells.append('\n'.join(lines))
src = '\n\n'.join(cells).replace('plt.show()', 'pass')
for rep in args.replace:
    old, new = rep.split('=>', 1)
    assert old in src, f"--replace target not found: {old!r}"
    src = src.replace(old, new)

prelude = f"""import sys, matplotlib
matplotlib.use('Agg')
sys.path[:0] = [{work!r}, {HERE!r}]
import stubs
stubs.install_fake_bars()
{'stubs.install_fake_ollama()' if args.ollama else ''}
def display(x): print('[display]', type(x).__name__)
{args.setup}
"""
script = os.path.join(work, '_notebook_run.py')
open(script, 'w').write(prelude + '\n' + src)

nb_dir = os.path.dirname(os.path.join(work, args.notebook))
rc = subprocess.run([sys.executable, '-W', 'ignore', script], cwd=nb_dir).returncode
if args.keep:
    print(f"\n[kept] {work}")
else:
    shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
