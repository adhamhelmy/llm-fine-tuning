"""Static checks for notebooks: valid JSON, no outputs, no hardcoded keys, pyflakes on code cells.

    python nb_lint.py                     # all unsloth/*.ipynb
    python nb_lint.py path/a.ipynb ...

Magic lines (%pip, !pip) become `pass`. Pyflakes runs through `uvx`. IPython builtins
(display, get_ipython) are not reported as undefined. Exits 1 on errors.
Unused imports are printed as warnings: review them, but some are pre-existing.
"""

import glob
import json
import os
import re
import subprocess
import sys
import tempfile

paths = sys.argv[1:] or sorted(glob.glob('unsloth/*.ipynb'))
IPYTHON_BUILTINS = ('display', 'get_ipython')
SECRET_RE = re.compile(r"(hf_[A-Za-z0-9]{30,}|PK[A-Z0-9]{16,}|(?:KEY|TOKEN|SECRET)\w*\s*=\s*['\"][A-Za-z0-9/+]{20,}['\"])")
errors = 0
tmp = tempfile.mkdtemp(prefix='nblint_')

for path in paths:
    try:
        nb = json.load(open(path))
    except Exception as e:
        print(f"ERROR {path}: invalid JSON ({e})"); errors += 1; continue

    code = []
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])
        if SECRET_RE.search(src):
            print(f"ERROR {path} cell {i}: looks like a hardcoded secret"); errors += 1
        if c['cell_type'] != 'code':
            continue
        if c.get('outputs') or c.get('execution_count'):
            print(f"ERROR {path} cell {i}: has outputs (clear before committing)"); errors += 1
        lines = []
        for line in src.split('\n'):
            m = re.match(r'^(\s*)([%!].*)$', line)
            lines.append(f"{m[1]}pass  # {m[2]}" if m else line)
        code.append(f"# ---- cell {i}\n" + '\n'.join(lines))

    py = os.path.join(tmp, os.path.basename(path) + '.py')
    open(py, 'w').write('\n\n'.join(code))
    res = subprocess.run(['uvx', '-q', 'pyflakes', py], capture_output=True, text=True)
    for line in (res.stdout + res.stderr).splitlines():
        if any(f"undefined name '{b}'" in line for b in IPYTHON_BUILTINS) or not line.strip():
            continue
        # map the concatenated-file line number back to a cell
        m = re.match(r'.*?\.py:(\d+):', line)
        cell = ''
        if m:
            before = open(py).read().split('\n')[:int(m[1])]
            cell = next((l[len('# ---- '):] for l in reversed(before) if l.startswith('# ---- cell ')), '')
        level = 'WARN ' if 'imported but unused' in line or 'redefinition of unused' in line else 'ERROR'
        errors += level == 'ERROR'
        print(f"{level} {path} [{cell}]: {line.split(': ', 1)[-1]}")

print(f"\n{len(paths)} notebook(s), {errors} error(s)")
sys.exit(1 if errors else 0)
