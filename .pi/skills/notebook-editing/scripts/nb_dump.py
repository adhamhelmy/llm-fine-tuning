"""Print notebooks as readable text with ORIGINAL cell indexes (what nbedit.NB uses).

    python nb_dump.py unsloth/strategy_tester.ipynb            # all cells
    python nb_dump.py unsloth/strategy_tester.ipynb 6 9-12     # selected cells / ranges
    python nb_dump.py unsloth/*.ipynb --out /tmp/nbsrc         # one .py per notebook (for grep/diff)
"""

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('paths', nargs='+', help='notebooks, optionally followed by cell indexes / ranges')
ap.add_argument('--out', help='write <out>/<name>.ipynb.py per notebook instead of printing')
args = ap.parse_args()

notebooks = [p for p in args.paths if p.endswith('.ipynb')]
wanted = set()
for tok in (p for p in args.paths if not p.endswith('.ipynb')):
    a, _, b = tok.partition('-')
    wanted.update(range(int(a), int(b or a) + 1))

for path in notebooks:
    cells = json.load(open(path))['cells']
    text = ''.join(
        f"# ---- cell {i} [{c['cell_type']}]" + (f" ({len(c.get('outputs', []))} outputs)" if c.get('outputs') else '')
        + '\n' + ''.join(c['source']) + '\n\n'
        for i, c in enumerate(cells) if not wanted or i in wanted)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, os.path.basename(path) + '.py'), 'w') as f:
            f.write(text)
    else:
        print(f"######## {path} ({len(cells)} cells)\n{text}")
