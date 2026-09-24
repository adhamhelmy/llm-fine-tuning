---
name: notebook-editing
description: Read, edit, clear and lint the Jupyter notebooks in unsloth/*.ipynb with minimal diffs. Use whenever a task involves changing a notebook, porting code between notebooks and trading_rl, or checking notebooks before a commit.
---

# Editing this repo's notebooks

The notebooks are JSON and target Colab, Kaggle, Apple Silicon (MLX) or local Ollama. Don't edit them with the `edit`/`write` tools or regex over raw JSON. Use the scripts below, which are relative to this skill directory and should be run from the repo root.

## Workflow

1. **Read:** `python3 scripts/nb_dump.py unsloth/X.ipynb [6 9-12]` prints the cells with their **original** indexes. Use `--out /tmp/nbsrc` to dump every notebook to `.py` for grep/diff.
2. **Edit** in a Python heredoc with `scripts/nbedit.py`:
   ```python
   import sys; sys.path.insert(0, '.pi/skills/notebook-editing/scripts')
   from nbedit import NB, SETUP_CELL
   n = NB('unsloth/X.ipynb')
   n.sub(n.find('bt_instance = Backtrader()'), 'Backtrader()', 'Backtester(verbose=False)')
   n.set(6, 'import json', 'from trading_rl import Backtester')
   n.save()
   ```
   - Every operation asserts on the original content (a prefix or an exact match count). If an assert fires, re-dump the cell and fix the edit; never loosen the assert.
   - Indexes refer to the original notebook until `save()`. Put all edits to one notebook in a single script, because a second run sees new indexes.
   - Cell sources often lack a trailing newline, so match `"\nfoo"` rather than `"foo\n"` for a last line.
3. **Lint:** run `python3 scripts/nb_lint.py`. It checks JSON validity, cleared outputs, hardcoded keys, and pyflakes undefined names. Fix every ERROR. For WARNs (unused imports), remove the ones your change introduced.
4. **Run** CPU notebooks with the `trading-rl-smoke-test` skill (`run_notebook.py`).

## Repo conventions

- **Write format:** `json.dump(nb, f, indent=1, ensure_ascii=False)` plus a trailing newline. `nbedit.save_notebook` does this; any other format produces a whole-file diff.
- **No outputs or execution counts are committed.** To clear them: `for c in nb['cells']: if c['cell_type']=='code': c['outputs']=[]; c['execution_count']=None`.
- **Secrets are empty placeholders** (`HF_TOKEN = ''`, `ALPACA_API_KEY = ''`, …) that the user fills in at runtime. Don't add secret-loader helpers, and never paste real keys (the repo is public). `strategy_tester.ipynb` uses Colab `userdata`; leave that as is.
- **No inline copies of shared logic.** Backtester, rewards, prompts, code extraction and evaluation come from `trading_rl`. Notebooks keep only their model-specific code, such as a `generate(prompt) -> str` function passed to `evaluate_sample`/`evaluate_symbol`.
- **Every notebook using `trading_rl` has the setup cell** (`nbedit.SETUP_CELL`) right after its installs. It imports `../trading_rl` from a local clone, or pip-installs from GitHub on Colab/Kaggle. So `trading_rl` changes must be pushed before a Colab run sees them; say so to the user.
- **Import order:** in Unsloth notebooks, `from trading_rl.model import Unsloth` (or `from unsloth import …`) must come before trl/transformers imports.
- **Prompts:** comparison notebooks use `make_prompt(..., compact=True)` to stay comparable with old results; training uses the full prompt. Don't switch without asking.
- **Result columns:** `status` (`profitable`/`loss`/`no_trades`/…), `reward_score`, `strategy_code`. Drop `strategy_code` before writing `Summary.csv`.
- Don't merge or delete notebooks without asking. `strategy_tester` vs `model_comparison`, and `strategy_tester_ollama` vs `model_comparison_mlx`, overlap on purpose until the user decides.
