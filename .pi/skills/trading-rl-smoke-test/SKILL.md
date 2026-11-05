---
name: trading-rl-smoke-test
description: Verify changes to the trading_rl package or CPU-only notebooks (Ollama backend) offline, with synthetic Alpaca bars and a fake Ollama instead of real APIs or a GPU. Use after editing backtest/reward/prompt/evaluate code or those notebooks, or before telling the user something works.
---

# trading_rl offline smoke tests

This repo has no test suite, and the real flows need Alpaca keys and sometimes a CUDA GPU. These scripts stub both out. **Never call the real Alpaca API from a test.**

All paths below are relative to this skill directory. Run the commands from the repo root. Python is Homebrew 3.14; macOS has no `timeout` command, so use the bash tool's timeout.

## 1. Package smoke test (run after any `trading_rl/` change)

```bash
python3 scripts/smoke.py
```

It checks:
- every reward status and its reward value, and `<think>` stripping;
- extraction edge cases (trailing prose, several fences, unclosed fence, indentation) and the AST method check;
- the prompt round-trip;
- the TRL reward function: per-row symbols from dataset columns and from prompts, memoisation, and the worker pool;
- `RewardConfig` metrics, costs, CAGR, `n_orders`, clipping and names;
- `ScoringPool`: hard timeout on a C-level loop, a crashed worker, printing code, parity with in-process scoring;
- symbol sets, `bar_issues`, the disk cache, `append_csv` schema growth;
- the timeout on the main thread and in a thread;
- `evaluate_symbol` + `save_strategy`, including `plot_1.png`.

`pytest` (from the repo root) runs this same script via `tests/test_smoke.py`. CI runs it too.

For the ladder it checks:
- the few-shot examples run;
- repair, refine and the analyst → coder graph, using a scripted fake LLM;
- **no test leakage**: a spy on `Backtester.run` asserts that feedback uses only the pre-window and that the test window is backtested once;
- anonymization, `LadderConfig()` == the original prompt, RAG, resumable `run_ablation`, `report` / `paired_delta` / `select_best`, and `dump_prompts`.

It exits 1 if any check fails.

If you change the reward ladder or add a status, update `SAMPLE_OUTPUTS` in `scripts/stubs.py` and the checks in `scripts/smoke.py`.

## 2. Run a CPU notebook end to end

- It runs from a temp copy of the repo, so no artifacts land in the real tree.
- Magics are dropped, so the notebook imports the **local** `trading_rl`, not the GitHub version.
- `--replace OLD=>NEW` asserts that `OLD` exists. Use it to shrink sample counts.

Example (about 1 min with a tiny config):

```bash
python3 scripts/run_notebook.py path/to/notebook.ipynb --ollama \
  --replace "N_SAMPLES = 2=>N_SAMPLES = 1"
```

## What can't be tested here

`trading_rl/model.py` and `scripts/main.py` need unsloth/CUDA. Check them statically with `ruff check .`, and tell the user they were not executed.

## Stub details

- `stubs.install_fake_bars()` patches `Backtester._fetch_bars` to return a deterministic random walk per symbol. It also sets `Backtester.DEFAULT_CACHE_DIR = None` so synthetic bars never reach the disk cache, so call it before creating Backtesters.
- You can also put a custom frame in `Backtester._cache[Backtester.cache_key(symbol, start, end)]`. Bars need `open/high/low/close/volume` columns and a tz-aware index.
- Synthetic prices are not realistic, so check statuses and plumbing, not return values.
