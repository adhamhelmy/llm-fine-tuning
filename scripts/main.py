"""Try, test, compare or train models in the ladder framework.

Edit the SETTINGS below, then run one of:

    python scripts/main.py try       # 1 strategy from MODELS[0], printed in full
    python scripts/main.py test      # MODELS[0] on every config in CONFIGS, with a report
    python scripts/main.py compare   # every model in MODELS on the same CONFIGS, side by side
    python scripts/main.py train     # GRPO on MODELS[0] (CUDA only)

Alpaca keys (and HF_TOKEN for training) are read from the environment or from .env.

Copy-paste commands (run from the repo root):

    # 1. check nothing is broken (offline, synthetic data, free)
    python .pi/skills/trading-rl-smoke-test/scripts/smoke.py

    # 2. one strategy, to check the setup end to end
    python scripts/main.py try

    # 3. dev-split ladder run (edit CONFIGS below first)
    python scripts/main.py test

    # 4. final numbers: set SPLIT = 'test' below, then
    python scripts/main.py test

    # 5. base vs GRPO adapter: list both in MODELS below, then
    python scripts/main.py compare

    # 6. robustness report on a results CSV
    python -c "from trading_rl import Backtester; from trading_rl.robustness import full_report; full_report('ladder_results/qwen3.8-27b-mlx/eval_test.csv', Backtester(), out_dir='ladder_results/qwen3.8-27b-mlx/robustness')"

    # GRPO training (CUDA machine only)
    pip install -e ".[train]"
    python scripts/main.py train
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO)

from trading_rl import LadderConfig, RewardConfig

# ── SETTINGS ───────────────────────────────────────────────────────────────────

# (backend, model, lora adapter or None). Backends: 'ollama', 'mlx', 'unsloth' (CUDA).
MODELS = [
    ('ollama', 'qwen3.8:27b-mlx', None),
    # ('mlx', 'mlx_models/qwen2.5-coder-7b', 'mlx_adapters/grpo'),
    # ('unsloth', 'unsloth/Ministral-3-3B-Instruct-2512', None),
]

# Ladder configs to evaluate. LadderConfig() is the original training prompt.
CONFIGS = [
    LadderConfig(),
    LadderConfig(api_reference=True),
    LadderConfig(api_reference=True, repair_rounds=3),
]

SPLIT     = 'dev'      # 'dev' while experimenting; 'test' only for final numbers
SYMBOLS   = 'dow30'    # 'dow30', 'mag7', 'training', or 'AAPL,MSFT'
N_SAMPLES = 1          # strategies per (config, symbol)
OUT_DIR   = os.path.join(REPO, 'ladder_results')

# Reward for training AND evaluation (keep them equal). RewardConfig() = avg annual return with
# 'low' costs (1 bp commission + 2 bps slippage per side); RewardConfig.original() = the
# frictionless reward of the first runs. Other metrics: 'cagr', 'sharpe', 'calmar',
# 'excess_cagr', or any fn(result) -> float. See trading_rl/rewards.py.
REWARD = RewardConfig()
WORKERS = None         # backtest subprocesses (None: cpu_count - 1, max 8; 0: in-process)

TEMPERATURE    = 1.0
MAX_NEW_TOKENS = 1536
OLLAMA_URL     = 'http://localhost:11434'

# ───────────────────────────────────────────────────────────────────────────────


def load_env():
    """Read KEY=VALUE lines from .env into os.environ (existing variables win)."""
    path = os.path.join(REPO, '.env')
    if os.path.exists(path):
        for line in open(path):
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ.setdefault(key, value.strip('"\''))


def model_label(model):
    backend, name, adapter = model
    return name.split('/')[-1].replace(':', '-') + ('+lora' if adapter else '')


def load_model(model, stream=False):
    """Return generate(prompt) -> str for one model. stream=True prints tokens as they arrive."""
    backend, name, adapter = model

    if backend == 'ollama':
        import json
        import requests

        def generate(prompt):
            r = requests.post(f'{OLLAMA_URL}/api/chat', timeout=600, stream=stream, json={
                'model': name, 'stream': stream,
                'messages': [{'role': 'user', 'content': prompt}],
                'options': {'temperature': TEMPERATURE, 'num_predict': MAX_NEW_TOKENS, 'num_ctx': 8192}})
            r.raise_for_status()
            if not stream:
                return r.json()['message']['content']
            content, thinking = [], False
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line).get('message', {})
                if msg.get('thinking'):          # reasoning tokens (not returned, only shown)
                    if not thinking:
                        print('\n<thinking>', flush=True)
                        thinking = True
                    print(msg['thinking'], end='', flush=True)
                if msg.get('content'):
                    if thinking:
                        print('\n</thinking>\n', flush=True)
                        thinking = False
                    print(msg['content'], end='', flush=True)
                    content.append(msg['content'])
            print(flush=True)
            return ''.join(content)
        return generate

    if backend == 'mlx':
        from mlx_lm import generate as mlx_generate, load as mlx_load
        from mlx_lm.sample_utils import make_sampler
        mlx_model, tokenizer = mlx_load(name, adapter_path=adapter)

        def generate(prompt):
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                 tokenize=False, add_generation_prompt=True)
            return mlx_generate(mlx_model, tokenizer, prompt=text, max_tokens=MAX_NEW_TOKENS,
                                sampler=make_sampler(temp=TEMPERATURE), verbose=False)
        return generate

    if backend == 'unsloth':
        from trading_rl.model import Unsloth
        llm = Unsloth(name, max_seq_length=8192, lora_adapter_path=adapter,
                      adapter_trainable=False, hf_token=os.environ.get('HF_TOKEN'))
        return lambda prompt: llm.generate(prompt, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)

    raise ValueError(f"unknown backend: {backend}")


def split_window():
    from trading_rl import DEV_END, DEV_START, TEST_END, TEST_START
    return {'dev': (DEV_START, DEV_END), 'test': (TEST_START, TEST_END)}[SPLIT]


def make_backtester(symbols, start, end):
    """Backtester with bars for the window and for the feedback window before it."""
    from trading_rl import Backtester
    from trading_rl.ladder import feedback_window
    backtester = Backtester(verbose=False)
    backtester.load_bars(symbols, start, end)
    backtester.load_bars(symbols, *feedback_window(start))
    return backtester


# ── actions ────────────────────────────────────────────────────────────────────

def try_model():
    """Generate one strategy for the first symbol with the last config and print it."""
    from trading_rl import generate_strategy, resolve_symbols
    symbol, (start, end), config = resolve_symbols(SYMBOLS)[0], split_window(), CONFIGS[-1]
    backtester = make_backtester([symbol], start, end)

    print(f"{model_label(MODELS[0])} | {config.name} | {symbol} {start} -> {end}")
    rec = generate_strategy(load_model(MODELS[0], stream=True), config, backtester, symbol, start, end,
                            log=print, reward=REWARD)
    print("\n" + "=" * 30 + " FINAL STRATEGY " + "=" * 30)
    print(rec['strategy_code'] or '(no code)')
    print(f"\nstatus: {rec['status']}  reward: {rec['reward_score']} ({rec['reward']})  "
          f"annual return: {rec['avg_annual_return_pct']}  CAGR: {rec['cagr_pct']}  LLM calls: {rec['llm_calls']}")
    if rec['error']:
        print(f"error: {rec['error']}")


def run(models):
    """Evaluate each model on every config. Results go to a resumable CSV per model."""
    import pandas as pd
    from trading_rl import ScoringPool, baseline_table, resolve_symbols, run_ablation
    symbols, (start, end) = resolve_symbols(SYMBOLS), split_window()
    backtester = make_backtester(symbols, start, end)
    baselines = baseline_table(backtester, symbols, start, end, reward=REWARD)

    results = []
    # one worker is enough here (generation is sequential); it gives a hard backtest timeout
    with ScoringPool(workers=1) as pool:
        for model in models:
            label = model_label(model)
            os.makedirs(f'{OUT_DIR}/{label}', exist_ok=True)
            csv = f'{OUT_DIR}/{label}/eval_{SPLIT}.csv'
            print(f"\n### {label} -> {csv}")
            results.append(run_ablation(load_model(model), backtester, CONFIGS, symbols, start, end,
                                        N_SAMPLES, csv, model=label, reward=REWARD,
                                        pool=pool if WORKERS != 0 else None))
    return pd.concat(results), baselines


def test_model():
    """MODELS[0] on every config."""
    from trading_rl import format_report, report
    results, baselines = run(MODELS[:1])
    print(format_report(report(results, baselines)).to_string(index=False))


def compare_models():
    """All MODELS on every config; one table per config, one row per model."""
    from trading_rl import format_report, report
    results, baselines = run(MODELS)
    for config in CONFIGS:
        print(f"\n=== {config.name} ===")
        rows = results[results['config'] == config.name]
        print(format_report(report(rows, baselines, by='model'), by='model').to_string(index=False))


def train_model():
    """GRPO on MODELS[0] with the prompt + context of CONFIGS[0] (needs CUDA)."""
    from trading_rl.model import Unsloth  # must be imported before trl / transformers
    from trading_rl import (TRAIN_END, TRAIN_START, TRAINING_SYMBOLS, Backtester,
                            make_strategy_reward, prompt_fn)
    from trading_rl.dataset import build_dataset

    backend, name, adapter = MODELS[0]
    assert backend == 'unsloth', "training needs an 'unsloth' model"
    backtester = Backtester(verbose=False)
    backtester.load_bars(TRAINING_SYMBOLS, TRAIN_START, TRAIN_END)
    dataset = build_dataset(TRAINING_SYMBOLS, TRAIN_START, TRAIN_END,
                            prompt_fn=prompt_fn(CONFIGS[0], backtester))

    model = Unsloth(model_name=name, max_seq_length=1024, lora_adapter_path=adapter)
    steps = 100
    # Dataset rows carry symbol/start/end columns, so each completion is scored on its own
    # window (a TRL reward call holds generation_batch_size completions from several prompts).
    model.train(
        steps=steps,
        reward_functions=[make_strategy_reward(backtester, reward=REWARD, workers=WORKERS,
                                               save_dir='successful_strategies')],
        dataset=dataset,
        hub_model_id=model.hf_repo_name(steps),
        hf_token=os.environ.get('HF_TOKEN'),
    )


ACTIONS = {'try': try_model, 'test': test_model, 'compare': compare_models, 'train': train_model}

if __name__ == '__main__':
    load_env()
    action = sys.argv[1] if len(sys.argv) > 1 else 'try'
    if action not in ACTIONS:
        sys.exit(f"usage: python scripts/main.py [{'|'.join(ACTIONS)}]")
    ACTIONS[action]()
