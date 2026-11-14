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
    pip install -e ".[train]" vllm      # vllm: FAST_INFERENCE (or set it to False)
    python scripts/main.py train
"""

import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO)

from trading_rl import LadderConfig, RewardConfig

# ── SETTINGS ───────────────────────────────────────────────────────────────────

# (backend, model, lora adapter or None). Backends: 'ollama', 'mlx', 'unsloth' (CUDA).
MODELS = [
    # ('ollama', 'qwen3.8:27b-mlx', None),
    ('ollama', 'ornith-1.5:9b', None),
    # Not downloaded yet (ollama pull <name> first; ~60 GB together):
    # ('ollama', 'hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M', None),   # 36B MoE, 3B active, ~22 GB
    # ('ollama', 'hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M', None),     # 36B MoE, 3B active, ~22 GB
    # ('ollama', 'hf.co/bartowski/FrogNano-4B-2609-GGUF:Q4_K_M', None),     # 4.6B dense, ~3 GB
    # ('ollama', 'gpt-oss:20b', None),                                       # 21B MoE, ~13 GB
    # ('mlx', 'mlx_models/qwen2.5-coder-7b', 'mlx_adapters/grpo'),
    # ('unsloth', 'unsloth/Ministral-3-3B-Instruct-2512', None),
]

# Ladder configs to evaluate. LadderConfig() is the original training prompt.
CONFIGS = [
    LadderConfig(api_reference=True, repair_rounds=3),
]

SPLIT     = 'test'      # 'dev' while experimenting; 'test' only for final numbers
SYMBOLS   = 'eval'     # 'eval' (DOW30 minus bad-data symbols), 'dow30', 'mag7', 'training', or 'AAPL,MSFT'
N_SAMPLES = 1          # strategies per (config, symbol)
OUT_DIR   = os.path.join(REPO, 'ladder_results')

# Reward for training AND evaluation (keep them equal). RewardConfig() = avg annual return with
# 'low' costs (1 bp commission + 2 bps slippage per side); RewardConfig.original() = the
# frictionless reward of the first runs. Other metrics: 'cagr', 'sharpe', 'calmar',
# 'excess_cagr', 'alpha', or any fn(result) -> float. See trading_rl/rewards.py.
# RewardConfig.alpha() = Jensen's alpha vs buy-and-hold of the same symbol (the final GRPO run).
REWARD = RewardConfig.alpha()
WORKERS = None         # backtest subprocesses (None: cpu_count - 1, max 8; 0: in-process)

TEMPERATURE    = 1.0
MAX_NEW_TOKENS = 1536  # evaluation generation budget (thinking tokens count against it too)
THINK          = None  # Ollama thinking models: None (model default), False, True, or
                       # 'low'/'medium'/'high'. Thinking tokens are slow and share MAX_NEW_TOKENS.
# Guards against thinking spirals (Ollama). A call that is cut off returns whatever answer it has
# (usually none), which scores as a failure and, with repair_rounds, triggers a repair.
CALL_TIMEOUT   = 600   # seconds per LLM call (wall clock)
LOOP_GUARD     = True  # cut off a call whose thinking/answer starts repeating itself

# GRPO lengths (tokens). Measured prompts (Qwen tokenizer): baseline ~320, + API reference ~1.4k,
# full context stack ~2k. train() refuses prompts longer than TRAIN_MAX_PROMPT (TRL would cut
# them from the left). Completions get TRAIN_MAX_SEQ - TRAIN_MAX_PROMPT (~2k tokens).
TRAIN_MAX_SEQ    = 5120
TRAIN_MAX_PROMPT = 3072
# vLLM generation during GRPO (pip install vllm). Several times faster than HF generate; lower
# TRAIN_GPU_MEMORY (share of GPU memory vLLM reserves) if training runs out of memory.
FAST_INFERENCE   = True
TRAIN_GPU_MEMORY = 0.6
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


def _repeating(text, window=200, times=3):
    """True when the last `window` characters already appear `times` times: a model stuck
    repeating the same passage (a thinking spiral)."""
    return len(text) >= window * times and text.count(text[-window:]) >= times


def load_model(model, stream=False):
    """Return generate(prompt) -> str for one model. stream=True prints tokens as they arrive."""
    backend, name, adapter = model

    if backend == 'ollama':
        import json
        import requests

        def generate(prompt):
            # Always streamed from Ollama, so a runaway call can be cut off mid-generation
            # (closing the connection makes Ollama stop). `stream` only controls printing.
            t0, content, thinking, in_think, reason = time.time(), [], [], False, None
            with requests.post(f'{OLLAMA_URL}/api/chat', timeout=(10, 120), stream=True, json={
                    'model': name, 'stream': True,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'options': {'temperature': TEMPERATURE, 'num_predict': MAX_NEW_TOKENS, 'num_ctx': 8192},
                    **({} if THINK is None else {'think': THINK})}) as r:
                r.raise_for_status()
                for i, line in enumerate(r.iter_lines()):
                    if not line:
                        continue
                    chunk = json.loads(line)
                    msg = chunk.get('message', {})
                    if msg.get('thinking'):          # reasoning tokens (not returned, only shown)
                        thinking.append(msg['thinking'])
                        if stream:
                            print(('\n<thinking>\n' if not in_think else '') + msg['thinking'], end='', flush=True)
                        in_think = True
                    if msg.get('content'):
                        content.append(msg['content'])
                        if stream:
                            print(('\n</thinking>\n' if in_think else '') + msg['content'], end='', flush=True)
                        in_think = False
                    if chunk.get('done'):
                        reason = chunk.get('done_reason')
                        break
                    if time.time() - t0 > CALL_TIMEOUT:
                        reason = 'timeout'
                        break
                    if LOOP_GUARD and i % 50 == 49 and (_repeating(''.join(thinking)) or _repeating(''.join(content))):
                        reason = 'loop'
                        break
            if stream:
                print(flush=True)
            if reason in ('timeout', 'loop') or (reason == 'length' and not content):
                why = {'timeout': f'over CALL_TIMEOUT={CALL_TIMEOUT}s', 'loop': 'output started repeating',
                       'length': f'hit MAX_NEW_TOKENS={MAX_NEW_TOKENS} before answering'}[reason]
                print(f"  [ollama] call cut off: {why} ({time.time() - t0:.0f}s, "
                      f"thinking {len(''.join(thinking))} chars, answer {len(''.join(content))} chars)", flush=True)
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
    from trading_rl.dataset import build_dataset, train_windows

    backend, name, adapter = MODELS[0]
    assert backend == 'unsloth', "training needs an 'unsloth' model"
    backtester = Backtester(verbose=False)
    windows = train_windows(TRAIN_START, TRAIN_END)    # 2016-19 + its halves; full window first
    for s, e in windows:
        backtester.load_bars(TRAINING_SYMBOLS, s, e)
    dataset = build_dataset(TRAINING_SYMBOLS, prompt_fn=prompt_fn(CONFIGS[0], backtester),
                            windows=windows, shuffle_seed=0)

    model = Unsloth(model_name=name, max_seq_length=TRAIN_MAX_SEQ, max_prompt_length=TRAIN_MAX_PROMPT,
                    lora_adapter_path=adapter, fast_inference=FAST_INFERENCE,
                    gpu_memory_utilization=TRAIN_GPU_MEMORY)
    steps = 200
    # Dataset rows carry symbol/start/end columns, so each completion is scored on its own
    # window (a TRL reward call holds generation_batch_size completions from several prompts).
    model.train(
        steps=steps,
        reward_functions=[make_strategy_reward(backtester, reward=REWARD, workers=WORKERS,
                                               save_dir='successful_strategies')],
        dataset=dataset,
        hub_model_id=model.hf_repo_name(steps),
        hf_token=os.environ.get('HF_TOKEN'),
        hub_strategy='checkpoint', save_steps=25,
        # 8 completions per prompt (G=2 turns every reward gap into +-0.707); batch scaling
        # keeps the size of the reward gap
        num_generations=8, gradient_accumulation_steps=8, scale_rewards='batch',
    )


ACTIONS = {'try': try_model, 'test': test_model, 'compare': compare_models, 'train': train_model}

if __name__ == '__main__':
    load_env()
    action = sys.argv[1] if len(sys.argv) > 1 else 'try'
    if action not in ACTIONS:
        sys.exit(f"usage: python scripts/main.py [{'|'.join(ACTIONS)}]")
    ACTIONS[action]()
