"""Show what a TRL GRPO reward function receives per call (CPU, tiny model, ~1 min).

Reproduces the batching bug fixed in rewards.make_strategy_reward: with per-device batch 1 x
gradient accumulation 8 and num_generations 2, one reward call holds 8 completions from 4
different prompts, so reading the symbol from prompts[0] scores 6/8 completions on the wrong stock.

    pip install torch "trl==0.22.2" "transformers<4.57" datasets accelerate
    python scripts/trace_grpo_batches.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from trading_rl.dataset import build_rows  # noqa: E402
from trading_rl.prompt import extract_trading_parameters  # noqa: E402

SYMS = ['AAPL', 'MSFT', 'JPM', 'KO', 'TSLA', 'NVDA', 'GS', 'WMT']
calls = 0


def logging_reward(completions, prompts, **kwargs):
    global calls
    calls += 1
    own = [extract_trading_parameters(p[-1]['content'])[0] for p in prompts]
    old = own[0]   # the old reward backtested every completion on prompts[0]'s symbol
    print(f"\n=== reward call {calls}: {len(completions)} completions, {len(set(own))} distinct prompts")
    print(f"{'i':>2}  {'prompt symbol':13} {'kwargs symbol':13} {'OLD reward scored on':20} wrong?")
    for i, s in enumerate(own):
        print(f"{i:>2}  {s:13} {kwargs['symbol'][i]:13} {old:20} {'WRONG' if s != old else ''}")
    print(f"-> old reward: {sum(s != old for s in own)}/{len(own)} completions scored on the wrong symbol")
    return [0.0] * len(completions)


if __name__ == '__main__':
    args = GRPOConfig(output_dir='/tmp/grpo_trace', per_device_train_batch_size=1,
                      gradient_accumulation_steps=8, num_generations=2, max_steps=2,
                      max_prompt_length=512, max_completion_length=8, logging_steps=1,
                      report_to='none', use_cpu=True, save_strategy='no', seed=0)
    print(f"generation_batch_size={args.generation_batch_size} "
          f"steps_per_generation={args.steps_per_generation} "
          f"remove_unused_columns={args.remove_unused_columns}")
    GRPOTrainer(model='trl-internal-testing/tiny-Qwen2ForCausalLM-2.5', reward_funcs=logging_reward,
                args=args, train_dataset=Dataset.from_list(build_rows(SYMS, '2016-01-01', '2019-12-31'))).train()
