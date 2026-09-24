"""Offline smoke test for trading_rl (no Alpaca, no GPU). Exits non-zero on failure.

    python .pi/skills/trading-rl-smoke-test/scripts/smoke.py [--repo PATH]
"""

import argparse
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')))
args = ap.parse_args()
sys.path[:0] = [args.repo, HERE]

import matplotlib
matplotlib.use('Agg')
import pandas as pd

import stubs
import trading_rl as tr

stubs.install_fake_bars()
SYM, START, END = 'AAPL', '2020-01-01', '2021-12-31'
bt = tr.Backtester(verbose=False)
failures = []


def check(name, cond, detail=''):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)
    if not cond:
        failures.append(name)


# 1. every reward status
for expected, text in stubs.SAMPLE_OUTPUTS.items():
    t = time.time()
    s = tr.score_strategy(tr.extract_function(text), bt, SYM, START, END, timeout=2)
    exp_reward = max(s.result.avg_annual_return_pct, 1) if expected == 'profitable' else tr.REWARDS[expected]
    check(f"status {expected}", s.status == expected and s.reward == exp_reward,
          f"got {s.status} reward={s.reward} ({time.time() - t:.1f}s)")

# 2. <think> blocks are ignored by the extractor
think = "<think>class Strategy(bt.Strategy): junk</think>" + stubs.SAMPLE_OUTPUTS['no_trades']
check("strip <think>", tr.extract_function(think).startswith("class Strategy(bt.Strategy):\n"))

# 3. prompt round-trip
p = tr.make_prompt(SYM, START, END)
check("prompt params", tr.extract_trading_parameters(p) == (SYM, START, END))
check("compact prompt", len(tr.make_prompt(SYM, START, END, compact=True)) < len(p))

# 4. TRL reward function
fn = tr.make_strategy_reward(bt, timeout=2)
rewards = fn([[{'content': stubs.SAMPLE_OUTPUTS['no_trades']}], [{'content': stubs.SAMPLE_OUTPUTS['missing_methods']}]],
             prompts=[[{'role': 'user', 'content': p}]] * 2)
check("reward fn", list(rewards) == [-1, -10], str(rewards))

# 5. timeout off the main thread (daemon-thread fallback)
import threading
box = {}
th = threading.Thread(target=lambda: box.setdefault('s', tr.score_strategy(
    tr.extract_function(stubs.SAMPLE_OUTPUTS['timeout']), bt, SYM, START, END, timeout=1)))
th.start(); th.join(10)
check("timeout in thread", box.get('s') is not None and box['s'].status == 'timeout')

# 6. evaluate + save with plots
with tempfile.TemporaryDirectory() as d:
    recs = tr.evaluate_symbol(lambda _: stubs.SAMPLE_OUTPUTS['profitable'], bt, SYM, START, END, 2, save_dir=d)
    saved = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]
    check("evaluate_symbol", len(recs) == 2 and all(r['status'] == 'profitable' for r in recs))
    check("save_strategy files", {'strategy.py', 'stats.json', 'plot_1.png'} <= {os.path.basename(f) for f in saved},
          f"{len(saved)} files")
    tr.summarize(pd.DataFrame(recs))

print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else ""))
sys.exit(1 if failures else 0)
