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

# ── 7. ladder (prompt / context / harness / loop / graph) ─────────────────────
from trading_rl import ladder as L
from trading_rl.context import EXAMPLES, api_reference, market_stats

TS, TE = '2022-01-01', '2023-12-31'          # evaluated window
FS, FE = L.feedback_window(TS)               # feedback window (before TS)
check("feedback window", (FS, FE) == ('2020-01-01', '2021-12-31'), f"{FS}..{FE}")

for i, code in enumerate(EXAMPLES):
    s = tr.score_strategy(code, bt, SYM, TS, TE, timeout=5)
    check(f"few-shot example {i+1} runs", s.status in ('profitable', 'loss', 'no_trades'), s.status)
check("api reference", 'CrossOver(data0, data1)' in api_reference())

calls = []   # every backtest window actually run
_orig_run = tr.Backtester.run
def _spy(self, strategy, symbols, start, end, *a, **k):
    calls.append((start, end)); return _orig_run(self, strategy, symbols, start, end, *a, **k)
tr.Backtester.run = _spy

stats = market_stats(bt, SYM, TS)
check("market stats pre-window", stats and stats['window'].endswith('2021-12-31'), stats and stats['window'])

GOOD, BAD = stubs.SAMPLE_OUTPUTS['profitable'], stubs.SAMPLE_OUTPUTS['exception']

class ScriptedLLM:
    """Broken code first; fixed code once it sees repair feedback; spec for the analyst."""
    def __init__(self): self.prompts = []
    def __call__(self, prompt):
        self.prompts.append(prompt)
        if 'ROLE: You are a quantitative analyst' in prompt: return "Buy above SMA20, exit below."
        if 'failed a dry run' in prompt or 'In-sample dry run' in prompt: return GOOD
        return BAD

def run_cfg(cfg):
    llm = ScriptedLLM(); calls.clear()
    rec = L.generate_strategy(llm, cfg, bt, SYM, TS, TE, timeout=5)
    return rec, llm, list(calls)

rec, llm, win = run_cfg(L.LadderConfig())
check("baseline: 1 call, test once", rec['llm_calls'] == 1 and win == [(TS, TE)] and rec['status'] == 'exception',
      f"{rec['llm_calls']} calls, windows={win}")

rec, llm, win = run_cfg(L.LadderConfig(repair_rounds=2))
check("repair fixes the error", rec['status'] == 'profitable' and rec['llm_calls'] == 2 and rec['candidates'] == 2,
      f"{rec['status']} calls={rec['llm_calls']}")
check("repair: feedback has error", 'ZeroDivisionError (line' in llm.prompts[1], llm.prompts[1][-300:])
check("no test leakage (repair)", win.count((TS, TE)) == 1 and set(win) <= {(TS, TE), (FS, FE)}, str(win))

rec, llm, win = run_cfg(L.LadderConfig(repair_rounds=1, refine_rounds=2))
check("refine rounds", rec['llm_calls'] == 4 and rec['candidates'] == 4, f"calls={rec['llm_calls']}")
check("refine: in-sample metrics only", 'In-sample dry run (2020-01-01 to 2021-12-31)' in llm.prompts[-1]
      and 'Buy-and-hold' in llm.prompts[-1])
check("no test leakage (refine)", win.count((TS, TE)) == 1 and set(win) <= {(TS, TE), (FS, FE)}, str(win))

rec, llm, _ = run_cfg(L.LadderConfig(analyst=True, repair_rounds=1))
check("graph: analyst -> coder", rec['spec'] and 'IMPLEMENT THIS STRATEGY SPECIFICATION' in llm.prompts[1]
      and 'Buy above SMA20' in llm.prompts[1] and 'Output ONLY' not in llm.prompts[0] and 'Do NOT write code' in llm.prompts[0])

full = L.LadderConfig(examples=2, plan=True, api_reference=True, market_summary=True, rag_k=3)
p_full = L.build_prompt(full, SYM, TS, TE, bt)
check("full prompt sections", all(x in p_full for x in ('BACKTRADER API REFERENCE', 'MARKET CONTEXT for AAPL',
      'RELEVANT STRATEGY KNOWLEDGE', 'Example 2:', '<think>', 'OUTPUT FORMAT')))
check("GRPO header still parses", tr.extract_trading_parameters(L.prompt_fn(full, bt)(SYM, TS, TE)) == (SYM, TS, TE))
p_anon = L.build_prompt(full.replace(anonymize=True), SYM, TS, TE, bt)
check("anonymize hides ticker/dates", SYM not in p_anon and TS not in p_anon and '2021' not in p_anon)
check("default prompt unchanged", L.build_prompt(L.LadderConfig(), SYM, TS, TE) == tr.make_prompt(SYM, TS, TE))

kb = tr.KnowledgeBase.default()
check("RAG retrieval", kb.retrieve('mean-reverting range-bound', 2)[0].title.startswith(('RSI', 'Bollinger', 'Stochastic')),
      kb.retrieve('mean-reverting range-bound', 1)[0].title)

check("stage configs", [c.name for c in L.stage_configs('harness', L.LadderConfig(examples=2))]
      == ['examples=2', 'examples=2,repair_rounds=1', 'examples=2,repair_rounds=3'])

with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, 'abl.csv')
    cfgs = [L.LadderConfig(), L.LadderConfig(repair_rounds=1)]
    llm = ScriptedLLM()
    df = L.run_ablation(llm, bt, cfgs, ['AAPL', 'MSFT'], TS, TE, 2, out, model='m', timeout=5)
    n1 = len(llm.prompts)
    df = L.run_ablation(llm, bt, cfgs, ['AAPL', 'MSFT'], TS, TE, 2, out, model='m', timeout=5)
    check("ablation resumes", len(df) == 8 and len(llm.prompts) == n1, f"rows={len(df)} calls {n1}->{len(llm.prompts)}")
    base = tr.baseline_table(bt, ['AAPL', 'MSFT'], TS, TE)
    rep = L.report(df, base)
    check("report", list(rep['profitable']) == [0.0, 1.0] and 'beat_bh' in rep, str(rep[['config', 'profitable']].values.tolist()))
    print(L.format_report(rep).to_string())
    delta = L.paired_delta(df, 'baseline', 'repair_rounds=1')
    check("paired delta", delta['delta'] == 1.0 and delta['n_symbols'] == 2, str(delta))
    check("select best", L.select_best(rep, cfgs).name == 'repair_rounds=1')
    L.dump_prompts(os.path.join(d, 'prompts'), L.stage_configs('graph', full), bt)
    check("dump prompts", len(os.listdir(os.path.join(d, 'prompts'))) == 3)
tr.Backtester.run = _orig_run

print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else ""))
sys.exit(1 if failures else 0)
