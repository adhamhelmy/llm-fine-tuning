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

# 4. TRL reward function (in-process and with a worker pool)
fn = tr.make_strategy_reward(bt, timeout=2, workers=0)
rewards = fn([[{'content': stubs.SAMPLE_OUTPUTS['no_trades']}], [{'content': stubs.SAMPLE_OUTPUTS['missing_methods']}]],
             prompts=[[{'role': 'user', 'content': p}]] * 2)
check("reward fn", list(rewards) == [-1, -10], str(rewards))

# A TRL reward call holds several prompts: each completion must use its OWN row's window.
from trading_rl.dataset import build_rows
rows = build_rows(['AAPL', 'MSFT'], START, END)
check("dataset columns", rows[1]['symbol'] == 'MSFT' and rows[1]['start'] == START and 'prompt' in rows[1])
good = stubs.SAMPLE_OUTPUTS['profitable']
per_row = [tr.score_strategy(tr.extract_function(good), bt, s, START, END).reward for s in ('AAPL', 'MSFT')]
batch = dict(completions=[[{'content': good}]] * 4, prompts=[r['prompt'] for r in rows for _ in (0, 1)],
             symbol=['AAPL', 'AAPL', 'MSFT', 'MSFT'], start=[START] * 4, end=[END] * 4)
check("reward per row (kwargs)", fn(**batch) == [per_row[0]] * 2 + [per_row[1]] * 2 and per_row[0] != per_row[1],
      f"{per_row}")
no_kw = {k: batch[k] for k in ('completions', 'prompts')}
check("reward per row (prompt)", fn(**no_kw) == [per_row[0]] * 2 + [per_row[1]] * 2)

runs = []
_orig_run0 = tr.Backtester.run
tr.Backtester.run = lambda self, *a, **k: (runs.append(a[1]), _orig_run0(self, *a, **k))[1]
fresh = tr.make_strategy_reward(bt, timeout=2, workers=0)
fresh(**batch); fresh(**batch)
tr.Backtester.run = _orig_run0
check("reward memo", runs == ['AAPL', 'MSFT'], str(runs))

pooled = tr.make_strategy_reward(bt, timeout=2, workers=2)
check("reward fn (pool)", pooled(**batch) == [per_row[0]] * 2 + [per_row[1]] * 2)
pooled.pool().close()

# 4b. extraction edge cases
body = "class Strategy(bt.Strategy):\n    def __init__(self):\n        pass\n    def next(self):\n        pass\n"
for label, text in {
    'trailing prose': body + "\nThis strategy buys dips.",
    'fence + prose': "```python\n" + body + "```\nExplanation.",
    'later fence': "```bash\npip install x\n```\n```python\n" + body + "```",
    'unclosed fence': "```python\n" + body,
    'indented': "Code:\n" + "\n".join("    " + l for l in body.splitlines()) + "\nDone.",
    'col-0 comment': body.replace("    def next", "# trading logic\n    def next"),
}.items():
    code = tr.extract_function(text)
    check(f"extract: {label}", code is not None and tr.validate_code(code) is None and tr.has_required_functions(code),
          repr(code)[:120])
check("methods via ast (comment doesn't count)",
      not tr.has_required_functions("class Strategy(bt.Strategy):\n    # def __init__(self):\n    def next(self): pass\n"))
broken = "class Strategy(bt.Strategy):\n    def __init__(self):\n        x = (\n    def next(self): pass\n"
check("syntax error -> invalid_code", tr.score_strategy(broken, bt, SYM, START, END).status == 'invalid_code')

# 4c. reward configs, costs, CAGR, traded
code = tr.extract_function(good)
orig = tr.score_strategy(code, bt, SYM, START, END, reward=tr.RewardConfig.original())
dflt = tr.score_strategy(code, bt, SYM, START, END)
check("costs lower the return", dflt.result.return_pct < orig.result.return_pct,
      f"{orig.result.return_pct:.2f} -> {dflt.result.return_pct:.2f}")
r = dflt.result
check("cagr + orders", r.n_orders > 2 and r.traded and abs(r.cagr_pct) > 0 and
      abs((1 + r.cagr_pct / 100) ** 2 - (1 + r.return_pct / 100)) < 0.02, str(r))
for metric, expect in (('cagr', max(r.cagr_pct, 1)), ('return', max(r.return_pct, 1)),
                       ('sharpe', max(r.sharpe_ratio or 0, 1))):
    got = tr.score_strategy(code, bt, SYM, START, END, reward=tr.RewardConfig(metric)).reward
    check(f"metric {metric}", abs(got - expect) < 1e-9 if r.return_pct > 0 else got == 0, f"{got} vs {expect}")
custom = tr.RewardConfig(lambda res: res.cagr_pct - res.max_drawdown_pct, floor=None, loss=None)
check("custom metric", abs(tr.score_strategy(code, bt, SYM, START, END, reward=custom).reward
                           - (r.cagr_pct - r.max_drawdown_pct)) < 1e-9)
bh = tr.benchmark_result(bt, SYM, START, END)
ex = tr.score_strategy(code, bt, SYM, START, END, reward=tr.RewardConfig('excess_cagr', floor=None, loss=None))
check("excess_cagr metric", abs(ex.reward - (r.cagr_pct - bh.cagr_pct)) < 1e-9 and bh.n_orders == 1,
      f"{ex.reward:.2f}")
check("clip", tr.score_strategy(code, bt, SYM, START, END, reward=tr.RewardConfig(clip=(None, 0.5))).reward <= 0.5)
check("reward names", tr.RewardConfig().name == 'avg_annual,costs=low'
      and tr.RewardConfig.original().name == 'avg_annual,costs=none'
      and tr.RewardConfig('sharpe', costs=(0, 5), loss=None).name == 'sharpe,costs=0+5bps,loss=None')
loser = tr.BacktestResult(-5.0, -0.3, -2.0, 10.0, -2.5, 4)
check("loss rewards", tr.RewardConfig().score(loser) == ('loss', 0)
      and tr.RewardConfig('cagr', loss=None, clip=(-0.99, None)).score(loser) == ('loss', -0.99)
      and tr.RewardConfig().score(loser._replace(n_orders=0)) == ('no_trades', -1))

# 4c'. Jensen's alpha: buy-and-hold has beta ~1 / alpha ~0; a half-invested copy has beta ~0.5;
# the alpha reward ranks negative alphas below positive ones and every trader above a non-trader
from trading_rl.backtest import jensen_alpha
import numpy as np
px = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, 500))
a0, b0, _ = jensen_alpha(px * 3, px)
half = 0.5 * px / px[0] + 0.5            # 50% stock, 50% cash
a1, b1, _ = jensen_alpha(half, px)
noise = np.random.default_rng(1).normal(0, 0.002, 499)
boosted = np.cumprod(np.r_[1, 1 + (px[1:] / px[:-1] - 1) + 0.0004 + noise])   # +4 bps/bar on top
a2, b2, t2 = jensen_alpha(boosted, px)
check("jensen alpha", abs(a0) < 1e-6 and abs(b0 - 1) < 1e-9 and abs(a2 - 0.0004 * 252 * 100) < 5
      and abs(b2 - 1) < 0.01 and t2 > 2 and 0.4 < b1 < 0.6 and jensen_alpha(px[:5], px[:5]) == (0.0, 0.0, 0.0),
      f"a0={a0:.3g} b0={b0:.3g} b1={b1:.3g} a2={a2:.3g} t2={t2:.3g}")
check("wiped-out account = worst alpha", jensen_alpha(np.r_[px[:100], -1.0, px[:100]], px[:201])[0] == -100.0)
PEEK = """class Strategy(bt.Strategy):
    def __init__(self):
        pass
    def next(self):
        if self.data.close[1] > self.data.close[0] and not self.position:
            self.order_target_percent(target=0.95)
"""
pk = tr.score_strategy(PEEK, bt, SYM, START, END)
raw = tr.score_strategy(PEEK.replace("        pass", "        self.df = self.data.p.dataname"), bt, SYM, START, END)
check("lookahead blocked", pk.status == 'exception' and 'IndexError' in pk.error
      and raw.status == 'invalid_code' and 'dataname' in raw.error, f"{pk.status} / {raw.status}")
check("bh beta ~1", abs(bh.beta - 1) < 0.05 and abs(bh.alpha_pct) < 1, f"beta={bh.beta:.3f} alpha={bh.alpha_pct:.3f}")
ar = tr.RewardConfig.alpha()
good_a, bad_a, worse_a = (loser._replace(alpha_pct=v) for v in (12.0, -3.0, -40.0))
check("alpha reward", ar.score(good_a) == ('loss', 12.0) and -0.99 < ar.score(worse_a)[1] < ar.score(bad_a)[1] < 0
      and ar.score(loser._replace(alpha_pct=500.0))[1] == 50.0 and ar.score(good_a._replace(n_orders=0))[1] == -1
      and tr.score_strategy(code, bt, SYM, START, END, reward=tr.RewardConfig('alpha', floor=None, loss=None)).reward == r.alpha_pct,
      ar.name)

# 4d. subprocess pool: hard timeout (C-level loop SIGALRM can't break), crash, print, parity
from trading_rl.scoring import ScoringPool
C_LOOP = body.replace("    def next(self):\n        pass", "    def next(self):\n        sum(range(10 ** 12))")
CRASH = "import os\n" + body.replace("    def next(self):\n        pass", "    def next(self):\n        os._exit(3)")
PRINTS = body.replace("    def next(self):\n        pass", "    def next(self):\n        print('noise' * 100)\n        if not self.position: self.buy()")
with ScoringPool(workers=2, timeout=1, grace=1) as pool:
    t = time.time()
    outs = tr.score_many([(C_LOOP, SYM, START, END), (CRASH, SYM, START, END), (PRINTS, SYM, START, END),
                          (code, 'MSFT', START, END)], bt, timeout=1, pool=pool)
    check("pool: hard timeout", outs[0].status == 'timeout' and time.time() - t < 15, f"{time.time() - t:.1f}s")
    check("pool: crash contained", outs[1].status == 'exception' and 'WorkerCrashed' in outs[1].error, str(outs[1]))
    check("pool: print-safe", outs[2].status in ('profitable', 'loss'), str(outs[2])[:120])
    inproc = tr.score_strategy(code, bt, 'MSFT', START, END)
    check("pool == in-process", outs[3].reward == inproc.reward and outs[3].result == inproc.result)
    check("pool recovers", tr.score_strategy(code, bt, SYM, START, END, pool=pool).reward == dflt.reward)
    exc = tr.score_strategy(tr.extract_function(stubs.SAMPLE_OUTPUTS['exception']), bt, SYM, START, END, pool=pool)
    check("pool: exception line", exc.status == 'exception' and 'ZeroDivisionError (line' in exc.error, exc.error)

# 4e. symbols, data checks, disk cache, CSV schema evolution
check("resolve_symbols dedupe", tr.resolve_symbols(['aapl', 'AAPL ', 'msft']) == ['AAPL', 'MSFT']
      and tr.resolve_symbols('aapl, msft,aapl') == ['AAPL', 'MSFT'])
check("symbol sets", len(tr.TRAINING_SYMBOLS) == 35 == len(set(tr.TRAINING_SYMBOLS))
      and len(set(tr.DOW30)) == 30 and 'INTC' not in tr.TRAINING_SYMBOLS
      and len(tr.EVAL_SYMBOLS) == 29 and tr.resolve_symbols('eval') == tr.EVAL_SYMBOLS
      and not set(tr.SYMBOL_ISSUES) & set(tr.TRAINING_SYMBOLS + tr.EVAL_SYMBOLS))
check("usable_symbols", tr.usable_symbols(['AAPL', 'COIN', 'NIO'], '2016-01-01') == ['AAPL']
      and tr.usable_symbols(['COIN', 'NIO'], '2022-01-01') == ['COIN', 'NIO'])
flat = stubs.fake_frame('X', '2020-01-01', '2020-06-30')
flat.iloc[10:60, flat.columns.get_loc('close')] = flat['close'].iloc[10]
flat.iloc[10:60, flat.columns.get_loc('volume')] = 0
check("bar_issues", tr.bar_issues(flat) and not tr.bar_issues(stubs.fake_frame('X', '2020-01-01', '2020-06-30'))
      and tr.bar_issues(pd.DataFrame()) == ['no bars']
      and tr.bar_issues(stubs.fake_frame('X', '2020-03-01', '2020-06-30'), start='2020-01-01')[0].startswith('first bar'))
with tempfile.TemporaryDirectory() as d:
    cached = tr.Backtester(verbose=False, cache_dir=d)
    cached.get_bars('ZZZ', START, END)
    tr.Backtester._cache.pop(('ZZZ', 'day', START, END))
    fetches = []
    _f = tr.Backtester._fetch_bars
    tr.Backtester._fetch_bars = lambda self, *a, **k: (fetches.append(a), _f(self, *a, **k))[1]
    again = tr.Backtester(verbose=False, cache_dir=d).get_bars('ZZZ', START, END)
    tr.Backtester._fetch_bars = _f
    check("disk cache", not fetches and len(again) > 400 and os.listdir(d) == ['ZZZ_day_2020-01-01_2021-12-31.pkl'])
    out = os.path.join(d, 'r.csv')
    tr.append_csv(out, {'a': 1, 'b': 2})
    tr.append_csv(out, {'a': 3, 'b': 4, 'c': 5})
    tr.append_csv(out, {'a': 6})
    got = pd.read_csv(out)
    check("append_csv schema", list(got.columns) == ['a', 'b', 'c'] and got['a'].tolist() == [1, 3, 6])

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

lines = []
rec = L.generate_strategy(ScriptedLLM(), L.LadderConfig(repair_rounds=2, refine_rounds=1, analyst=True),
                          bt, SYM, TS, TE, timeout=5, log=lambda *a, **k: lines.append(' '.join(map(str, a))))
logged = '\n'.join(lines)
check("log=print traces every step", all(t in logged for t in
      ('LLM call 1: analyst', 'repair 1/2', 'refine 1/1', '[feedback #1] status=exception',
       '[select]', '[final] status=')), logged[-400:])

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
    L.dump_prompts(os.path.join(d, 'prompts'), [full, full.replace(analyst=True)], bt)
    check("dump prompts", len(os.listdir(os.path.join(d, 'prompts'))) == 3)
tr.Backtester.run = _orig_run

# ── 8. robustness: costs, B&H comparison, stats, years, duplicates ──────────────
from trading_rl import robustness as R
SMA = ("class Strategy(bt.Strategy):\n    def __init__(self):\n        self.fast = bt.indicators.SMA(self.data.close, period=10)\n"
       "        self.slow = bt.indicators.SMA(self.data.close, period=30)\n    def next(self):\n"
       "        if not self.position and self.fast[0] > self.slow[0]:\n            self.order_target_percent(target=0.95)\n"
       "        elif self.position and self.fast[0] < self.slow[0]:\n            self.close()\n")
SMA_RENAMED = SMA.replace('self.fast', 'self.quick').replace('self.slow', 'self.lagging') + "        # tweak\n"
FLIP = ("class Strategy(bt.Strategy):\n    def __init__(self):\n        pass\n    def next(self):\n"
        "        if self.position:\n            self.close()\n        else:\n            self.order_target_percent(target=0.95)\n")
RS, RE = '2022-01-01', '2024-12-31'
res = pd.DataFrame([{'model': 'm', 'config': c, 'symbol': sym, 'start': RS, 'end': RE, 'sample': smp,
                     'status': 'loss', 'strategy_code': code}
                    for sym in ('AAPL', 'MSFT') for smp in (1, 2)
                    for c, code in (('sma', SMA if smp == 1 else SMA_RENAMED), ('flip', FLIP))])
rb = R.rebacktest(res, bt, cost_levels={'none': (0, 0), 'high': (10, 15)}, timeout=20)
check("rebacktest rows", len(rb) == (len(res) + 2) * 2 and rb['rerun_status'].ne('error').all(),
      str(rb['rerun_status'].value_counts().to_dict()))
f = rb[(rb['config'] == 'flip') & (rb['symbol'] == 'AAPL') & (rb['sample'] == 1)].set_index('cost')
check("costs reduce return", f.at['high', 'return_pct'] < f.at['none', 'return_pct'] - 5
      and f.at['high', 'commission_paid'] > 0 and f.at['none', 'commission_paid'] == 0,
      f"{f.at['none', 'return_pct']:.1f} -> {f.at['high', 'return_pct']:.1f}")
check("turnover/trades", f.at['none', 'n_trades'] > 300 and f.at['none', 'turnover'] > 50,
      f"trades={f.at['none', 'n_trades']} turnover={f.at['none', 'turnover']:.0f}")
bh = rb[(rb['config'] == 'buy_and_hold') & (rb['cost'] == 'none') & (rb['symbol'] == 'AAPL')].iloc[0]
check("year columns", all(f'year_{y}' in rb for y in (2022, 2023, 2024)) and bh['n_trades'] == 1
      and bh['volatility_pct'] > 0)
sma_rows = rb[(rb['config'] == 'sma') & (rb['symbol'] == 'AAPL') & (rb['sample'] == 1)].set_index('cost')
plain = tr.score_strategy(SMA, bt, 'AAPL', RS, RE, reward=tr.RewardConfig.original()).result
check("cost 'none' == original reward backtest", abs(plain.return_pct - sma_rows.at['none', 'return_pct']) < 1e-9)
costly = tr.score_strategy(SMA, bt, 'AAPL', RS, RE, reward=tr.RewardConfig(costs='high')).result
check("cost 'high' == RewardConfig(costs='high')", abs(costly.return_pct - sma_rows.at['high', 'return_pct']) < 1e-9)
cr = R.cost_report(rb)
check("cost report", len(cr) == 4 and cr['bh_annual'].notna().all()
      and cr.set_index(['config', 'cost']).at[('flip', 'high'), 'beat_bh'] == 0.0)
print(R.format_cost_report(cr).to_string())
check("per asset", len(R.per_asset(rb, 'high')) == 4)
yr = R.year_report(rb, 'none')
check("year report", set(yr['year']) == {2022, 2023, 2024} and set(yr['regime']) == {'bear', 'bull'})
# Wilcoxon exact p for n=5 all positive = 2/32; all ties of zero -> n=0
check("wilcoxon exact", abs(R.signed_rank_test([1, 2, 3, 4, 5])['p'] - 0.0625) < 1e-12
      and R.signed_rank_test([0, 0])['n'] == 0)
mc = R.mcnemar(pd.DataFrame({'config': ['a'] * 6 + ['b'] * 6, 'symbol': list('xyzxyz') * 2,
                             'sample': [1, 1, 1, 2, 2, 2] * 2,
                             'status': ['exception'] * 6 + ['loss'] * 6}), 'a', 'b')
check("mcnemar", mc['b_only'] == 6 and abs(mc['p'] - 2 / 64) < 1e-12, str(mc))
# renamed identifiers + a comment = exact duplicate; a changed period = near-duplicate
near = res.copy()
near.loc[(near['config'] == 'sma') & (near['symbol'] == 'MSFT') & (near['sample'] == 2), 'strategy_code'] = \
    SMA.replace('period=30', 'period=40')
summ, pairs = R.duplicates(near)
sma = summ.set_index('config').loc['sma']
check("near-duplicates", sma['unique_exact'] == 2 and sma['n_clusters'] == 1 and pairs['exact'].sum() == 3 + 6
      and summ.set_index('config').at['flip', 'unique_exact'] == 1, summ.to_string())

print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else ""))
sys.exit(1 if failures else 0)
