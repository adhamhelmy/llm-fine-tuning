"""The LLM-engineering ladder: prompt -> context -> harness -> loop -> graph, as ablations.

Every generation setting is a `LadderConfig`. Stages are explored greedily: evaluate the
variants of one stage on the DEV window, keep the best, and build the next stage on top of it:

    best = LadderConfig()
    for stage in ('prompt', 'context', 'harness', 'loop', 'graph'):
        configs = stage_configs(stage, best)
        df = run_ablation(generate, bt, configs, symbols, DEV_START, DEV_END, n_samples, 'dev.csv')
        best = select_best(report(df, baselines), configs)
    # then: final TEST evaluation of the chosen rungs; GRPO trains with best's prompt+context

Harness and loop feedback (runtime errors, in-sample metrics) is computed ONLY on the
FEEDBACK_YEARS before the evaluated window, so nothing about the evaluated window leaks.
The evaluated window itself is backtested exactly once, on the final selected strategy.
"""

import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .backtest import Backtester, resolve_symbols
from .baselines import BASELINES
from .codegen import extract_function, strip_thinking
from .config import FEEDBACK_YEARS
from .context import api_reference, few_shot_examples, format_market_summary, market_stats
from .knowledge import KnowledgeBase
from .prompt import make_prompt
from .rewards import score_strategy

OK_STATUSES = ('profitable', 'loss')
VALID_STATUSES = ('profitable', 'loss', 'no_trades')


@dataclass(frozen=True)
class LadderConfig:
    # prompt engineering
    compact: bool = False        # drop the class skeleton (the comparison notebooks' prompt)
    examples: int = 0            # few-shot example strategies
    plan: bool = False           # <think> reasoning before the code
    anonymize: bool = False      # hide ticker/dates: memorization probe (evaluation only)
    # context engineering
    api_reference: bool = False  # Backtrader API cheat sheet
    market_summary: bool = False # pre-window statistics of the stock
    rag_k: int = 0               # retrieved strategy-knowledge documents
    # harness engineering
    repair_rounds: int = 0       # feed errors / "never traded" back to the model
    # loop engineering (inference time; GRPO is the training-time loop)
    refine_rounds: int = 0       # feed in-sample metrics back and keep the best candidate
    # graph engineering
    analyst: bool = False        # analyst node writes a spec, coder node implements it

    @property
    def name(self):
        parts = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v != f.default:
                parts.append(f.name if v is True else f"{f.name}={v}")
        return ','.join(parts) or 'baseline'

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @property
    def uses_feedback(self):
        return self.repair_rounds > 0 or self.refine_rounds > 0


# Variants tried at each stage, applied on top of the best config of the previous stage.
STAGES = {
    'prompt':  [{}, {'compact': True}, {'examples': 2}, {'plan': True}, {'examples': 2, 'plan': True}],
    'context': [{}, {'api_reference': True}, {'market_summary': True}, {'rag_k': 3},
                {'api_reference': True, 'market_summary': True, 'rag_k': 3}],
    'harness': [{}, {'repair_rounds': 1}, {'repair_rounds': 3}],
    'loop':    [{}, {'refine_rounds': 2}, {'refine_rounds': 4}],
    'graph':   [{}, {'analyst': True}],
    'probe':   [{}, {'anonymize': True}],   # diagnostic, not a ladder stage
}


def stage_configs(stage, base=LadderConfig()):
    out = []
    for variant in STAGES[stage]:
        cfg = base.replace(**variant)
        if cfg not in out:
            out.append(cfg)
    return out


def feedback_window(start, years=FEEDBACK_YEARS):
    end_d = date.fromisoformat(start) - timedelta(days=1)
    start_d = end_d.replace(year=end_d.year - years) + timedelta(days=1)
    return start_d.isoformat(), end_d.isoformat()


# ── prompts ────────────────────────────────────────────────────────────────────

ANALYST_SETUP = """\
CONSTRAINTS: the strategy will be implemented as a Backtrader bt.Strategy on this single stock's
daily bars, using only Backtrader built-in indicators, orders via buy/sell/close/
order_target_percent, no external data and no lookahead."""

ANALYST_INSTRUCTION = """\
ROLE: You are a quantitative analyst. Do NOT write code. Write a concise strategy
specification (at most 150 words) that a developer will implement in Backtrader:
1. Market hypothesis (which behaviour of this stock the strategy exploits)
2. Indicators with exact parameters (Backtrader built-in indicators only)
3. Exact entry and exit rules
4. Position sizing and risk control"""

REPAIR_TEMPLATE = """\
{prompt}

Your previous answer:
```python
{code}
```

It failed a dry run: {feedback}

Fix the problem and return the complete corrected class only."""

REFINE_TEMPLATE = """\
{prompt}

Your previous answer:
```python
{code}
```

In-sample dry run ({fstart} to {fend}): {metrics}. Buy-and-hold over the same period:
{bh_annual:.2f}% average annual return.
Improve the strategy's risk-adjusted return on unseen future data. Avoid overfitting to this
period (keep rules simple and parameters conventional). Return the complete class only."""


class _Context:
    """Per-(symbol, window) context shared by all samples: stats, sections, B&H feedback."""

    def __init__(self, backtester, kb):
        self.backtester, self.kb = backtester, kb or KnowledgeBase.default()
        self._stats, self._bh = {}, {}

    def stats(self, symbol, start):
        key = (symbol, start)
        if key not in self._stats:
            try:
                self._stats[key] = market_stats(self.backtester, symbol, start, FEEDBACK_YEARS)
            except Exception as e:
                print(f"  (market stats unavailable for {symbol}: {str(e)[:80]})")
                self._stats[key] = None
        return self._stats[key]

    def bh_annual(self, symbol, start, end):
        key = (symbol, start, end)
        if key not in self._bh:
            s = score_strategy(BASELINES['buy_and_hold'], self.backtester, symbol, start, end, timeout=60)
            self._bh[key] = s.result.avg_annual_return_pct if s.result else float('nan')
        return self._bh[key]

    def sections(self, cfg, symbol, start):
        stats = self.stats(symbol, start) if (cfg.market_summary or cfg.rag_k) else None
        label = None if cfg.anonymize else symbol
        query = ' '.join((stats or {}).get('regime', [])) + ' daily stock trading strategy robust'
        return [
            api_reference() if cfg.api_reference else None,
            format_market_summary(stats, label, not cfg.anonymize) if cfg.market_summary else None,
            self.kb.context_block(query, cfg.rag_k) if cfg.rag_k else None,
            few_shot_examples(cfg.examples) if cfg.examples else None,
        ]


def build_prompt(cfg: LadderConfig, symbol, start, end, backtester=None, kb=None, _ctx=None):
    """The exact coder prompt for a config (context needs a backtester for market stats)."""
    ctx = _ctx or _Context(backtester, kb)
    return make_prompt(symbol, start, end, compact=cfg.compact, plan=cfg.plan,
                       anonymize=cfg.anonymize, sections=ctx.sections(cfg, symbol, start))


def build_analyst_prompt(cfg: LadderConfig, symbol, start, end, backtester=None, kb=None, _ctx=None):
    """Graph node 1: same context as the coder (minus code examples), no code-output rules."""
    ctx = _ctx or _Context(backtester, kb)
    header = ("Design a trading strategy for a US-listed stock, backtested on daily bars over several years."
              if cfg.anonymize else f"Design a trading strategy for {symbol} from {start} to {end}.")
    sections = ctx.sections(cfg.replace(examples=0), symbol, start)
    return '\n\n'.join(p for p in [header, ANALYST_SETUP, *sections, ANALYST_INSTRUCTION] if p)


def prompt_fn(cfg: LadderConfig, backtester=None, kb=None):
    """(symbol, start, end) -> prompt, for GRPO datasets (prompt + context stages only)."""
    assert not cfg.anonymize, "GRPO's reward parses symbol/dates from the prompt header"
    ctx = _Context(backtester, kb)
    return lambda symbol, start, end: build_prompt(cfg, symbol, start, end, _ctx=ctx)


def _metrics_text(r):
    sharpe = f"{r.sharpe_ratio:.2f}" if r.sharpe_ratio is not None else "n/a"
    return (f"total return {r.return_pct:.2f}%, average annual return {r.avg_annual_return_pct:.2f}%, "
            f"Sharpe {sharpe}, max drawdown {r.max_drawdown_pct:.2f}%")


def _repair_feedback(score):
    if score.status == 'no_trades':
        return "the strategy never placed a trade. Make the entry conditions reachable."
    if score.status == 'timeout':
        return "the backtest timed out (infinite loop or extremely slow logic)."
    return f"{score.status}: {score.error}"


# ── one sample through the whole pipeline ──────────────────────────────────────

def generate_strategy(generate, cfg: LadderConfig, backtester: Backtester, symbol, start, end,
                      kb=None, timeout=10, cash=10_000, _ctx=None):
    """Run prompt -> (graph) -> generation -> (harness/loop on the feedback window) -> final
    test on [start, end]. Returns a flat record."""
    ctx = _ctx or _Context(backtester, kb)
    rec = {'config': cfg.name, 'config_json': cfg.to_json(), 'symbol': symbol, 'start': start,
           'end': end, 'status': None, 'reward_score': None, 'return_pct': None,
           'sharpe_ratio': None, 'avg_annual_return_pct': None, 'max_drawdown_pct': None,
           'strategy_code': None, 'error': None, 'llm_calls': 0, 'candidates': 1,
           'val_status': None, 'val_reward': None, 'val_avg_annual_return_pct': None,
           'spec': None, 'prompt_chars': None, 'prompt_sha': None, 'gen_seconds': 0.0}
    t0 = time.time()

    def call(prompt):
        rec['llm_calls'] += 1
        return generate(prompt)

    try:
        prompt = build_prompt(cfg, symbol, start, end, _ctx=ctx)
        if cfg.analyst:
            spec = strip_thinking(call(build_analyst_prompt(cfg, symbol, start, end, _ctx=ctx)))
            rec['spec'] = spec
            prompt += f"\n\nIMPLEMENT THIS STRATEGY SPECIFICATION:\n{spec}"
        rec['prompt_chars'] = len(prompt)
        rec['prompt_sha'] = hashlib.sha1(prompt.encode()).hexdigest()[:10]
        code = extract_function(call(prompt))

        if cfg.uses_feedback:
            fstart, fend = feedback_window(start)
            val = score_strategy(code, backtester, symbol, fstart, fend, cash=cash, timeout=timeout)
            candidates, repairs, refines = [(code, val)], 0, 0
            while True:
                if val.status not in OK_STATUSES:
                    if repairs >= cfg.repair_rounds:
                        break
                    repairs += 1
                    follow = REPAIR_TEMPLATE.format(prompt=prompt, code=code or '(none)',
                                                    feedback=_repair_feedback(val))
                else:
                    if refines >= cfg.refine_rounds:
                        break
                    refines += 1
                    follow = REFINE_TEMPLATE.format(prompt=prompt, code=code, fstart=fstart, fend=fend,
                                                    metrics=_metrics_text(val.result),
                                                    bh_annual=ctx.bh_annual(symbol, fstart, fend))
                code = extract_function(call(follow))
                val = score_strategy(code, backtester, symbol, fstart, fend, cash=cash, timeout=timeout)
                candidates.append((code, val))
            # keep the best candidate by in-sample reward (latest wins ties)
            code, val = max(reversed(candidates), key=lambda cv: cv[1].reward)
            rec.update(candidates=len(candidates), val_status=val.status, val_reward=val.reward,
                       val_avg_annual_return_pct=val.result.avg_annual_return_pct if val.result else None)
    except Exception as e:  # generation / context failure
        rec.update(status='exception', reward_score=-2, error=f"pipeline: {str(e)[:200]}",
                   gen_seconds=time.time() - t0)
        return rec

    rec['gen_seconds'] = time.time() - t0
    score = score_strategy(code, backtester, symbol, start, end, cash=cash, timeout=timeout)
    rec.update(strategy_code=code, status=score.status, reward_score=score.reward, error=score.error)
    if score.result is not None:
        rec.update(score.result._asdict())
    return rec


def run_ablation(generate, backtester: Backtester, configs, symbols, start, end, n_samples,
                 out_csv, model='', kb=None, timeout=10, cash=10_000):
    """Evaluate every config x symbol x sample; appends to out_csv and resumes from it."""
    symbols = resolve_symbols(symbols)
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev['model'], prev['config'], prev['symbol'], prev['start'], prev['sample']))
    ctx = _Context(backtester, kb)
    total = len(configs) * len(symbols) * n_samples
    i = 0
    for cfg in configs:
        for symbol in symbols:
            for sample in range(1, n_samples + 1):
                i += 1
                if (model, cfg.name, symbol, start, sample) in done:
                    continue
                rec = generate_strategy(generate, cfg, backtester, symbol, start, end,
                                        timeout=timeout, cash=cash, _ctx=ctx)
                rec.update(model=model, sample=sample)
                pd.DataFrame([rec]).to_csv(out_csv, mode='a', index=False,
                                           header=not os.path.exists(out_csv))
                print(f"[{i}/{total}] {cfg.name:40.40} {symbol:5} s{sample} -> {rec['status']}"
                      + (f" ({rec['candidates']} cand.)" if rec['candidates'] > 1 else ""))
    df = pd.read_csv(out_csv)
    return df[(df['model'] == model) & (df['start'] == start) &
              (df['config'].isin([c.name for c in configs]))]


# ── statistics ─────────────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def bootstrap_mean(x, n_boot=2000, seed=0):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return (x.mean(), *np.percentile(boots, [2.5, 97.5]))


def add_indicators(df, baselines=None):
    """Adds boolean columns valid / traded / profitable / beat_bh and bh_annual."""
    df = df.copy()
    df['valid'] = df['status'].isin(VALID_STATUSES)
    df['traded'] = df['status'].isin(OK_STATUSES)
    df['profitable'] = df['status'].eq('profitable')
    if baselines is not None:
        bh = baselines[baselines['config'] == 'buy_and_hold'].set_index(['symbol', 'start'])
        df['bh_annual'] = [bh['avg_annual_return_pct'].get((s, st), math.nan)
                           for s, st in zip(df['symbol'], df['start'])]
        df['beat_bh'] = df['traded'] & (df['avg_annual_return_pct'] > df['bh_annual'])
        df['excess_annual'] = df['avg_annual_return_pct'] - df['bh_annual']
    return df


def report(df, baselines=None, by='config'):
    """One row per config. Rates are over ALL samples (failures count as failures), with
    Wilson 95% CIs; means use bootstrap 95% CIs. *_lo/*_hi columns hold the bounds."""
    df = add_indicators(df, baselines)
    rows = []
    for name, g in df.groupby(by, sort=False):
        row = {by: name, 'n': len(g)}
        for col in ('valid', 'traded', 'profitable') + (('beat_bh',) if 'beat_bh' in g else ()):
            row[col], row[f'{col}_lo'], row[f'{col}_hi'] = wilson(int(g[col].sum()), len(g))
        row['mean_reward'], row['mean_reward_lo'], row['mean_reward_hi'] = bootstrap_mean(g['reward_score'])
        t = g[g['traded']]
        row['annual_traded'], row['annual_traded_lo'], row['annual_traded_hi'] = bootstrap_mean(t['avg_annual_return_pct'])
        if 'excess_annual' in t:
            row['excess_traded'], row['excess_traded_lo'], row['excess_traded_hi'] = bootstrap_mean(t['excess_annual'])
        row['median_sharpe_traded'] = t['sharpe_ratio'].median()
        row['median_drawdown_traded'] = t['max_drawdown_pct'].median()
        if 'llm_calls' in g:
            row['llm_calls'] = g['llm_calls'].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def format_report(rep, by='config'):
    """Human-readable table: 'value [lo, hi]' strings, rates in %."""
    out = pd.DataFrame({by: rep[by], 'n': rep['n']})
    for col in ('valid', 'traded', 'profitable', 'beat_bh'):
        if col in rep:
            out[f'{col} %'] = [f"{v*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"
                               for v, lo, hi in zip(rep[col], rep[f'{col}_lo'], rep[f'{col}_hi'])]
    for col, label in (('mean_reward', 'reward'), ('annual_traded', 'annual % (traded)'),
                       ('excess_traded', 'excess vs B&H % (traded)')):
        if col in rep:
            out[label] = [f"{v:.2f} [{lo:.2f}, {hi:.2f}]" if not math.isnan(v) else '-'
                          for v, lo, hi in zip(rep[col], rep[f'{col}_lo'], rep[f'{col}_hi'])]
    for col in ('median_sharpe_traded', 'llm_calls'):
        if col in rep:
            out[col] = rep[col].round(2)
    return out


def markdown_table(df):
    """GitHub-flavoured markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    rows = [' | '.join(str(v) for v in r) for r in df.itertuples(index=False)]
    return '\n'.join([f"| {' | '.join(cols)} |", '|' + '---|' * len(cols)] + [f"| {r} |" for r in rows]) + '\n'


def paired_delta(df, a, b, metric='profitable', baselines=None, n_boot=5000, seed=0, by='config'):
    """Difference (b - a) in `metric`, paired by symbol, with a symbol-level bootstrap CI
    and a two-sided bootstrap p-value. metric: valid/traded/profitable/beat_bh/reward_score/..."""
    df = add_indicators(df, baselines)
    per = df[df[by].isin([a, b])].groupby(['symbol', by])[metric].mean().astype(float).unstack(by)
    per = per.dropna(subset=[a, b])
    d = (per[b] - per[a]).to_numpy()
    if len(d) == 0:
        return {'a': a, 'b': b, 'metric': metric, 'n_symbols': 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'a': a, 'b': b, 'metric': metric, 'n_symbols': len(d),
            'delta': d.mean(), 'lo': lo, 'hi': hi, 'p': p}


def select_best(rep, configs, metric='profitable', by='config'):
    """Best config among `configs` on `metric` (ties: higher mean reward, then fewer LLM calls)."""
    names = [c.name for c in configs]
    r = rep[rep[by].isin(names)].copy()
    r['_calls'] = -r['llm_calls'] if 'llm_calls' in r else 0
    best = r.sort_values([metric, 'mean_reward', '_calls'], ascending=False).iloc[0][by]
    return configs[names.index(best)]


def dump_prompts(outdir, configs, backtester=None, symbol='AAPL', start=None, end=None, kb=None):
    """Write the exact prompt of every config (and the harness/loop/graph templates) to outdir."""
    from .config import TEST_END, TEST_START
    start, end = start or TEST_START, end or TEST_END
    os.makedirs(outdir, exist_ok=True)
    ctx = _Context(backtester, kb)
    for cfg in configs:
        with open(os.path.join(outdir, f"{cfg.name}.txt"), 'w') as f:
            if cfg.analyst:
                f.write("=== ANALYST NODE ===\n" + build_analyst_prompt(cfg, symbol, start, end, _ctx=ctx)
                        + "\n\n=== CODER NODE (followed by the analyst's specification) ===\n")
            f.write(build_prompt(cfg, symbol, start, end, _ctx=ctx) + "\n")
    with open(os.path.join(outdir, "_feedback_templates.txt"), 'w') as f:
        f.write(f"=== HARNESS: REPAIR ===\n{REPAIR_TEMPLATE}\n\n=== LOOP: REFINE ===\n{REFINE_TEMPLATE}\n")
    print(f"Wrote {len(configs) + 1} files to {outdir}/")
