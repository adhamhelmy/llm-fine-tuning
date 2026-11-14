"""Post-hoc robustness analysis of generated strategies (for the paper's results section).

Takes any results DataFrame/CSV with `strategy_code` (run_ablation, evaluate_symbol, the
comparison notebooks) and:

1. re-backtests every strategy that traded, plus the baselines, at several transaction-cost
   levels (`rebacktest`), recording drawdown, turnover, trade count, volatility and
   calendar-year returns;
2. compares every strategy with buy-and-hold at the same cost level (`cost_report`);
3. gives per-asset statistics: bootstrap CIs and Wilcoxon signed-rank tests on excess returns
   (`per_asset`), McNemar for validity between two configs (`mcnemar`) and a paired
   per-symbol Wilcoxon on any metric (`paired_wilcoxon`);
4. splits results by calendar year / market regime (`year_report`);
5. finds exact and near-duplicate strategies (`duplicates`).

This is reporting only. Nothing here feeds back into generation or config selection, so
re-running the TEST window at several cost levels does not break the ladder's leakage rules.
Never pick a config on these numbers over TEST.

No scipy: the Wilcoxon and McNemar p-values are computed exactly (normal approx. for n > 60).
"""

import ast
import io
import keyword
import math
import os
import tokenize

import backtrader as bt
import numpy as np
import pandas as pd

from .backtest import Backtester, call_with_timeout
from .baselines import BASELINES
from .codegen import extract_strategy, validate_code
from .config import COST_LEVELS  # noqa: F401  (re-exported; 'low' is the default reward's level)
from .ladder import append_csv, bootstrap_mean, wilson

# Calendar-year market regime of the TEST split (S&P 500: 2022 -19%, 2023 +24%, 2024 +23%).
REGIMES = {2022: 'bear', 2023: 'bull', 2024: 'bull'}

KEY = ['model', 'config', 'symbol', 'start', 'end', 'sample']
TRADED = ('profitable', 'loss')


def _load(df):
    return pd.read_csv(df) if isinstance(df, (str, os.PathLike)) else df.copy()


# ── 1. re-backtest at several cost levels ──────────────────────────────────────

def _run_code(code, backtester, symbol, start, end, commission, slippage, cash, timeout):
    if validate_code(code):
        raise ValueError('invalid code')
    strategy = extract_strategy(code)
    return call_with_timeout(backtester.run_detailed, timeout, strategy, symbol, start, end,
                             cash=cash, commission=commission, slippage=slippage)


def rebacktest(results, backtester: Backtester, cost_levels=None, baselines=('buy_and_hold',),
               out_csv=None, cash=10_000, timeout=30, include_no_trades=False):
    """Re-run every traded strategy in `results` (+ baselines per symbol/window) at each cost
    level. Returns one row per (strategy, cost level). Resumable when out_csv is given.

    The level matching the evaluation's RewardConfig.costs ('low' by default, 'none' for
    RewardConfig.original()) should reproduce the evaluation's numbers; status 'error' rows are strategies that raised or timed out on re-run."""
    cost_levels = cost_levels or COST_LEVELS
    df = _load(results)
    for col, default in (('model', ''), ('config', 'baseline'), ('sample', 1)):
        if col not in df:
            df[col] = default
    if 'end' not in df:
        raise ValueError("results need start/end columns (the evaluated window)")
    keep = TRADED + (('no_trades',) if include_no_trades else ())
    todo = df[df['status'].isin(keep) & df['strategy_code'].notna()][KEY + ['strategy_code']]

    # baseline rows: one per (symbol, window), model tag '_baseline'
    windows = df[['symbol', 'start', 'end']].drop_duplicates()
    base_rows = [{'model': '_baseline', 'config': name, 'symbol': w.symbol, 'start': w.start,
                  'end': w.end, 'sample': 1, 'strategy_code': BASELINES[name]}
                 for name in baselines for w in windows.itertuples()]
    todo = pd.concat([pd.DataFrame(base_rows), todo], ignore_index=True)

    done = set()
    if out_csv and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(map(tuple, prev[KEY + ['cost']].astype(str).values))
    rows = []
    total, i = len(todo) * len(cost_levels), 0
    for r in todo.itertuples(index=False):
        for cost, (c_bps, s_bps) in cost_levels.items():
            i += 1
            key = tuple(str(getattr(r, k)) for k in KEY) + (cost,)
            if key in done:
                continue
            rec = {k: getattr(r, k) for k in KEY}
            rec.update(cost=cost, commission_bps=c_bps, slippage_bps=s_bps)
            try:
                res, extras = _run_code(r.strategy_code, backtester, r.symbol, r.start, r.end,
                                        c_bps / 1e4, s_bps / 1e4, cash, timeout)
                rec.update(res._asdict(), **extras,
                           rerun_status=('profitable' if res.return_pct > 0 else 'loss')
                           if res.traded else 'no_trades')
            except Exception as e:  # incl. TimeoutError
                rec.update(rerun_status='error', error=str(e)[:200])
            rows.append(rec)
            if out_csv:
                append_csv(out_csv, rec)
            if i % 50 == 0:
                print(f"  rebacktest [{i}/{total}]")
    if out_csv:
        return pd.read_csv(out_csv)
    return pd.DataFrame(rows)


def with_bh(rb, baseline='buy_and_hold'):
    """Adds bh_* columns (buy-and-hold at the same symbol/window/cost) plus excess_annual,
    excess_return, beat_bh. Baseline rows themselves are dropped."""
    rb = rb.copy()
    bh = rb[(rb['model'] == '_baseline') & (rb['config'] == baseline)]
    year_cols = [c for c in rb if c.startswith('year_')]
    cols = ['avg_annual_return_pct', 'return_pct', 'max_drawdown_pct', 'sharpe_ratio',
            'volatility_pct'] + year_cols
    bh = bh[['symbol', 'start', 'end', 'cost'] + cols].rename(columns={c: f'bh_{c}' for c in cols})
    out = rb[rb['model'] != '_baseline'].merge(bh, on=['symbol', 'start', 'end', 'cost'], how='left')
    ok = out['rerun_status'].isin(TRADED + ('no_trades',))
    out['excess_annual'] = np.where(ok, out['avg_annual_return_pct'] - out['bh_avg_annual_return_pct'], np.nan)
    out['excess_return'] = np.where(ok, out['return_pct'] - out['bh_return_pct'], np.nan)
    out['beat_bh'] = ok & (out['excess_return'] > 0)
    return out


# ── 2. statistics helpers ──────────────────────────────────────────────────────

def signed_rank_test(d):
    """Two-sided Wilcoxon signed-rank test of median(d) == 0 (zeros dropped, average ranks
    for ties). Exact for n <= 60, normal approximation with tie correction above."""
    d = np.asarray(pd.Series(d).dropna(), dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {'n': 0, 'W': math.nan, 'p': math.nan}
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_plus = ranks[d > 0].sum()
    if n <= 60:
        r2 = np.rint(ranks * 2).astype(int)          # half-integer ranks -> integers
        dist = np.zeros(r2.sum() + 1)
        dist[0] = 1.0
        for r in r2:                                  # subset-sum distribution of W+
            dist[r:] += dist[:-r].copy()
        dist /= dist.sum()
        w = int(round(w_plus * 2))
        p = min(1.0, 2 * min(dist[:w + 1].sum(), dist[w:].sum()))
    else:
        mean = n * (n + 1) / 4
        _, counts = np.unique(ranks, return_counts=True)
        var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
        z = (w_plus - mean) / math.sqrt(var)
        p = math.erfc(abs(z) / math.sqrt(2))
    return {'n': n, 'W': w_plus, 'p': p}


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (NaNs kept)."""
    p = pd.Series(p, dtype=float)
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = (ok * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return p.where(p.isna(), adj.reindex(p.index))


def _ci(x):
    v, lo, hi = bootstrap_mean(x)
    return v, lo, hi


# ── 3. reports ─────────────────────────────────────────────────────────────────

def cost_report(rb, by=('model', 'config')):
    """One row per group x cost level, strategies vs buy-and-hold at the same cost.

    Rates (still_traded, profitable, beat_bh) are over all re-run strategies with Wilson CIs;
    means use bootstrap CIs. wilcoxon_p tests per-symbol mean excess return vs 0 across symbols
    (one value per symbol, so samples of the same symbol are not treated as independent)."""
    df = with_bh(rb)
    by = list(by)
    rows = []
    for key, g in df.groupby(by + ['cost'], sort=False):
        n = len(g)
        traded = g['rerun_status'].isin(TRADED)
        row = dict(zip(by + ['cost'], key), n=n)
        for name, k in (('still_traded', traded.sum()),
                        ('profitable', g['rerun_status'].eq('profitable').sum()),
                        ('beat_bh', g['beat_bh'].sum())):
            row[name], row[f'{name}_lo'], row[f'{name}_hi'] = wilson(int(k), n)
        for col in ('avg_annual_return_pct', 'excess_annual', 'excess_return'):
            row[col], row[f'{col}_lo'], row[f'{col}_hi'] = _ci(g[col])
        row['bh_annual'] = g['bh_avg_annual_return_pct'].mean()
        t = g[traded]
        if 'alpha_pct' in t:
            # Jensen's alpha vs buy-and-hold (%/yr), over strategies that still traded
            row['alpha_pct'], row['alpha_pct_lo'], row['alpha_pct_hi'] = _ci(t['alpha_pct'])
            row['median_beta'] = t['beta'].median()
            k = int((g['alpha_t'] > 1.96).sum()) if 'alpha_t' in g else 0
            row['alpha_sig'], row['alpha_sig_lo'], row['alpha_sig_hi'] = wilson(k, n)
        for col in ('sharpe_ratio', 'max_drawdown_pct', 'volatility_pct', 'turnover', 'n_trades'):
            row[f'median_{col}'] = t[col].median() if col in t else math.nan
        row['median_bh_drawdown'] = g['bh_max_drawdown_pct'].median()
        row['median_bh_volatility'] = g['bh_volatility_pct'].median()
        per_sym = g.groupby('symbol')['excess_return'].mean()
        row['n_symbols'] = len(per_sym)
        row['wilcoxon_p'] = signed_rank_test(per_sym)['p']
        rows.append(row)
    return pd.DataFrame(rows)


def per_asset(rb, cost='mid', by=('model', 'config')):
    """One row per group x symbol at one cost level: mean return / excess with bootstrap CIs
    across samples, beat-B&H rate (Wilson) and a Wilcoxon test of the sample excesses vs 0.
    p_bh is Benjamini-Hochberg adjusted within each group (many symbols = many tests)."""
    df = with_bh(rb)
    df = df[df['cost'] == cost]
    by = list(by)
    rows = []
    for key, g in df.groupby(by + ['symbol'], sort=False):
        row = dict(zip(by + ['symbol'], key), n=len(g))
        row['annual'], row['annual_lo'], row['annual_hi'] = _ci(g['avg_annual_return_pct'])
        row['bh_annual'] = g['bh_avg_annual_return_pct'].iloc[0]
        row['excess'], row['excess_lo'], row['excess_hi'] = _ci(g['excess_return'])
        row['beat_bh'], row['beat_bh_lo'], row['beat_bh_hi'] = wilson(int(g['beat_bh'].sum()), len(g))
        row['median_drawdown'] = g['max_drawdown_pct'].median()
        row['bh_drawdown'] = g['bh_max_drawdown_pct'].iloc[0]
        row['wilcoxon_p'] = signed_rank_test(g['excess_return'])['p']
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out['p_bh'] = out.groupby(by, group_keys=False)['wilcoxon_p'].apply(bh_adjust)
    return out


def year_report(rb, cost='mid', by=('model', 'config'), regimes=None):
    """One row per group x calendar year at one cost level: mean strategy vs B&H return for
    that year (bootstrap CI on the difference), beat rate, and Wilcoxon over symbols."""
    regimes = REGIMES if regimes is None else regimes
    df = with_bh(rb)
    df = df[(df['cost'] == cost) & df['rerun_status'].isin(TRADED + ('no_trades',))]
    by = list(by)
    years = sorted(int(c[5:]) for c in df if c.startswith('year_') and not c.startswith('bh_'))
    rows = []
    for key, g in df.groupby(by, sort=False):
        for y in years:
            s, b = g[f'year_{y}'], g[f'bh_year_{y}']
            m = s.notna() & b.notna()
            if not m.any():
                continue
            ex = (s - b)[m]
            row = dict(zip(by, key), year=y, regime=regimes.get(y, ''), n=int(m.sum()))
            row['annual'] = s[m].mean()
            row['bh_annual'] = b[m].mean()
            row['excess'], row['excess_lo'], row['excess_hi'] = _ci(ex)
            row['beat_bh'], row['beat_bh_lo'], row['beat_bh_hi'] = wilson(int((ex > 0).sum()), int(m.sum()))
            row['wilcoxon_p'] = signed_rank_test(ex.groupby(g.loc[m, 'symbol']).mean())['p']
            rows.append(row)
    return pd.DataFrame(rows)


def mcnemar(results, a, b, metric='valid', by='config'):
    """Exact McNemar test between configs a and b on a binary metric (valid / traded /
    profitable), paired by (symbol, sample). b10 = a succeeded and b failed."""
    from .ladder import add_indicators
    df = add_indicators(_load(results))
    if 'sample' not in df:
        df['sample'] = df.groupby([by, 'symbol']).cumcount() + 1
    piv = df[df[by].isin([a, b])].pivot_table(index=['symbol', 'sample'], columns=by,
                                               values=metric, aggfunc='first').dropna()
    if piv.empty:
        return {'a': a, 'b': b, 'metric': metric, 'n_pairs': 0}
    xa, xb = piv[a].astype(bool), piv[b].astype(bool)
    b10, b01 = int((xa & ~xb).sum()), int((~xa & xb).sum())
    n = b10 + b01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b10, b01) + 1)) / 2 ** n)
    return {'a': a, 'b': b, 'metric': metric, 'n_pairs': len(piv),
            'rate_a': xa.mean(), 'rate_b': xb.mean(), 'a_only': b10, 'b_only': b01, 'p': p}


def paired_wilcoxon(df, a, b, metric='excess_return', by='config', group='symbol'):
    """Wilcoxon signed-rank on per-symbol means of `metric` (b - a). Works on results
    (e.g. metric='reward_score') or on with_bh(rebacktest(...)) filtered to one cost."""
    per = df[df[by].isin([a, b])].groupby([group, by])[metric].mean().unstack(by).dropna(subset=[a, b])
    d = per[b] - per[a]
    t = signed_rank_test(d)
    return {'a': a, 'b': b, 'metric': metric, 'n_symbols': len(d),
            'median_delta': d.median() if len(d) else math.nan, 'W': t['W'], 'p': t['p']}


# ── 4. near-duplicate strategies ───────────────────────────────────────────────

_KNOWN = (set(dir(bt)) | set(dir(bt.indicators)) | set(dir(bt.Strategy)) | set(dir(__builtins__))
          | {'self', 'bt', 'params', 'p', 'data', 'datas', 'close', 'open', 'high', 'low',
             'volume', 'datetime', 'position', 'size', 'period', 'target', 'Strategy'})


def code_tokens(code):
    """Token sequence with comments/docstrings/formatting removed and user-chosen identifiers
    renamed in order of appearance (v0, v1, ...); backtrader API names and literals kept."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):    # drop docstrings
            body = getattr(node, 'body', None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                    and isinstance(getattr(body[0], 'value', None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        src = ast.unparse(tree)
    except (SyntaxError, ValueError):
        src = code
    names, out = {}, []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.NAME and not keyword.iskeyword(tok.string) and tok.string not in _KNOWN:
                out.append(names.setdefault(tok.string, f'v{len(names)}'))
            elif tok.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING):
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        out = src.split()
    return out


def _shingles(tokens, k=5):
    return {tuple(tokens[i:i + k]) for i in range(max(1, len(tokens) - k + 1))}


def duplicates(results, by=('model', 'config'), threshold=0.8, k=5, across_groups=False):
    """Exact and near-duplicate strategies within each group (or across all with
    across_groups=True). Near-duplicate = Jaccard similarity of normalised token 5-grams
    >= threshold; clusters are connected components.

    Returns (summary, pairs): summary has n, unique_exact, n_clusters, unique_share,
    largest_cluster_share, mean_similarity per group; pairs lists every near-duplicate pair."""
    df = _load(results)
    df = df[df['strategy_code'].notna()].reset_index(drop=True)
    by = [] if across_groups else [c for c in by if c in df]
    toks = [code_tokens(c) for c in df['strategy_code']]
    sh = [_shingles(t, k) for t in toks]
    canon = [' '.join(t) for t in toks]
    groups = df.groupby(by).groups if by else {'all': df.index}
    summ, pairs = [], []
    for key, idx in groups.items():
        idx = list(idx)
        parent = {i: i for i in idx}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        sims = []
        for x, i in enumerate(idx):
            for j in idx[x + 1:]:
                sim = len(sh[i] & sh[j]) / len(sh[i] | sh[j])
                sims.append(sim)
                if sim >= threshold:
                    parent[find(i)] = find(j)
                    pairs.append({**(dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {}),
                                  'row_a': i, 'row_b': j, 'symbol_a': df.at[i, 'symbol'],
                                  'symbol_b': df.at[j, 'symbol'], 'similarity': sim,
                                  'exact': canon[i] == canon[j]})
        n = len(idx)
        n_pairs = n * (n - 1) / 2
        clusters = pd.Series([find(i) for i in idx]).value_counts()
        row = dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {'group': 'all'}
        row.update(n=n, unique_exact=len({canon[i] for i in idx}), n_clusters=len(clusters),
                   unique_share=len(clusters) / n if n else math.nan,
                   largest_cluster_share=clusters.iloc[0] / n if n else math.nan,
                   # pairs with no shared shingle have similarity 0
                   mean_similarity=(sum(sims) / n_pairs) if n_pairs else math.nan)
        summ.append(row)
    return pd.DataFrame(summ), pd.DataFrame(pairs)


# ── 5. formatting ──────────────────────────────────────────────────────────────

def _fmt(v, lo, hi, pct=False, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '-'
    s = 100 if pct else 1
    return f"{v*s:.{1 if pct else nd}f} [{lo*s:.{1 if pct else nd}f}, {hi*s:.{1 if pct else nd}f}]"


def format_cost_report(rep, by=('model', 'config')):
    """Readable version of cost_report: 'value [lo, hi]' strings, rates in %."""
    by = list(by)
    out = rep[by + ['cost', 'n']].copy()
    for col in ('still_traded', 'profitable', 'beat_bh'):
        out[f'{col} %'] = [_fmt(*r, pct=True) for r in rep[[col, f'{col}_lo', f'{col}_hi']].values]
    out['annual %'] = [_fmt(*r) for r in rep[['avg_annual_return_pct', 'avg_annual_return_pct_lo',
                                               'avg_annual_return_pct_hi']].values]
    out['B&H annual %'] = rep['bh_annual'].round(2)
    out['excess return %'] = [_fmt(*r) for r in rep[['excess_return', 'excess_return_lo',
                                                      'excess_return_hi']].values]
    if 'alpha_pct' in rep:
        out['alpha %/yr'] = [_fmt(*r) for r in rep[['alpha_pct', 'alpha_pct_lo', 'alpha_pct_hi']].values]
        out['alpha t>1.96 %'] = [_fmt(*r, pct=True) for r in rep[['alpha_sig', 'alpha_sig_lo', 'alpha_sig_hi']].values]
        out['beta'] = rep['median_beta'].round(2)
    out['wilcoxon p'] = rep['wilcoxon_p'].map(lambda p: f"{p:.3g}" if not math.isnan(p) else '-')
    for col, label in (('median_max_drawdown_pct', 'MDD'), ('median_bh_drawdown', 'B&H MDD'),
                       ('median_volatility_pct', 'vol %'), ('median_turnover', 'turnover/yr'),
                       ('median_n_trades', 'trades'), ('median_sharpe_ratio', 'sharpe')):
        out[label] = rep[col].round(2)
    return out


def full_report(results, backtester: Backtester, cost_levels=None, cost='mid', out_dir=None,
                by=('model', 'config'), out_csv=None, **kw):
    """Run everything; returns a dict of DataFrames and (with out_dir) writes CSV + markdown."""
    from .ladder import markdown_table
    rb = rebacktest(results, backtester, cost_levels, out_csv=out_csv, **kw)
    by = [c for c in by if c in rb]
    tables = {
        'rebacktest': rb,
        'cost': cost_report(rb, by),
        'per_asset': per_asset(rb, cost, by),
        'by_year': year_report(rb, cost, by),
    }
    tables['duplicates'], tables['duplicate_pairs'] = duplicates(results, by)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, t in tables.items():
            t.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
        with open(os.path.join(out_dir, 'report.md'), 'w') as f:
            f.write("## Transaction costs vs buy-and-hold\n\n"
                    + markdown_table(format_cost_report(tables['cost'], by)))
            f.write(f"\n## By year (cost={cost})\n\n" + markdown_table(tables['by_year'].round(3)))
            f.write(f"\n## Per asset (cost={cost})\n\n" + markdown_table(tables['per_asset'].round(3)))
            f.write("\n## Near-duplicates\n\n" + markdown_table(tables['duplicates'].round(3)))
        print(f"Wrote {len(tables)} tables + report.md to {out_dir}/")
    return tables
