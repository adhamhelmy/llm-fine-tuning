"""Model-agnostic evaluation: any `generate(prompt) -> str` callable (Unsloth, MLX, Ollama...)."""

from .backtest import Backtester
from .codegen import extract_function
from .prompt import make_prompt
from .rewards import save_strategy, score_strategy


def evaluate_sample(generate, backtester: Backtester, symbol, start, end,
                    prompt=None, timeout=10, save_dir=None, cash=10_000):
    """Generate one strategy for `symbol`, score it and return a flat result record.
    save_dir: if set, profitable strategies are saved there (with plots)."""
    prompt = prompt or make_prompt(symbol, start, end)
    rec = {
        'symbol': symbol, 'status': None, 'reward_score': None,
        'return_pct': None, 'sharpe_ratio': None,
        'avg_annual_return_pct': None, 'max_drawdown_pct': None,
        'strategy_code': None, 'error': None,
    }
    try:
        code = extract_function(generate(prompt))
    except Exception as e:  # generation itself failed
        rec.update(status='exception', reward_score=-2, error=f"generate: {str(e)[:200]}")
        return rec

    score = score_strategy(code, backtester, symbol, start, end, cash=cash, timeout=timeout)
    rec.update(strategy_code=code, status=score.status, reward_score=score.reward, error=score.error)
    if score.result is not None:
        rec.update(score.result._asdict())
    if save_dir and score.status == 'profitable':
        save_strategy(code, score.result, symbol, start, end, save_dir, backtester)
    return rec


def evaluate_symbol(generate, backtester: Backtester, symbol, start, end, n_samples,
                    prompt=None, timeout=10, save_dir=None, cash=10_000):
    """Generate and score n_samples candidates for one symbol."""
    results = []
    for i in range(n_samples):
        rec = evaluate_sample(generate, backtester, symbol, start, end, prompt, timeout, save_dir, cash)
        rec['sample'] = i + 1
        if rec['error'] and rec['status'] == 'exception':
            print(f"    [{symbol} s{i+1}] Exception: {rec['error'][:100]}")
        results.append(rec)

    print(f"  {symbol}: {[r['status'] for r in results]}  scores={[r['reward_score'] for r in results]}")
    return results


def summarize(df):
    """Print status breakdown and stats for a DataFrame of evaluation records."""
    n = len(df)
    print("=== Status breakdown ===")
    print(df["status"].value_counts().to_string())

    traded = df[df["status"].isin(["profitable", "loss"])]
    print(f"\nStrategies that ran:    {len(df[df['status'].isin(['profitable', 'loss', 'no_trades'])])}/{n}")
    print(f"Strategies that traded: {len(traded)}/{n}")

    if len(traded):
        print("\n=== Stats (strategies that traded) ===")
        metrics = ["return_pct", "avg_annual_return_pct", "sharpe_ratio", "max_drawdown_pct"]
        print(traded[metrics].describe().to_string())
        print(f"\nPositive-return rate:   {(traded['return_pct'] > 0).mean():.1%}")
