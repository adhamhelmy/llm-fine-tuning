# 3 Methodology (revised draft)

> Draft structured around the LLM-engineering ladder (prompt → context → harness → loop → graph),
> as requested by the reviewers. Every design element named here is implemented in the `trading_rl`
> package (the module is given in brackets) and evaluated by `unsloth/ladder_ablation.ipynb`.
> **[TBD]** marks numbers that come from the ablation runs. Exact prompt texts are in `docs/paper/prompts/`
> (static variants) and `ladder_results/<model>/prompts/` (all variants, written by each run).

## 3.1 Task and overview

Given a stock *s* and a backtest window *[t₀, t₁]*, the LLM must output Python source code for a
`backtrader` strategy class (`class Strategy(bt.Strategy)` with `__init__` and `next`). The code is
executed on historical daily bars for *s* over *[t₀, t₁]* with $10,000 of initial cash. The outcome
is the strategy's return, Sharpe ratio and maximum drawdown.

We treat strategy generation as a stack of engineering layers and measure each layer's contribution
before adding the next:

| Stage | Question | What we vary |
|---|---|---|
| **Prompt** | How is the task specified? | rules and skeleton, few-shot examples, explicit reasoning |
| **Context** | What information does the model get? | Backtrader API reference, pre-window market statistics, retrieved strategy knowledge (RAG) |
| **Harness** | What checks and tools surround a single generation? | extraction, static validation, sandboxed execution, a dry run with error feedback and repair |
| **Loop** | How does the system improve over iterations? | inference-time refinement on in-sample metrics; **GRPO** fine-tuning (training-time loop) |
| **Graph** | Does decomposing the task into roles help? | analyst node (specification) → coder node (implementation) |

The central claim is conditional: GRPO is evaluated **on top of the best configuration found for the
lower stages**, not against a weak prompt-only baseline.

## 3.2 Experimental protocol

**Data.** Daily split- and dividend-adjusted bars from Alpaca's historical data API for a 30-stock
US large-cap universe (Dow-30-like; `config.TESTER_SYMBOLS`). The GRPO training set uses the
38-symbol `TRAINING_SYMBOLS`.

**Time splits** (`config.py`). The splits do not overlap and are used for different purposes:

| Split | Window | Use |
|---|---|---|
| Train | 2016-01-01 – 2019-12-31 | GRPO training rewards |
| Dev | 2020-01-01 – 2021-12-31 | selecting the best option at every ladder stage |
| Test | 2022-01-01 – 2024-12-31 | the reported numbers only; never used for any choice |

**Leakage controls.**
1. *Feedback window.* Whenever the system uses backtest feedback (the harness dry run, loop
   refinement), that feedback comes from the 2 years **before** the evaluated window
   (`ladder.feedback_window`). The evaluated window is backtested exactly once, on the final
   strategy.
2. *Context.* Market statistics are computed only from bars before *t₀* (`context.market_stats`). The
   RAG corpus contains no dated market facts.
3. *Memorization probe.* LLMs may remember price histories from pre-training. We re-run the final
   configuration with the ticker and dates removed from the prompt (`anonymize`). A large drop in
   performance would indicate reliance on memorized hindsight rather than on strategy design.

**Sampling.** Temperature 1.0 and at most 1,536 new tokens. Dev uses 3 samples × 30 symbols = 90 per
configuration; test uses 5 × 30 = 150 per rung. **[adjust to the final budget]**

**Outcome taxonomy** (`rewards.score_strategy`). Every sample gets exactly one status:
`missing_methods` (no class, or no `__init__`/`next`), `invalid_code` (syntax error or a non-stdlib
import), `exception` (runtime error), `timeout` (>10 s), `no_trades`, `loss` or `profitable`.

**Metrics** (`ladder.report`). All rates are computed over **all** samples, so a crash counts as a
failure:
- *valid*: ran without error;
- *traded*: placed at least one trade;
- *profitable*: return > 0;
- *beats B&H*: average annual return above buy-and-hold on the same stock and window.

Rates are reported with Wilson 95% CIs. For strategies that traded, we also report the mean average
annual return and the excess over buy-and-hold, with bootstrap 95% CIs, and the median Sharpe ratio
and maximum drawdown.

**Baselines** (`baselines.py`). Buy-and-hold and an SMA 50/200 crossover, run through the same
backtester.

**Statistical comparison** (`ladder.paired_delta`). Differences between two configurations are paired
by symbol: we average each metric per symbol, bootstrap over symbols (5,000 resamples) and report the
mean difference, its 95% CI and a two-sided bootstrap p-value.

**Stage-wise selection.** Starting from the baseline prompt, each stage evaluates its variants
(Table 1) *on top of the best configuration so far*, on the dev split, and keeps the variant with the
highest profitable rate (ties are broken by mean reward, then fewer LLM calls). This greedy
forward selection means each stage is only run after the stages below it have been optimized. The
selected configuration after each stage (a *rung*) is then evaluated once on test.

**Table 1 – Variants per stage** (`ladder.STAGES`)

| Stage | Variants (added to the best config so far) |
|---|---|
| Prompt | baseline; compact (no skeleton); 2 few-shot examples; plan-then-code; examples + plan |
| Context | none; API reference; market summary; RAG (k = 3); all three |
| Harness | none; repair ≤ 1 round; repair ≤ 3 rounds |
| Loop (inference) | none; refine 2 rounds; refine 4 rounds |
| Graph | single agent; analyst → coder |

## 3.3 Stage 1 – Prompt engineering (`prompt.py`)

**Baseline prompt (P0).** The full text is in `prompts/baseline.txt`. It consists of:
- a header naming the stock and window;
- the framework contract (the class is passed to `run_backtest(StrategyClass, …)`);
- ten strict rules, each targeting a failure mode we observed:
  - output only one class (the output is parsed automatically);
  - the class must be named `Strategy`;
  - no imports (the sandbox provides only `bt`);
  - no external data;
  - it must work on a single symbol;
  - indicators are created in `__init__`;
  - trading logic goes in `next`;
  - only four order methods may be used;
  - no plotting or logging;
  - no lookahead;
- a class skeleton showing the output format.

**Variants.**
- *Compact*: the same rules without the skeleton. This was the prompt used in our earlier
  cross-model comparison, and is kept for comparability.
- *Few-shot*: two example strategies (EMA crossover; RSI mean reversion with a trend filter;
  `context.EXAMPLES`). They are presented as format and API illustrations, with an instruction not
  to copy them. All examples are checked to run by the test suite.
- *Plan-then-code*: the model first reasons inside `<think>` tags about the market behaviour to
  exploit and about how to avoid errors. The reasoning is removed before code extraction.

**Result.** Table 2, prompt block **[TBD]**. We report which variant was selected and how much it
changed the valid and profitable rates relative to P0.

## 3.4 Stage 2 – Context engineering (`context.py`, `knowledge.py`)

In our original experiments the dominant failure of non-code models was runtime exceptions, for
example 14–17 of 20 samples for Ministral-3 3B/8B/14B (`models/*/Summary.csv`). This points to a
lack of API knowledge rather than of trading knowledge. We therefore separate *technical* context
from *financial* context:

- **Backtrader API reference.** A compact cheat sheet whose indicator section is generated
  automatically from the installed backtrader version. It lists the exact class name, default
  parameters and output line names for 18 commonly used indicators, so it cannot be out of date. It
  also lists idioms and known pitfalls: `[0]` vs `[-1]` indexing, comparing values instead of line
  objects, crossovers, order semantics, the unavailability of TA-Lib, warm-up, and attribute names
  that shadow `Strategy` methods (e.g. `self.stop`, which silently breaks the backtest).
- **Market summary.** Statistics of the stock over the 2 years *before t₀*: annualized return and
  volatility, maximum drawdown, share of days above the 200-day SMA, SMA 50/200 state, lag-1 return
  autocorrelation and average true range. From these we derive regime tags (up/down/range-bound;
  low/moderate/high volatility; momentum/mean-reverting).
- **Financial RAG.** A curated corpus of 12 documents on strategy families and when they tend to
  work (trend following, time-series momentum, RSI/Bollinger mean reversion, Donchian breakouts,
  volatility-scaled sizing, ATR stops, regime filters, benchmark awareness, overfitting). The
  corpus is retrieved with BM25, using the stock's regime tags as the query, and the top k = 3
  documents are inserted.
  - The corpus is deliberately free of dated facts, so retrieval cannot leak information about the
    test period.
  - `KnowledgeBase.from_dir` accepts any larger corpus, e.g. textbooks or papers. We discuss news
    and fundamentals retrieval, which would need point-in-time data to avoid lookahead, as future
    work.

**Result.** Table 2, context block **[TBD]**.

## 3.5 Stage 3 – Harness engineering (`codegen.py`, `backtest.py`, `rewards.py`, `ladder.py`)

1. **Extraction.** Reasoning blocks (`<think>…</think>`) are removed. The class is taken from the
   first fenced code block, or from the raw text starting at `class Strategy(bt.Strategy):`.
2. **Static validation.** The code is checked for the required methods and parsed with Python's
   `ast`, and any import outside the standard library is rejected.
3. **Execution.** The code is `exec`'d with only `bt` in scope, then backtested with Backtrader:
   - $10,000 initial cash and no commission;
   - Sharpe, annual-return and drawdown analyzers;
   - a 10-second wall-clock limit (SIGALRM, with a thread fallback when not on the main thread).
4. **Dry run and repair** (the variable in this stage). The candidate is backtested on the feedback
   window. If it fails, the model receives its previous code and a precise error message, e.g.
   `ZeroDivisionError (line 12): …` (the line number refers to the generated code), or "the strategy
   never placed a trade". It then produces a corrected class. This is repeated for up to R rounds
   (R ∈ {0, 1, 3}); the template is in `prompts/_feedback_templates.txt`.

**Result.** Table 2, harness block **[TBD]**. The expected effect is mainly on the valid rate. Its
cost is reported as the mean number of LLM calls.

## 3.6 Stage 4 – Loop engineering

### 3.6.1 Inference-time refinement (`ladder.generate_strategy`)

Once a candidate trades on the feedback window, the model receives the in-sample metrics (return,
average annual return, Sharpe, drawdown), buy-and-hold's return over the same period, and an
instruction to improve out-of-sample risk-adjusted return without overfitting. After up to N rounds
(N ∈ {0, 2, 4}), the candidate with the best **in-sample** reward is kept; test data is never used
for this choice. This is the non-parametric counterpart of RL: it uses the same feedback signal
without updating any weights.

### 3.6.2 GRPO fine-tuning (`model.py`, `rewards.py`, `dataset.py`)

- **Training prompts.** Built with the **selected prompt and context configuration**
  (`ladder.prompt_fn(best)`), one per training symbol, over the train split. Harness and loop steps
  happen at inference time and are not part of the training prompt.
- **Reward.** The same outcome taxonomy, mapped to a scalar:

  | Status | Reward |
  |---|---|
  | missing_methods | −10 |
  | invalid_code | −3 |
  | exception or timeout | −2 |
  | no_trades | −1 |
  | loss | 0 |
  | profitable | max(average annual return %, 1) |

  The ordering encodes a curriculum: first produce a class, then valid code, then code that runs,
  then code that trades, then code that makes money.
- **Optimization.** Unsloth with a 4-bit base model and LoRA (r = 32, α = 64, applied to the
  attention and MLP projections), via TRL's `GRPOTrainer`:
  - learning rate 5·10⁻⁶ (linear schedule, 10% warm-up), AdamW-8bit, weight decay 0.01;
  - per-device batch 1 × gradient accumulation 8;
  - **G = [2 → revise, see below] generations per prompt**, sampling temperature 1.0;
  - [steps, model, hardware].
- **Evaluation.** The RL model is evaluated **at the same rungs** as its base model. Its gain is
  reported as the paired difference to the base model at the best non-RL configuration.

**Result.** Table 3 **[TBD]**.

## 3.7 Stage 5 – Graph engineering (`ladder.build_analyst_prompt`)

A two-node graph:
1. The **analyst** node receives the same context (minus code examples) but no coding rules. It
   writes a specification of at most 150 words: market hypothesis, indicators with parameters, entry
   and exit rules, sizing and risk.
2. The **coder** node receives the selected coding prompt plus that specification, followed by the
   same harness and loop.

Richer graphs (e.g. a risk-critic node, or several competing analysts) are left to future work.

**Result.** Table 2, graph block **[TBD]**.

## 3.8 Threats to validity

- **Memorization of price history:** addressed by the anonymized probe (§3.2).
- **Selection bias:** every choice is made on dev; test is used once per rung.
- **Multiple comparisons:** we report CIs and paired tests for adjacent rungs only. **[Optionally
  apply a Holm correction.]**
- **Market frictions:** no commissions or slippage; long-only strategies are typical. **[Consider
  adding 5–10 bps per trade.]**
- **Universe:** US large caps, 2016–2024, daily bars only.
- **Compute:** greedy selection does not explore every combination of stages. Interactions between
  stages are only partly covered, by the "all three" context variant.

---

**Table 2 – Ladder on the test split (base model)** — from `ladder_results/<model>/test_report.md`
**[TBD]**

**Table 3 – Base vs. GRPO at identical rungs** **[TBD]**
