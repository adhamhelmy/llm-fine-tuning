# Configurations tested

Record of the ladder configurations tried before the generation config was fixed. The raw
result files (`ladder_results/qwen3.8-27b-mlx/{dev,test}.csv`, `selection.json`) were deleted
after this summary was written, and so was the selection code (`ladder.STAGES`,
`stage_configs`, `select_best`).

**Final config:** `LadderConfig(api_reference=True, repair_rounds=3)` (`CONFIGS` in
`scripts/main.py`).

## How they were tested

- Model: `qwen3.8:27b-mlx` (Ollama, thinking on by default), temperature 1.0, 1536 new tokens.
- Greedy stage-wise selection: each stage's variants were added on top of the best config so far,
  evaluated on **dev** (2020–2021), and the variant with the highest profitable rate was kept.
- Only **AAPL** with 1 sample per config on dev (2 on test). These are smoke-level numbers, not
  evidence for or against any config.

Variants per stage:

| Stage | Variants (added to the best config so far) |
|---|---|
| Prompt | baseline; 2 few-shot examples |
| Context | none; API reference; market summary; RAG (k = 3); all three |
| Harness | none; repair ≤ 3 rounds |
| Loop | none; refine 2 rounds |
| Probe (diagnostic) | none; anonymized ticker/dates |

Selected per stage (`selection.json`): prompt `examples=2` → context: no change → harness
`examples=2,repair_rounds=3` → loop `examples=2,repair_rounds=3,refine_rounds=2`.

## Dev results (2020-01-01 – 2021-12-31, AAPL, 1 sample)

| Config | Status | LLM calls | Seconds | Avg annual % |
|---|---|---|---|---|
| baseline | missing_methods | 1 | 105 | |
| examples=2 | invalid_code | 1 | 223 | |
| examples=2,api_reference | missing_methods | 1 | 212 | |
| examples=2,market_summary | missing_methods | 1 | 189 | |
| examples=2,rag_k=3 | missing_methods | 1 | 196 | |
| examples=2,api_reference,market_summary,rag_k=3 | missing_methods | 1 | 182 | |
| examples=2,repair_rounds=3 | exception | 4 | 475 | |
| examples=2,repair_rounds=3,refine_rounds=2 | profitable | 5 | 1148 | 5.7 |

Buy-and-hold AAPL on dev: 54.2% avg annual.

## Test results (2022-01-01 – 2024-12-31, AAPL, 2 samples)

| Config | Statuses | Mean LLM calls | Mean seconds | Mean avg annual % |
|---|---|---|---|---|
| baseline | missing_methods, no_trades | 1.0 | 298 | 0.0 |
| examples=2 | missing_methods, no_trades | 1.0 | 303 | 0.0 |
| examples=2,repair_rounds=3 | profitable, profitable | 2.5 | 383 | 6.5 |
| examples=2,repair_rounds=3,refine_rounds=2 | profitable, profitable | 5.5 | 937 | 10.1 |

Buy-and-hold AAPL on test: 15.9% avg annual.

## Notes

- Most failures without repair were `missing_methods` ("no class found"). The likely cause is the
  thinking model using up the 1536-token budget before writing the code (generations in the Ollama
  log run to the cap). Repair rounds recovered most of them.
- The final config (`api_reference,repair_rounds=3`) was not the one the greedy selection picked
  (`examples=2,repair_rounds=3[,refine_rounds=2]`).
