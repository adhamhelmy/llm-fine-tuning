import subprocess, sys

def pip_install(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)

pip_install(
    "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git",
    "backtrader",
    "alpaca-trade-api",
    "trl==0.22.2",
    "huggingface_hub",
    "trackio",
)

import os
import re
import json
import shutil
import random
from datetime import datetime

import torch
import backtrader as bt
from alpaca_trade_api.rest import REST, TimeFrame
from datasets import Dataset
from unsloth import FastVisionModel, execute_with_time_limit, check_python_modules
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer

# ── Kaggle secrets ────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    ALPACA_API_KEY    = _secrets.get_secret("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = _secrets.get_secret("ALPACA_SECRET_KEY")
    HF_TOKEN          = _secrets.get_secret("HF_TOKEN")
except Exception:
    ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
    HF_TOKEN          = os.environ.get("HF_TOKEN", "")

MODEL_NAME   = "unsloth/Ministral-8B-Instruct-2410"
HF_REPO      = "adhamhelmy/ministral-8b-trading-grpo"
TRAIN_STEPS  = 500
MAX_SEQ_LEN  = 2048

# ── Model ─────────────────────────────────────────────────────────────────────

class Unsloth:
    def __init__(self, model_name, lora_rank=32, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True):
        self.max_seq_length = max_seq_length

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=False,
        )
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        prompt_tokens = self.tokenizer(
            Data._make_prompt("AAPL"), return_tensors="pt"
        ).input_ids.shape[-1]
        self.max_prompt_length = prompt_tokens + 1

    def train(self, steps, reward_functions, dataset):
        max_completion_length = self.max_seq_length - self.max_prompt_length

        args = GRPOConfig(
            temperature=1.0,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_steps=max(1, int(0.1 * steps)),
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_generations=2,
            max_prompt_length=self.max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=steps,
            save_steps=steps,
            push_to_hub=bool(HF_TOKEN),
            hub_model_id=HF_REPO if HF_TOKEN else None,
            hub_strategy="end",
            report_to="trackio",
            run_name="ministral-8b-trading-grpo-500steps",
            output_dir="/kaggle/working/outputs",
        )

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.max_length = None

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_functions,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()

        self.model.save_pretrained("/kaggle/working/grpo_saved_lora")
        self.tokenizer.save_pretrained("/kaggle/working/grpo_saved_lora")
        print("LoRA saved to /kaggle/working/grpo_saved_lora")

        if HF_TOKEN:
            self.model.push_to_hub(HF_REPO, token=HF_TOKEN, save_method="lora")
            self.tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
            print(f"Pushed to https://huggingface.co/{HF_REPO}")


# ── Backtrader wrapper ────────────────────────────────────────────────────────

class Backtrader:
    _instance  = None
    _data_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.rest_api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY,
                             "https://paper-api.alpaca.markets")
        self._prefetch_bars()

    def _get_bars(self, symbol, timeframe, start, end):
        key = (symbol, str(timeframe), start, end)
        if key not in Backtrader._data_cache:
            Backtrader._data_cache[key] = self.rest_api.get_bars(
                symbol, timeframe, start, end, adjustment="all"
            ).df
        return Backtrader._data_cache[key]

    def _prefetch_bars(self):
        print(f"Pre-fetching bar data for {len(Data.SYMBOLS)} symbols...")
        for i, symbol in enumerate(Data.SYMBOLS, 1):
            self._get_bars(symbol, TimeFrame.Day, Data.START, Data.END)
            print(f"  [{i}/{len(Data.SYMBOLS)}] {symbol} ✓")
        print("Done prefetching.")

    def _run_backtest(self, strategy, symbol, start, end, cash=10000):
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(cash)
        cerebro.addstrategy(strategy)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe",
                            timeframe=bt.TimeFrame.Days, compression=1,
                            riskfreerate=0.05, annualize=True)
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annual")
        cerebro.addanalyzer(bt.analyzers.DrawDown,     _name="drawdown")

        bars = self._get_bars(symbol, TimeFrame.Day, start, end)
        cerebro.adddata(bt.feeds.PandasData(dataname=bars, name=symbol))

        initial = cerebro.broker.getvalue()
        results = cerebro.run()
        final   = cerebro.broker.getvalue()
        ret     = (final / initial - 1) * 100

        strat          = results[0]
        sharpe         = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        annual_returns = strat.analyzers.annual.get_analysis()
        avg_annual     = (sum(annual_returns.values()) / len(annual_returns) * 100) if annual_returns else 0.0
        max_dd         = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]

        print(f"  return={ret:.2f}% sharpe={sharpe} annual={avg_annual:.2f}% dd={max_dd:.2f}%")
        return ret, sharpe, avg_annual, max_dd

    @execute_with_time_limit(10)
    def _timed_backtest(self, strategy, symbol, start, end, cash):
        return self._run_backtest(strategy, symbol, start, end, cash)

    def execute_strategy(self, strategy, symbol, start, end, cash=10000):
        return self._timed_backtest(strategy, symbol, start, end, cash)


# ── Dataset ───────────────────────────────────────────────────────────────────

class Data:
    SYMBOLS = [
        "AAPL", "AMGN", "AXP",  "BA",   "CAT",
        "CRM",  "CSCO", "CVX",  "DIS",  "DOW",
        "GS",   "HD",   "HON",  "IBM",  "JNJ",
        "JPM",  "KO",   "MCD",  "MMM",  "MRK",
        "MSFT", "NKE",  "NVDA", "PG",   "TRV",
        "UNH",  "V",    "VZ",   "WBA",  "WMT",
        "AMZN", "COIN", "GE",   "GOOGL","NFLX",
        "NIO",  "TSLA", "UVV",
    ]
    START = "2016-01-01"
    END   = "2024-12-31"

    @staticmethod
    def _make_prompt(symbol):
        return f"""
Create a trading strategy for {symbol} from {Data.START} to {Data.END} that is fully compatible with the following backtesting setup:

- Framework: Backtrader
- Strategy must subclass bt.Strategy
- The strategy will be passed directly into: run_backtest(StrategyClass, symbols, start, end, timeframe, cash)

STRICT RULES:
1. Output ONLY a single Python class definition (no explanations, no markdown, no comments outside the class).
2. The class MUST be named Strategy.
3. Do NOT include imports (bt is already available).
4. Do NOT reference external data, files, APIs, or indicators outside Backtrader.
5. The strategy MUST work for Single-symbol strategies.
6. All indicators must be created in __init__.
7. Trading logic must be implemented in next().
8. Orders must use only: self.buy(), self.sell(), self.close(), self.order_target_percent()
9. No plotting, printing, logging, or analyzers.
10. Strategy must be deterministic and backtest-safe (no lookahead bias).

OUTPUT FORMAT:
Return ONLY the Python class exactly like this structure:

class Strategy(bt.Strategy):

    params = dict(
        # parameters here
    )

    def __init__(self):
        # indicator definitions

    def next(self):
        # trading logic

DO NOT output anything else.
        """.strip()

    def __init__(self):
        records = [
            {"prompt": [{"role": "user", "content": self._make_prompt(sym)}], "answer": 0}
            for sym in self.SYMBOLS
        ]
        self.dataset = Dataset.from_list(records)

    def get_dataset(self):
        return self.dataset


# ── Reward functions ──────────────────────────────────────────────────────────

class RewardFunctions:
    RED   = "\033[91m"
    RESET = "\033[0m"
    _step_count = 0

    @staticmethod
    def extract_function(text):
        if text.count("```") >= 2:
            first  = text.find("```") + 3
            second = text.find("```", first)
            fx = text[first:second].strip().removeprefix("python\n")
            fx = fx[fx.find("class Strategy"):]
            if fx.startswith("class Strategy(bt.Strategy):"):
                return fx
        idx = text.find("class Strategy(bt.Strategy):")
        return text[idx:] if idx != -1 else None

    @staticmethod
    def function_works(function):
        if function is None:
            return False
        ok, info = check_python_modules(function)
        return not (ok is False or "error" in info)

    @staticmethod
    def has_required_functions(text):
        return (bool(re.search(r"def\s+__init__\s*\([^)]*\)\s*:", text)) and
                bool(re.search(r"def\s+next\s*\([^)]*\)\s*:", text)))

    @staticmethod
    def extract_trading_parameters(prompt):
        match = re.search(r"trading strategy for (\w+) from (\S+) to (\S+)", prompt)
        if match:
            return match.group(1), match.group(2), match.group(3)
        raise ValueError(f"Could not extract trading parameters from prompt: {prompt[:100]}")

    @staticmethod
    def extract_strategy(func):
        namespace = {"bt": bt}
        exec(func, namespace)
        return namespace["Strategy"]

    @staticmethod
    def save_strategy(function, ret, sharpe, avg_annual, max_dd, symbol, start, end):
        save_dir = "/kaggle/working/successful_strategies"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_dir = os.path.join(save_dir, f"{timestamp}_{symbol}")
        os.makedirs(strategy_dir, exist_ok=True)

        with open(os.path.join(strategy_dir, "strategy.py"), "w") as f:
            f.write(function)

        stats = {
            "symbol": symbol, "start": start, "end": end,
            "return_pct": round(ret, 4),
            "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
            "avg_annual_return_pct": round(avg_annual, 4),
            "max_drawdown_pct": round(max_dd, 4),
        }
        with open(os.path.join(strategy_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Strategy saved → {strategy_dir}/")

    @staticmethod
    def strategy_succeeds(completions, prompts, **kwargs):
        scores    = []
        backtrader = Backtrader()

        prompt_text = prompts[0][-1]["content"]
        symbol, start, end = RewardFunctions.extract_trading_parameters(prompt_text)
        RewardFunctions._step_count += 1
        print("=" * 50)
        print(f"Step {RewardFunctions._step_count} | {symbol} {start}→{end}")

        for completion in completions:
            response = completion[0]["content"]
            function = RewardFunctions.extract_function(response)

            if not RewardFunctions.has_required_functions(function or ""):
                scores.append(-10)
                continue

            if not RewardFunctions.function_works(function):
                scores.append(-3)
                continue

            try:
                strategy = RewardFunctions.extract_strategy(function)
                ret, sharpe, avg_annual, max_dd = backtrader.execute_strategy(
                    strategy, symbol=symbol, start=start, end=end
                )

                if ret == 0 and sharpe is None:
                    scores.append(-1)
                    continue

                if ret > 0:
                    print(function)
                    RewardFunctions.save_strategy(function, ret, sharpe, avg_annual, max_dd, symbol, start, end)
                    scores.append(max(avg_annual, 1))
                else:
                    scores.append(0)

            except TimeoutError:
                print("Timeout")
                scores.append(-2)
            except Exception as e:
                print(f"{RewardFunctions.RED}Exception: {str(e)[:100]}{RewardFunctions.RESET}")
                scores.append(-2)

        print(scores)
        return scores


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data  = Data()
    model = Unsloth(model_name=MODEL_NAME)
    model.train(
        steps=TRAIN_STEPS,
        reward_functions=[RewardFunctions.strategy_succeeds],
        dataset=data.get_dataset(),
    )
