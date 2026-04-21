# unsloth
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer
from unsloth import execute_with_time_limit
from unsloth import check_python_modules
import plotly.graph_objects as go

# backtrader
import backtrader as bt

# strategies folder import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
from IPython.display import Image, display

# data
from datasets import Dataset

# reward functions
import re
import json
import shutil
from datetime import datetime

HF_TOKEN = ''
ALPACA_API_KEY = ''
ALPACA_SECRET_KEY = ''


def plot_training_rewards(log_history):
    entries = [e for e in log_history if "reward" in e]
    steps   = [e["step"]   for e in entries]
    rewards = [e["reward"] for e in entries]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=rewards,
        mode='lines+markers',
        name='Reward',
        line=dict(color='royalblue', width=2),
        marker=dict(size=6),
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.update_layout(
        title='Training Reward vs Steps',
        xaxis_title='Step',
        yaxis_title='Mean Reward',
        hovermode='x unified',
        template='plotly_dark',
    )
    fig.show()


class Unsloth:

    def __init__(self, model_name: str, lora_rank: int = 32, max_seq_length: int = 4096, load_in_4bit=True, fast_inference=False, max_prompt_length=512):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        self.load_in_4bit = load_in_4bit
        self.fast_inference = fast_inference

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            fast_inference=self.fast_inference,
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    def generate(self, inputs: str):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": inputs.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )

        output = self.model.generate(
            **self.tokenizer(images=None, text=text, return_tensors="pt").to("cuda"),
            temperature=1.0,
            max_new_tokens=self.max_seq_length - self.max_prompt_length,
            streamer=TextStreamer(self.tokenizer, skip_prompt=True),
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _hf_repo_name(self, steps: int) -> str:
        base = self.model_name.split("/")[-1].lower()
        return f"{base}-v1-{steps}"

    def train(self, steps: int, reward_functions: list, dataset: torch.utils.data.Dataset, hf_token: str = None):
        max_prompt_length = self.max_prompt_length + 1
        max_completion_length = self.max_seq_length - max_prompt_length

        training_args = GRPOConfig(
            temperature=1.0,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=2,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=steps,
            save_steps=steps,
            report_to="none",
            output_dir="outputs",
        )

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        plot_training_rewards(trainer.state.log_history)

        self.model.save_pretrained("grpo_saved_lora")
        self.tokenizer.save_pretrained("grpo_saved_lora")

        if hf_token:
            repo_name = self._hf_repo_name(steps)
            print(f"Pushing LoRA adapter to HuggingFace: {repo_name} ...")
            self.model.push_to_hub(repo_name, token=hf_token, save_method="lora")
            self.tokenizer.push_to_hub(repo_name, token=hf_token)
            print(f"Saved to https://huggingface.co/{repo_name}")


class Backtrader:
    _instance = None
    _data_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.api_key = ALPACA_API_KEY
        self.api_secret = ALPACA_SECRET_KEY
        self.base_url = "https://paper-api.alpaca.markets"
        self.rest_api = REST(self.api_key, self.api_secret, self.base_url)
        self.load_bars()

    def _get_bars(self, symbol, timeframe, start, end):
        key = (symbol, str(timeframe), start, end)
        if key not in Backtrader._data_cache:
            Backtrader._data_cache[key] = self.rest_api.get_bars(symbol, timeframe, start, end, adjustment='all').df
        return Backtrader._data_cache[key]

    def load_bars(self, symbols=None, start=None, end=None, timeframe=TimeFrame.Day):
        """Pre-fetch and cache bar data for all symbols before training."""
        symbols = symbols or Data.SYMBOLS
        start   = start   or Data.START
        end     = end     or Data.END

        print(f"Pre-fetching bar data for {len(symbols)} symbols ({start} → {end})...")
        for i, symbol in enumerate(symbols, 1):
            self._get_bars(symbol, timeframe, start, end)
            print(f"  [{i}/{len(symbols)}] {symbol} ✓")
        print(f"Done. {len(Backtrader._data_cache)} entries cached.")

    def run_backtest(self, strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10000, plot=True):
        '''params:
            strategy: the strategy you wish to backtest, an instance of backtrader.Strategy
            symbols: the symbol (str) or list of symbols List[str] you wish to backtest on
            start: start date of backtest in format 'YYYY-MM-DD'
            end: end date of backtest in format: 'YYYY-MM-DD'
            timeframe: the timeframe the strategy trades on (size of bars)
            cash: the starting cash of backtest
        returns: (return_pct, sharpe_ratio, avg_annual_return_pct, max_drawdown_pct)
        '''

        cerebro = bt.Cerebro(stdstats=True)
        cerebro.broker.setcash(cash)

        cerebro.addstrategy(strategy)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        if isinstance(symbols, str):
            alpaca_data = self._get_bars(symbols, timeframe, start, end)
            cerebro.adddata(bt.feeds.PandasData(dataname=alpaca_data, name=symbols))
        elif isinstance(symbols, (list, set)):
            for symbol in symbols:
                alpaca_data = self._get_bars(symbol, timeframe, start, end)
                cerebro.adddata(bt.feeds.PandasData(dataname=alpaca_data, name=symbol))

        initial_portfolio_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_portfolio_value = cerebro.broker.getvalue()
        _return = (final_portfolio_value / initial_portfolio_value - 1) * 100

        strat = results[0]
        sharpe_ratio = strat.analyzers.mysharpe.get_analysis().get('sharperatio')
        annual_returns = strat.analyzers.annual_return.get_analysis()
        avg_annual_return = (sum(annual_returns.values()) / len(annual_returns) * 100) if annual_returns else 0.0
        max_drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']

        print(f"return={_return:.2f}% sharpe={sharpe_ratio:.2f} annual={avg_annual_return:.2f}% drawdown={max_drawdown:.2f}%")

        if plot:
            cerebro.plot(iplot=False)
            for i, fig_num in enumerate(plt.get_fignums(), start=1):
                plt.figure(fig_num)
                filename = f'backtest_plot_{i}.png'
                plt.savefig(filename, dpi=140, bbox_inches='tight')
                display(Image(filename))
            plt.close('all')

        return _return, sharpe_ratio, avg_annual_return, max_drawdown

    @execute_with_time_limit(10)
    def _timed_backtest(self, strategy, symbols, start, end, timeframe, cash):
        return self.run_backtest(strategy, symbols, start, end, timeframe, cash, plot=False)

    def execute_strategy(self, strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10000):
        """Execute strategy with 10 second time limit, then plot outside the timeout window."""
        _return, sharpe_ratio, avg_annual_return, max_drawdown = self._timed_backtest(
            strategy, symbols, start, end, timeframe, cash
        )
        if _return > 0:
            self.run_backtest(strategy, symbols, start, end, timeframe, cash, plot=True)
        return _return, sharpe_ratio, avg_annual_return, max_drawdown

    def run_strategy_from_file(self, strategy_file, symbols, start, end, timeframe=TimeFrame.Day, cash=10000):
        """Load a strategy from a .py file in the strategies folder and run backtest."""
        if strategy_file.endswith('.py'):
            strategy_file = strategy_file[:-3]

        import importlib
        strategy_module = importlib.import_module(strategy_file)
        StrategyClass = strategy_module.Strategy
        return self.run_backtest(StrategyClass, symbols, start, end, timeframe, cash)


class Data:
    SYMBOLS = [
        'AAPL', 'AMGN', 'AXP', 'BA',  'CAT',
        'CRM',  'CSCO', 'CVX', 'DIS', 'DOW',
        'GS',   'HD',   'HON', 'IBM', 'JNJ',
        'JPM',  'KO',   'MCD', 'MMM', 'MRK',
        'MSFT', 'NKE',  'NVDA', 'PG',  'TRV',
        'UNH',  'V',    'VZ',  'WBA', 'WMT',
    ]
    START = '2016-01-01'
    END   = '2024-12-31'

    def __init__(self):
        records = []
        for symbol in self.SYMBOLS:
            prompt = self._make_prompt(symbol)
            records.append({
                "prompt": [{"role": "user", "content": prompt}],
                "answer": 0,
            })
        self.dataset = Dataset.from_list(records)

    def _make_prompt(self, symbol: str) -> str:
        return f"""
        Create a trading strategy for {symbol} from {self.START} to {self.END} that is fully compatible with the following backtesting setup:

        - Framework: Backtrader
        - Strategy must subclass bt.Strategy
        - The strategy will be passed directly into:
        run_backtest(StrategyClass, symbols, start, end, timeframe, cash)

        STRICT RULES:
        1. Output ONLY a single Python class definition (no explanations, no markdown, no comments outside the class).
        2. The class MUST be named Strategy.
        3. Do NOT include imports (bt is already available).
        4. Do NOT reference external data, files, APIs, or indicators outside Backtrader.
        5. The strategy MUST work for Single-symbol strategies.
        6. All indicators must be created in __init__.
        7. Trading logic must be implemented in next().
        8. Orders must use only:
        - self.buy()
        - self.sell()
        - self.close()
        - self.order_target_percent()
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

    def get_dataset(self):
        return self.dataset


class RewardFunctions:

    RED = "\033[91m"
    RESET = "\033[0m"
    _step_count = 0

    def extract_function(text):
        """Extract Python function from markdown code blocks or raw output."""
        if text.count("```") >= 2:
            first = text.find("```") + 3
            second = text.find("```", first)
            fx = text[first:second].strip()
            fx = fx.removeprefix("python\n")
            fx = fx[fx.find("class Strategy"):]
            if fx.startswith("class Strategy(bt.Strategy):"):
                return fx
        idx = text.find("class Strategy(bt.Strategy):")
        if idx != -1:
            return text[idx:]
        return None

    def function_works(function):
        """Checks if the generated code is valid Python and can be executed."""
        if function is None:
            return False
        ok, info = check_python_modules(function)
        if ok is False or "error" in info:
            return False
        return True

    def extract_trading_parameters(prompt):
        """Extract symbol, start, and end date from the embedded prompt text."""
        match = re.search(r'trading strategy for (\w+) from (\S+) to (\S+)', prompt)
        if match:
            return match.group(1), match.group(2), match.group(3)
        raise ValueError(f"Could not extract trading parameters from prompt: {prompt[:100]}")

    def extract_strategy(func):
        namespace = {'bt': bt}
        exec(func, namespace)
        return namespace['Strategy']

    def has_required_functions(text):
        has_init = bool(re.search(r'def\s+__init__\s*\([^)]*\)\s*:', text))
        has_next = bool(re.search(r'def\s+next\s*\([^)]*\)\s*:', text))
        return has_init and has_next

    def save_strategy(function, _return, sharpe_ratio, avg_annual_return, max_drawdown, symbol, start, end):
        """Save a successful strategy with its stats and plots to a timestamped folder."""
        save_dir = "successful_strategies"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_dir = os.path.join(save_dir, f"{timestamp}_{symbol}")
        os.makedirs(strategy_dir, exist_ok=True)

        with open(os.path.join(strategy_dir, "strategy.py"), "w") as f:
            f.write(function)

        stats = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "return_pct": round(_return, 4),
            "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            "avg_annual_return_pct": round(avg_annual_return, 4),
            "max_drawdown_pct": round(max_drawdown, 4),
        }
        with open(os.path.join(strategy_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        for i in range(1, 10):
            src = f"backtest_plot_{i}.png"
            if os.path.exists(src):
                shutil.copy(src, os.path.join(strategy_dir, f"plot_{i}.png"))

        print(f"Saved to {strategy_dir}/")

    # (doesn't try, -10)->(invalid code, -3)->(exception, -2)->(no trading, -1)->(negative return, 0)->(positive return, max(avg_annual_return, 1))
    def strategy_succeeds(completions, prompts, **kwargs):
        """Reward valid moves even if strategy eventually fails."""
        scores = []
        backtrader = Backtrader()

        prompt_text = prompts[0][-1]["content"]
        symbol, start, end = RewardFunctions.extract_trading_parameters(prompt_text)
        RewardFunctions._step_count += 1
        print("=" * 50)
        print(f"Step {RewardFunctions._step_count} | {symbol}--{start}--{end}")

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
                _return, sharpe_ratio, avg_annual_return, max_drawdown = backtrader.execute_strategy(
                    strategy, symbols=symbol, start=start, end=end, timeframe=TimeFrame.Day, cash=10000
                )

                if _return == 0 and sharpe_ratio is None:
                    scores.append(-1)
                    continue

                if _return > 0:
                    print(function)
                    RewardFunctions.save_strategy(function, _return, sharpe_ratio, avg_annual_return, max_drawdown, symbol, start, end)
                    scores.append(max(avg_annual_return, 1))
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


def run_strategy(strategy_file, symbols=None, start=None, end=None, cash=10000):
    """Convenience function to load and run a strategy from the strategies folder.

    Args:
        strategy_file: filename (e.g., 'strategy_1' or 'strategy_1.py')
        symbols: symbol(s) to backtest. Defaults to Data.SYMBOLS[0] if None.
        start: start date. Defaults to Data.START if None.
        end: end date. Defaults to Data.END if None.
        cash: starting cash (default 10000)

    Returns:
        tuple: (return_pct, sharpe_ratio, avg_annual_return_pct, max_drawdown_pct)
    """
    symbols = symbols or Data.SYMBOLS[0]
    start   = start   or Data.START
    end     = end     or Data.END
    return Backtrader().run_strategy_from_file(strategy_file, symbols, start, end, cash=cash)


def main():
    unsloth = Unsloth(
        model_name="unsloth/Ministral-3-3B-Instruct-2512",
        lora_rank=16,
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=False,
    )
    data = Data()
    unsloth.train(
        steps=100,
        reward_functions=[RewardFunctions.strategy_succeeds],
        dataset=data.get_dataset(),
        hf_token=HF_TOKEN,
    )


if __name__ == "__main__":
    main()
