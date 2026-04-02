# unsloth
# import torch
# from unsloth import FastVisionModel
# from transformers import TextStreamer
# from trl import GRPOConfig, GRPOTrainer
# from unsloth import execute_with_time_limit
# from unsloth import check_python_modules

#backtrader
import backtrader as bt

# strategies folder import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
import matplotlib as mpl
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
from IPython.display import Image, display

# data
# from datasets import Dataset

#reward functions 
import re
import random
from datetime import datetime, timedelta

HF_TOKEN=''
ALPACA_API_KEY=''
ALPACA_SECRET_KEY=''


# class Unsloth:
    
#     def __init__(self, model_name: str, lora_rank: int = 32, max_seq_length: int = 4096, load_in_4bit = True, fast_inference = False, max_prompt_length = 1024):
#         self.model_name = model_name
#         self.lora_rank = lora_rank 
#         self.max_seq_length = max_seq_length
#         self.max_prompt_length = max_prompt_length
#         self.load_in_4bit = load_in_4bit
#         self.fast_inference = fast_inference
        
#         self.model, self.tokenizer = FastVisionModel.from_pretrained(
#             model_name = self.model_name,
#             max_seq_length = self.max_seq_length,
#             load_in_4bit = self.load_in_4bit, # True for QLora
#             fast_inference = self.fast_inference, # Enable vLLM fast inference, only for inference
#         )

#         self.model = FastVisionModel.get_peft_model(
#             self.model,
#             r = self.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#             target_modules = [
#                 "q_proj", "k_proj", "v_proj", "o_proj",
#                 "gate_proj", "up_proj", "down_proj",
#             ],
#             lora_alpha = self.lora_rank*2, # *2 speeds up training
#             use_gradient_checkpointing = "unsloth", # Reduces memory usage
#             random_state = 3407,
#         )
        
    
#     def generate(self, inputs: str):
#         text = self.tokenizer.apply_chat_template(
#             [{"role": "user", "content": inputs.strip()}],
#             tokenize = False,
#             add_generation_prompt = True,
#         )

#         _ = self.model.generate(
#             **self.tokenizer(images=None,text=text, return_tensors = "pt").to("cuda"),
#             temperature = 1.0,
#             max_new_tokens = 512,
#             streamer = TextStreamer(self.tokenizer, skip_prompt = False),
#         )
        
#         return _
    
#     def train(self, steps: int, reward_functions: list, dataset: torch.utils.data.Dataset):
#         max_prompt_length = self.max_prompt_length + 1 # + 1 just in case!
#         max_completion_length = self.max_seq_length - max_prompt_length

#         training_args = GRPOConfig(
#             temperature = 1.0,
#             learning_rate = 5e-6,
#             weight_decay = 0.001,
#             warmup_ratio = 0.1,
#             lr_scheduler_type = "linear",
#             optim = "adamw_8bit",
#             logging_steps = 1,
#             per_device_train_batch_size = 2,
#             gradient_accumulation_steps = 8, # Increase to 4 for smoother training
#             num_generations = 2, # Decrease if out of memory
#             max_prompt_length = max_prompt_length,
#             max_completion_length = max_completion_length,
#             # num_train_epochs = 1, # Set to 1 for a full training run
#             max_steps = steps,
#             save_steps = steps,
#             report_to = "none", # Can use Weights & Biases, TrackIO
#             output_dir = "outputs",
#         )

#         trainer = GRPOTrainer(
#             model = self.model,
#             processing_class = self.tokenizer,
#             reward_funcs = reward_functions,
#             args = training_args,
#             train_dataset = dataset,

#             # For optional training + evaluation
#             # train_dataset = new_dataset["train"],
#             # eval_dataset = new_dataset["test"],
#         )
        
#         trainer.train()


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
        self.api_secret_key = ALPACA_SECRET_KEY
        self.base_url = "https://paper-api.alpaca.markets"
        self.rest_api = REST(self.api_key, self.api_secret_key, self.base_url)

    def _get_bars(self, symbol, timeframe, start, end):
        key = (symbol, str(timeframe), start, end)
        if key not in Backtrader._data_cache:
            Backtrader._data_cache[key] = self.rest_api.get_bars(symbol, timeframe, start, end, adjustment='all').df
        return Backtrader._data_cache[key]

    def run_backtest(self, strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10000, plot=True):
        '''params:
            strategy: the strategy you wish to backtest, an instance of backtrader.Strategy
            symbols: the symbol (str) or list of symbols List[str] you wish to backtest on
            start: start date of backtest in format 'YYYY-MM-DD'
            end: end date of backtest in format: 'YYYY-MM-DD'
            timeframe: the timeframe the strategy trades on (size of bars)
            cash: the starting cash of backtest
            plot: whether to plot the results
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
        _return = (final_portfolio_value/initial_portfolio_value - 1)*100

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
                print(f'\nPlot saved to: {filename}')
            plt.close('all')

        return _return, sharpe_ratio, avg_annual_return, max_drawdown

    # @execute_with_time_limit(10)
    def execute_strategy(self, strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10000):
        """Execute strategy with 10 second time limit."""
        return self.run_backtest(strategy, symbols, start, end, timeframe, cash)

    def run_strategy_from_file(self, strategy_file, symbols, start, end, timeframe=TimeFrame.Day, cash=10000, plot=True):
        """Load a strategy from a .py file in the strategies folder and run backtest.

        Args:
            strategy_file: filename of strategy (e.g., 'strategy_1.py' or 'strategy_1')
            symbols: symbol(s) to backtest on
            start: start date 'YYYY-MM-DD'
            end: end date 'YYYY-MM-DD'
            timeframe: backtrader timeframe
            cash: starting cash
            plot: whether to plot the results
        """
        # Remove .py extension if present
        if strategy_file.endswith('.py'):
            strategy_file = strategy_file[:-3]

        # Import the strategy module dynamically
        import importlib
        strategy_module = importlib.import_module(strategy_file)

        # Get the Strategy class
        StrategyClass = strategy_module.Strategy

        # Run backtest
        return self.run_backtest(StrategyClass, symbols, start, end, timeframe, cash, plot)
 
      
class Data: 
    def __init__(self, prompt: str = None, n_samples: int = 10):
        self.prompt = """
        Create a trading strategy that is fully compatible with the following backtesting setup:

        - Framework: Backtrader
        - Strategy must subclass bt.Strategy
        - The strategy will be passed directly into:
        run_backtest(StrategyClass, symbols, start, end, timeframe, cash)

        STRICT RULES:
        1. Output ONLY a single Python class definition (no explanations, no markdown, no comments outside the class).
        2. The class MUST be named Strategy.
        3. Do NOT include imports (bt is already available).
        4. Do NOT reference external data, files, APIs, or indicators outside Backtrader.
        5. The strategy MUST work for:
        - Single-symbol strategies
        - Multi-symbol strategies (using self.datas)
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
        
        self.dataset = Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": self.prompt.strip()}],
                "answer": 0,
            }
        ] * n_samples)
        
    
    def get_prompt(self):
        return self.prompt
    
    def get_dataset(self):
        return self.dataset
    
    
class RewardFunctions:
    
    RED = "\033[91m"
    RESET = "\033[0m"

    def extract_function(text):
        """Extract Python function from markdown code blocks."""
        if text.count("```") >= 2:
            first = text.find("```") + 3
            second = text.find("```", first)
            fx = text[first:second].strip()
            fx = fx.removeprefix("python\n")
            fx = fx[fx.find("class Strategy"):]
            if fx.startswith("class Strategy(bt.Strategy):"):
                return fx
        return None

    def function_works(function):
        """Checks if the generated code is valid Python and can be executed."""

        if function is not None:
            ok, info = check_python_modules(function)

        if function is None:
            return False
        elif ok is False or "error" in info:
            return False  # Unsafe or error in code
        else:
            return True # Valid function

    def generate_trading_parameters(symbols_list=None, 
                                    min_days=365, 
                                    max_days=1095,
                                    earliest_start='2015-01-01',
                                    latest_end='2024-12-31'):
        """
        Generate random trading parameters: symbol, start date, and end date.
        """
        
        # Default symbols list if none provided
        if symbols_list is None:
            symbols_list = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 
                'JPM', 'V', 'WMT', 'DIS', 'NFLX', 'PYPL', 'INTC', 'AMD'
            ]
        
        # Select random symbol
        symbol = random.choice(symbols_list)
        
        # Parse date bounds
        earliest = datetime.strptime(earliest_start, '%Y-%m-%d')
        latest = datetime.strptime(latest_end, '%Y-%m-%d')
        
        # Generate random duration
        duration_days = 365
        
        # Generate random start date
        # Make sure there's enough room for the duration
        max_start_date = latest - timedelta(days=duration_days)
        
        if earliest > max_start_date:
            raise ValueError(f"Date range too small for minimum duration of {min_days} days")
        
        # Random start date between earliest and max_start_date
        days_range = (max_start_date - earliest).days
        random_days = random.randint(0, days_range)
        start_date = earliest + timedelta(days=random_days)
        
        # Calculate end date
        end_date = start_date + timedelta(days=duration_days)
        
        # Format as strings
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        return symbol, start_str, end_str

    def extract_strategy(func):
        # Create a namespace WITH the required imports
        namespace = {
        'bt': bt,
        }
        exec(func, namespace)
                
        # Extract the class from the namespace (assuming the class is named 'Strategy')
        Strategy = namespace['Strategy']

        return Strategy

    def has_required_functions(text):
        # Pattern to match __init__ method (with or without parameters)
        init_pattern = r'def\s+__init__\s*\([^)]*\)\s*:'
        
        # Pattern to match next method (with or without parameters)
        next_pattern = r'def\s+next\s*\([^)]*\)\s*:'
        
        # Check if both patterns are found in the text
        has_init = bool(re.search(init_pattern, text))
        has_next = bool(re.search(next_pattern, text))
        
        return has_init and has_next

    # (doesn't try, -10)->(invalid code, -3)->(exception, -2)->(no trading, -1)->(negative return, 0)->(positive return, return)
    def strategy_succeeds(completions, **kwargs):
        """Reward valid moves even if strategy eventually fails."""
        scores = []
        backtrader = Backtrader()
        symbol, start, end = RewardFunctions.generate_trading_parameters()
        print(f"{symbol}--{start}--{end}")
        
        
        for completion in completions:
            response = completion[0]["content"]
            function = RewardFunctions.extract_function(response)

            if not RewardFunctions.function_works(function):
                scores.append(-3)
                continue
                
            if not RewardFunctions.has_required_functions(function):
                scores.append(-10) # high penalty for not trying
                continue

            try:
                strategy = RewardFunctions.extract_strategy(function)
                _return, sharp_ratio = backtrader.execute_strategy(strategy, symbols=symbol, start=start, end=end, timeframe=TimeFrame.Day, cash=10000)

                if _return == 0 and sharp_ratio is None:
                    scores.append(-1) # strategy did not trade at all
                    continue
                
                scores.append(max(_return, 0))    

            except TimeoutError:
                print("Timeout")
                scores.append(-2.0)
            except Exception as e:
                print(f"{RED}Exception: {str(e)[:100]}{RESET}")
                scores.append(-2.0)

        return scores        

def run_strategy(strategy_file, symbols=None, start=None, end=None, cash=10000, plot=True):
    """Convenience function to load and run a strategy from the strategies folder.

    Args:
        strategy_file: filename (e.g., 'strategy_1' or 'strategy_1.py')
        symbols: symbol(s) to backtest. If None, uses default pool.
        start: start date. If None, uses random date.
        end: end date. If None, uses random date.
        cash: starting cash (default 10000)

    Returns:
        tuple: (return_pct, sharpe_ratio, avg_annual_return_pct, max_drawdown_pct)
    """
    backtrader = Backtrader()

    # Generate random parameters if not provided
    if symbols is None or start is None or end is None:
        symbols, start, end = RewardFunctions.generate_trading_parameters()
        print(f"Using random params: {symbols}--{start}--{end}")

    return backtrader.run_strategy_from_file(strategy_file, symbols, start, end, cash=cash, plot=plot)


DOW30 = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT',
    'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC',
    'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV',
    'UNH', 'V', 'VZ', 'WBA', 'WMT',
]


def main():
    print("Running optimized Dow 30 momentum strategy...")
    return_pct, sharpe_ratio, avg_annual_return, max_drawdown = run_strategy(
        'dow30_momentum.py',
        symbols=DOW30,
        start='2020-01-01',
        end='2024-12-31',
        plot=False
    )
    print(f"\n=== Results ===")
    print(f"Return: {return_pct:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Avg Annual Return: {avg_annual_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")


if __name__ == "__main__":
    main()
