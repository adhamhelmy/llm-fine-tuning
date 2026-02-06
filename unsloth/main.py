# unsloth
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer

#backtrader
import backtrader as bt
import matplotlib as mpl
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
from unsloth import execute_with_time_limit
from IPython.display import Image, display

# data
from datasets import Dataset

#reward functions 
import re
import random
from datetime import datetime, timedelta
from unsloth import check_python_modules



class Unsloth:
    
    def __init__(self, model_name: str, lora_rank: int = 32, max_seq_length: int = 4096, load_in_4bit = True, fast_inference = False, max_prompt_length = 1024):
        self.model_name = model_name
        self.lora_rank = lora_rank 
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        self.load_in_4bit = load_in_4bit
        self.fast_inference = fast_inference
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            load_in_4bit = self.load_in_4bit, # True for QLora
            fast_inference = self.fast_inference, # Enable vLLM fast inference, only for inference
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
            r = self.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = self.lora_rank*2, # *2 speeds up training
            use_gradient_checkpointing = "unsloth", # Reduces memory usage
            random_state = 3407,
        )
        
    
    def generate(self, inputs: str):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": inputs.strip()}],
            tokenize = False,
            add_generation_prompt = True,
        )

        _ = self.model.generate(
            **self.tokenizer(images=None,text=text, return_tensors = "pt").to("cuda"),
            temperature = 1.0,
            max_new_tokens = 512,
            streamer = TextStreamer(self.tokenizer, skip_prompt = False),
        )
        
        return _
    
    def train(self, steps: int, reward_functions: list, dataset: torch.utils.data.Dataset):
        max_prompt_length = self.max_prompt_length + 1 # + 1 just in case!
        max_completion_length = self.max_seq_length - max_prompt_length

        training_args = GRPOConfig(
            temperature = 1.0,
            learning_rate = 5e-6,
            weight_decay = 0.001,
            warmup_ratio = 0.1,
            lr_scheduler_type = "linear",
            optim = "adamw_8bit",
            logging_steps = 1,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8, # Increase to 4 for smoother training
            num_generations = 2, # Decrease if out of memory
            max_prompt_length = max_prompt_length,
            max_completion_length = max_completion_length,
            # num_train_epochs = 1, # Set to 1 for a full training run
            max_steps = steps,
            save_steps = steps,
            report_to = "none", # Can use Weights & Biases, TrackIO
            output_dir = "outputs",
        )

        trainer = GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = reward_functions,
            args = training_args,
            train_dataset = dataset,

            # For optional training + evaluation
            # train_dataset = new_dataset["train"],
            # eval_dataset = new_dataset["test"],
        )
        
        trainer.train()


class Backtrader:
    def __init__(self):
        # Load API Key ID and Secret Key from Colab secrets
        self.api_key = ""
        self.api_secret_key = ""
        self.base_url = "https://paper-api.alpaca.markets"
        self.rest_api = REST(self.api_key, self.api_secret_key, self.base_url)
        
    def run_backtest(self, strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10000):
        '''params:
            strategy: the strategy you wish to backtest, an instance of backtrader.Strategy
            symbols: the symbol (str) or list of symbols List[str] you wish to backtest on
            start: start date of backtest in format 'YYYY-MM-DD'
            end: end date of backtest in format: 'YYYY-MM-DD'
            timeframe: the timeframe the strategy trades on (size of bars) -
                    1 min: TimeFrame.Minute, 1 day: TimeFrame.Day, 5 min: TimeFrame(5, TimeFrameUnit.Minute)
            cash: the starting cash of backtest
        '''

        # initialize backtrader broker
        cerebro = bt.Cerebro(stdstats=True)
        cerebro.broker.setcash(cash)

        # add strategy
        cerebro.addstrategy(strategy)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')

        # historical data request
        if type(symbols) == str:
            symbol = symbols
            alpaca_data = self.rest_api.get_bars(symbol, timeframe, start, end,  adjustment='all').df
            data = bt.feeds.PandasData(dataname=alpaca_data, name=symbol)
            cerebro.adddata(data)
        elif type(symbols) == list or type(symbols) == set:
            for symbol in symbols:
                alpaca_data = self.rest_api.get_bars(symbol, timeframe, start, end, adjustment='all').df
                data = bt.feeds.PandasData(dataname=alpaca_data, name=symbol)
                cerebro.adddata(data)

        # run
        initial_portfolio_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_portfolio_value = cerebro.broker.getvalue()
        _return = (final_portfolio_value/initial_portfolio_value - 1)*100

        strat = results[0]
        sharpe_ratio = strat.analyzers.mysharpe.get_analysis()

        if _return != 0 and sharpe_ratio['sharperatio'] is not None:
            cerebro.plot(iplot=False)  # creates matplotlib figures when using Agg
            for i, fig_num in enumerate(plt.get_fignums(), start=1):
                plt.figure(fig_num)
                filename = f'backtest_plot_{i}.png'
                plt.savefig(filename, dpi=140, bbox_inches='tight')
                display(Image(filename))
            plt.close('all')

        return _return, sharpe_ratio['sharperatio']
    
    @execute_with_time_limit(10)
    def execute_strategy(self, strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10000):
        """Execute strategy with 10 second time limit."""
        return self.run_backtest(strategy, symbols, start, end, timeframe, cash)
 
      
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

def main():
    model = Unsloth(
        model_name = "unsloth/Ministral-3-3B-Reasoning-2512",
    )
    
    data = Data()
    
    
    model.train(
        reward_functions = [
            RewardFunctions.strategy_succeeds,
        ],
        dataset = data.get_dataset(),
    )
