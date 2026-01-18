from xml.parsers.expat import model
import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from trl import GRPOConfig, GRPOTrainer


class Unsloth:
    
    def __init__(self, max_seq_length: int = 4096, lora_rank: int = 32, max_prompt_length = 1024):
        self.lora_rank = lora_rank 
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

    def setup_model(self, model_name: str, load_in_4bit = False, fast_inference = False):
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name = model_name,
            max_seq_length = self.max_seq_length,
            load_in_4bit = load_in_4bit, # False for LoRA 16bit
            fast_inference = fast_inference, # Enable vLLM fast inference
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
        
        return self.model, self.tokenizer
    
    def generate(self, inputs: str, generation_config: dict):
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
    
    def train(self, reward_functions: list, dataset: torch.utils.data.Dataset):
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
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1, # Increase to 4 for smoother training
            num_generations = 2, # Decrease if out of memory
            max_prompt_length = max_prompt_length,
            max_completion_length = max_completion_length,
            # num_train_epochs = 1, # Set to 1 for a full training run
            max_steps = 100,
            save_steps = 100,
            report_to = "none", # Can use Weights & Biases, TrackIO
            output_dir = "outputs",
        )

        trainer = GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = [
                function_works,
                strategy_succeeds,
            ],
            args = training_args,
            train_dataset = dataset,

            # For optional training + evaluation
            # train_dataset = new_dataset["train"],
            # eval_dataset = new_dataset["test"],
        )
        
        trainer.train()


import backtrader as bt
import matplotlib as mpl

from alpaca_trade_api.rest import REST, TimeFrame
mpl.rcParams['figure.dpi'] = 140 # chart resolution

from typing import Callable
from unsloth import execute_with_time_limit

import matplotlib.pyplot as plt
from IPython.display import Image, display

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

        # add analytics
        # cerebro.addobserver(bt.observers.Value)
        # cerebro.addobserver(bt.observers.BuySell)
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
        print(f'Starting Portfolio Value: {initial_portfolio_value}')
        results = cerebro.run()
        final_portfolio_value = cerebro.broker.getvalue()
        _return = (final_portfolio_value/initial_portfolio_value - 1)*100
        print(f'Final Portfolio Value: {final_portfolio_value} ---> Return: {_return}%')

        strat = results[0]
        sharpe_ratio = strat.analyzers.mysharpe.get_analysis()
        print('Sharpe Ratio:', sharpe_ratio['sharperatio'])
        # plot (non-interactive backend) -> save figures and display in notebook

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
 
 
from datasets import Dataset
       
class Data: 
    def __init__(self, prompt: str = None):
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
    
    def get_prompt(self):
        return self.prompt
    
    def get_dataset(self, n_samples: int = 1000):
        return Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": self.prompt.strip()}],
                "answer": 0,
            }
        ] * n_samples)
    

def main():
    unsloth = Unsloth()
    model, tokenizer = unsloth.setup_model(
        model_name = "unsloth/Ministral-3-3B-Reasoning-2512",
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
    )
    data = Data()
    dataset = data.get_dataset(n_samples = 1000)
    unsloth.train(
        reward_functions = [
            function_works,
            strategy_succeeds,
        ],
        dataset = dataset,
    )
