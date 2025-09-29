import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
from datetime import datetime
import sys
from io import StringIO

# Import your existing backtest code
import logging
import uuid
from typing import List, Dict, Tuple, Optional
import re
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, retry_if_exception_type
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for trading parameters."""
    symbol: str = "AAPL"
    start: str = "2018-01-01"
    end: str = datetime.today().strftime("%Y-%m-%d")
    fast_window: int = 10
    slow_window: int = 50
    initial_cash: float = 100_000.0
    position_size_pct: float = 0.1
    commission_per_trade: float = 1.0
    slippage_pct: float = 0.0005
    trading_costs_enabled: bool = True
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    risk_free_rate: float = 0.04

def validate_inputs(symbol: str, start: str, end: str) -> None:
    if not re.match(r'^[A-Z0-9.]{1,10}$', symbol):
        raise ValueError("Invalid ticker symbol")
    try:
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format, use YYYY-MM-DD")
    if datetime.strptime(start, "%Y-%m-%d") >= datetime.strptime(end, "%Y-%m-%d"):
        raise ValueError("Start date must be before end date")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0, 0.1),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def fetch_data(symbol: str, start: str, end: str, trace_id: str) -> pd.DataFrame:
    logger.info(f"[{trace_id}] Fetching data for {symbol}")
    df = yf.download(symbol, start=start, end=end, progress=False, timeout=10, auto_adjust=False)
    if df.empty:
        logger.error(f"[{trace_id}] No data for {symbol}")
        raise ValueError("No data — check ticker or date range")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()

def compute_indicators(df: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    df['sma_fast'] = df['Adj Close'].rolling(fast_window).mean()
    df['sma_slow'] = df['Adj Close'].rolling(slow_window).mean()
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['bb_middle'] = df['Adj Close'].rolling(20).mean()
    bb_std = df['Adj Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df['signal_shift'] = df['signal'].shift(1).fillna(0)
    df['trade'] = df['signal'] - df['signal_shift']
    return df

def execute_trade(row, cash, position, entry_price, position_size_pct, slippage_pct, 
                 commission, stop_loss_pct, take_profit_pct, trace_id):
    price = row['Adj Close']
    trade_log_entry = None
    if position > 0 and entry_price > 0:
        if stop_loss_pct and price <= entry_price * (1 - stop_loss_pct):
            trade_proceeds = position * price * (1 - slippage_pct)
            cash += (trade_proceeds - commission)
            trade_log_entry = {'date': row.name, 'side': 'SELL', 'type': 'STOP_LOSS',
                             'price': price, 'shares': position, 'cash': cash,
                             'pnl': (price - entry_price) * position}
            position = 0
            entry_price = 0
            return cash, position, entry_price, trade_log_entry
        if take_profit_pct and price >= entry_price * (1 + take_profit_pct):
            trade_proceeds = position * price * (1 - slippage_pct)
            cash += (trade_proceeds - commission)
            trade_log_entry = {'date': row.name, 'side': 'SELL', 'type': 'TAKE_PROFIT',
                             'price': price, 'shares': position, 'cash': cash,
                             'pnl': (price - entry_price) * position}
            position = 0
            entry_price = 0
            return cash, position, entry_price, trade_log_entry
    if row['trade'] == 1 and position == 0:
        alloc = cash * position_size_pct
        if alloc > 0:
            shares_to_buy = np.floor(alloc / (price * (1 + slippage_pct)))
            if shares_to_buy > 0:
                trade_cost = shares_to_buy * price * (1 + slippage_pct)
                cash -= (trade_cost + commission)
                position += shares_to_buy
                entry_price = price
                trade_log_entry = {'date': row.name, 'side': 'BUY', 'type': 'SIGNAL',
                                 'price': price, 'shares': shares_to_buy, 'cash': cash, 'pnl': 0}
    elif row['trade'] == -1 and position > 0:
        trade_proceeds = position * price * (1 - slippage_pct)
        cash += (trade_proceeds - commission)
        trade_log_entry = {'date': row.name, 'side': 'SELL', 'type': 'SIGNAL',
                         'price': price, 'shares': position, 'cash': cash,
                         'pnl': (price - entry_price) * position}
        position = 0
        entry_price = 0
    return cash, position, entry_price, trade_log_entry

def backtest(df, initial_cash, position_size_pct, slippage_pct, commission, 
            stop_loss_pct, take_profit_pct, trace_id):
    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    equity_curve = []
    cash_history = []
    pos_history = []
    trade_log = []
    for idx, row in df.iterrows():
        cash, position, entry_price, trade_log_entry = execute_trade(
            row, cash, position, entry_price, position_size_pct, slippage_pct,
            commission, stop_loss_pct, take_profit_pct, trace_id)
        if trade_log_entry:
            trade_log.append(trade_log_entry)
        market_value = position * row['Adj Close']
        total_equity = cash + market_value
        equity_curve.append(total_equity)
        cash_history.append(cash)
        pos_history.append(position)
    df = df.assign(equity=equity_curve, cash=cash_history, position=pos_history)
    return df, trade_log

def compute_metrics(df, trade_log, initial_cash, slippage_pct, commission, risk_free_rate):
    ending_value = df['equity'].iloc[-1]
    total_return = (ending_value / initial_cash) - 1
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    trading_days_per_year = 252
    cagr = (ending_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0
    df['returns'] = df['equity'].pct_change().fillna(0)
    cum_max = df['equity'].cummax()
    drawdown = (df['equity'] - cum_max) / cum_max
    max_drawdown = drawdown.min()
    excess_returns = df['returns'].mean() - (risk_free_rate / trading_days_per_year)
    sharpe = (excess_returns / df['returns'].std()) * np.sqrt(trading_days_per_year) if df['returns'].std() != 0 else 0.0
    downside_returns = df['returns'][df['returns'] < 0]
    sortino = (excess_returns / downside_returns.std()) * np.sqrt(trading_days_per_year) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0.0
    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0.0
    buy_trades = [t for t in trade_log if t['side'] == 'BUY']
    sell_trades = [t for t in trade_log if t['side'] == 'SELL']
    wins = [t['pnl'] for t in sell_trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in sell_trades if t['pnl'] < 0]
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)) if sell_trades else 0
    buy_hold_return = (df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) - 1
    return {
        'ending_value': ending_value, 'total_return': total_return, 'cagr': cagr,
        'max_drawdown': max_drawdown, 'sharpe': sharpe, 'sortino': sortino,
        'calmar': calmar, 'num_trades': len(buy_trades), 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor,
        'expectancy': expectancy, 'buy_hold_return': buy_hold_return,
        'total_commission': commission * len(trade_log),
        'avg_days_in_trade': days / len(buy_trades) if buy_trades else 0
    }

class BacktestUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Strategy Backtest")
        self.root.geometry("1400x900")
        
        self.config = Config()
        self.results_queue = queue.Queue()
        self.running = False
        
        self.create_widgets()
        self.root.after(100, self.process_queue)
        
    def create_widgets(self):
        # Main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Configuration
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=0)
        
        # Right panel - Results
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        self.create_config_panel(left_frame)
        self.create_results_panel(right_frame)
        
    def create_config_panel(self, parent):
        # Title
        title_label = ttk.Label(parent, text="Backtest Configuration", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Scrollable frame for config
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", 
                             lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Symbol & Dates
        symbol_frame = ttk.LabelFrame(scrollable_frame, text="Symbol & Period", padding=10)
        symbol_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(symbol_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.symbol_var = tk.StringVar(value=self.config.symbol)
        ttk.Entry(symbol_frame, textvariable=self.symbol_var, width=15).grid(row=0, column=1, pady=2)
        
        ttk.Label(symbol_frame, text="Start Date:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.start_var = tk.StringVar(value=self.config.start)
        ttk.Entry(symbol_frame, textvariable=self.start_var, width=15).grid(row=1, column=1, pady=2)
        
        ttk.Label(symbol_frame, text="End Date:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.end_var = tk.StringVar(value=self.config.end)
        ttk.Entry(symbol_frame, textvariable=self.end_var, width=15).grid(row=2, column=1, pady=2)
        
        # Strategy Parameters
        strategy_frame = ttk.LabelFrame(scrollable_frame, text="Strategy Parameters", padding=10)
        strategy_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(strategy_frame, text="Fast SMA:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.fast_var = tk.IntVar(value=self.config.fast_window)
        ttk.Spinbox(strategy_frame, from_=5, to=100, textvariable=self.fast_var, 
                   width=13).grid(row=0, column=1, pady=2)
        
        ttk.Label(strategy_frame, text="Slow SMA:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.slow_var = tk.IntVar(value=self.config.slow_window)
        ttk.Spinbox(strategy_frame, from_=10, to=200, textvariable=self.slow_var, 
                   width=13).grid(row=1, column=1, pady=2)
        
        # Capital & Position
        capital_frame = ttk.LabelFrame(scrollable_frame, text="Capital Management", padding=10)
        capital_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(capital_frame, text="Initial Cash ($):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cash_var = tk.DoubleVar(value=self.config.initial_cash)
        ttk.Entry(capital_frame, textvariable=self.cash_var, width=15).grid(row=0, column=1, pady=2)
        
        ttk.Label(capital_frame, text="Position Size (%):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.position_var = tk.DoubleVar(value=self.config.position_size_pct * 100)
        ttk.Spinbox(capital_frame, from_=1, to=100, textvariable=self.position_var, 
                   width=13, increment=5).grid(row=1, column=1, pady=2)
        
        # Trading Costs
        costs_frame = ttk.LabelFrame(scrollable_frame, text="Trading Costs", padding=10)
        costs_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(costs_frame, text="Commission ($):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.commission_var = tk.DoubleVar(value=self.config.commission_per_trade)
        ttk.Entry(costs_frame, textvariable=self.commission_var, width=15).grid(row=0, column=1, pady=2)
        
        ttk.Label(costs_frame, text="Slippage (%):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.slippage_var = tk.DoubleVar(value=self.config.slippage_pct * 100)
        ttk.Entry(costs_frame, textvariable=self.slippage_var, width=15).grid(row=1, column=1, pady=2)
        
        # Risk Management
        risk_frame = ttk.LabelFrame(scrollable_frame, text="Risk Management", padding=10)
        risk_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stop_loss_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(risk_frame, text="Stop Loss (%)", 
                       variable=self.stop_loss_enabled).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.stop_loss_var = tk.DoubleVar(value=2.0)
        ttk.Entry(risk_frame, textvariable=self.stop_loss_var, width=15).grid(row=0, column=1, pady=2)
        
        self.take_profit_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(risk_frame, text="Take Profit (%)", 
                       variable=self.take_profit_enabled).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.take_profit_var = tk.DoubleVar(value=5.0)
        ttk.Entry(risk_frame, textvariable=self.take_profit_var, width=15).grid(row=1, column=1, pady=2)
        
        ttk.Label(risk_frame, text="Risk-Free Rate (%):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.risk_free_var = tk.DoubleVar(value=self.config.risk_free_rate * 100)
        ttk.Entry(risk_frame, textvariable=self.risk_free_var, width=15).grid(row=2, column=1, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=15)
        
        self.run_button = ttk.Button(button_frame, text="Run Backtest", 
                                     command=self.run_backtest, style="Accent.TButton")
        self.run_button.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_defaults).pack(fill=tk.X, pady=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_results_panel(self, parent):
        # Notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary Tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, 
                                                      font=("Courier", 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Charts Tab
        charts_frame = ttk.Frame(notebook)
        notebook.add(charts_frame, text="Charts")
        
        self.charts_canvas_widget = tk.Canvas(charts_frame)
        self.charts_canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Log Tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Log")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                  font=("Courier", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def reset_defaults(self):
        config = Config()
        self.symbol_var.set(config.symbol)
        self.start_var.set(config.start)
        self.end_var.set(config.end)
        self.fast_var.set(config.fast_window)
        self.slow_var.set(config.slow_window)
        self.cash_var.set(config.initial_cash)
        self.position_var.set(config.position_size_pct * 100)
        self.commission_var.set(config.commission_per_trade)
        self.slippage_var.set(config.slippage_pct * 100)
        self.stop_loss_enabled.set(False)
        self.take_profit_enabled.set(False)
        self.risk_free_var.set(config.risk_free_rate * 100)
        
    def run_backtest(self):
        if self.running:
            messagebox.showwarning("Running", "A backtest is already running!")
            return
        
        self.running = True
        self.run_button.config(state=tk.DISABLED, text="Running...")
        self.summary_text.delete(1.0, tk.END)
        self.log_text.delete(1.0, tk.END)
        
        # Start backtest in thread
        thread = threading.Thread(target=self.execute_backtest, daemon=True)
        thread.start()
        
    def execute_backtest(self):
        try:
            # Build config from UI
            config = Config()
            config.symbol = self.symbol_var.get().upper()
            config.start = self.start_var.get()
            config.end = self.end_var.get()
            config.fast_window = self.fast_var.get()
            config.slow_window = self.slow_var.get()
            config.initial_cash = self.cash_var.get()
            config.position_size_pct = self.position_var.get() / 100
            config.commission_per_trade = self.commission_var.get()
            config.slippage_pct = self.slippage_var.get() / 100
            config.stop_loss_pct = self.stop_loss_var.get() / 100 if self.stop_loss_enabled.get() else None
            config.take_profit_pct = self.take_profit_var.get() / 100 if self.take_profit_enabled.get() else None
            config.risk_free_rate = self.risk_free_var.get() / 100
            
            trace_id = str(uuid.uuid4())
            
            self.results_queue.put(("log", f"Starting backtest for {config.symbol}...\n"))
            
            # Validate and fetch
            validate_inputs(config.symbol, config.start, config.end)
            df = fetch_data(config.symbol, config.start, config.end, trace_id)
            self.results_queue.put(("log", f"Fetched {len(df)} trading days\n"))
            
            # Compute indicators
            df = compute_indicators(df, config.fast_window, config.slow_window)
            self.results_queue.put(("log", "Computed technical indicators\n"))
            
            # Run backtest
            df, trade_log = backtest(df, config.initial_cash, config.position_size_pct,
                                    config.slippage_pct, config.commission_per_trade,
                                    config.stop_loss_pct, config.take_profit_pct, trace_id)
            self.results_queue.put(("log", f"Executed {len(trade_log)} trades\n"))
            
            # Compute metrics
            metrics = compute_metrics(df, trade_log, config.initial_cash,
                                     config.slippage_pct, config.commission_per_trade,
                                     config.risk_free_rate)
            
            # Send results
            self.results_queue.put(("results", (config, df, trade_log, metrics)))
            self.results_queue.put(("log", "Backtest completed successfully!\n"))
            
        except Exception as e:
            self.results_queue.put(("error", str(e)))
        finally:
            self.running = False
            self.results_queue.put(("done", None))
            
    def process_queue(self):
        try:
            while True:
                msg_type, data = self.results_queue.get_nowait()
                
                if msg_type == "log":
                    self.log_text.insert(tk.END, data)
                    self.log_text.see(tk.END)
                    
                elif msg_type == "results":
                    config, df, trade_log, metrics = data
                    self.display_results(config, df, trade_log, metrics)
                    
                elif msg_type == "error":
                    messagebox.showerror("Error", f"Backtest failed:\n{data}")
                    self.log_text.insert(tk.END, f"\nERROR: {data}\n")
                    
                elif msg_type == "done":
                    self.run_button.config(state=tk.NORMAL, text="Run Backtest")
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)
            
    def display_results(self, config, df, trade_log, metrics):
        # Display summary
        summary = f"""
{'='*70}
{'BACKTEST PERFORMANCE REPORT':^70}
{'='*70}

{'STRATEGY CONFIGURATION':^70}
{'-'*70}
Symbol:                    {config.symbol}
Period:                    {df.index[0].date()} to {df.index[-1].date()}
Trading Days:              {len(df)}
Fast SMA:                  {config.fast_window}
Slow SMA:                  {config.slow_window}
Initial Capital:           ${config.initial_cash:,.2f}
Position Size:             {config.position_size_pct*100:.1f}%
Commission per Trade:      ${config.commission_per_trade:.2f}
Slippage:                  {config.slippage_pct*100:.2f}%
"""
        if config.stop_loss_pct:
            summary += f"Stop Loss:                 {config.stop_loss_pct*100:.1f}%\n"
        if config.take_profit_pct:
            summary += f"Take Profit:               {config.take_profit_pct*100:.1f}%\n"
            
        summary += f"""
{'PERFORMANCE METRICS':^70}
{'-'*70}
Ending Equity:             ${metrics['ending_value']:,.2f}
Total Return:              {metrics['total_return']*100:+.2f}%
CAGR:                      {metrics['cagr']*100:+.2f}%
Buy & Hold Return:         {metrics['buy_hold_return']*100:+.2f}%
Max Drawdown:              {metrics['max_drawdown']*100:.2f}%
Sharpe Ratio:              {metrics['sharpe']:.3f}
Sortino Ratio:             {metrics['sortino']:.3f}
Calmar Ratio:              {metrics['calmar']:.3f}

{'TRADE STATISTICS':^70}
{'-'*70}
Total Trades:              {metrics['num_trades']}
Win Rate:                  {metrics['win_rate']*100:.1f}%
Average Win:               ${metrics['avg_win']:,.2f}
Average Loss:              ${metrics['avg_loss']:,.2f}
Profit Factor:             {metrics['profit_factor']:.2f}
Expectancy:                ${metrics['expectancy']:.2f}
Total Commissions:         ${metrics['total_commission']:.2f}
Avg Days in Trade:         {metrics['avg_days_in_trade']:.1f}

{'='*70}
"""
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
        
        # Create charts
        self.create_charts(df, trade_log, config, metrics)
        
    def create_charts(self, df, trade_log, config, metrics):
        # Clear previous charts
        for widget in self.charts_canvas_widget.winfo_children():
            widget.destroy()
            
        fig = plt.Figure(figsize=(14, 10), dpi=100)
        
        # Price and SMAs
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(df.index, df['Adj Close'], label='Adj Close', alpha=0.6, linewidth=1.5)
        ax1.plot(df.index, df['sma_fast'], label=f'SMA{config.fast_window}', linewidth=1)
        ax1.plot(df.index, df['sma_slow'], label=f'SMA{config.slow_window}', linewidth=1)
        
        buys = [t for t in trade_log if t['side'] == 'BUY']
        sells = [t for t in trade_log if t['side'] == 'SELL']
        
        if buys:
            ax1.scatter([t['date'] for t in buys], [t['price'] for t in buys],
                       marker='^', color='green', s=50, label='Buy', zorder=5, alpha=0.7)
        if sells:
            ax1.scatter([t['date'] for t in sells], [t['price'] for t in sells],
                       marker='v', color='red', s=50, label='Sell', zorder=5, alpha=0.7)
        
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_title(f"{config.symbol} Price and Moving Averages", fontweight='bold', fontsize=10)
        ax1.set_ylabel('Price ($)', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # Equity Curve
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(df.index, df['equity'], label='Strategy Equity', linewidth=2, color='blue')
        
        buy_hold_equity = (df['Adj Close'] / df['Adj Close'].iloc[0]) * df['equity'].iloc[0]
        ax2.plot(df.index, buy_hold_equity, label='Buy & Hold', linewidth=2,
                color='orange', alpha=0.7, linestyle='--')
        
        ax2.fill_between(df.index, df['equity'], alpha=0.2)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.set_title('Equity Curve Comparison', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Equity ($)', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        
        # Drawdown
        ax3 = fig.add_subplot(3, 2, 3)
        cum_max = df['equity'].cummax()
        drawdown = (df['equity'] - cum_max) / cum_max * 100
        ax3.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
        ax3.plot(df.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Drawdown (%)', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)
        
        # RSI
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(df.index, df['rsi'], color='purple', linewidth=1)
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5, linewidth=0.8)
        ax4.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax4.set_title('RSI (14)', fontweight='bold', fontsize=10)
        ax4.set_ylabel('RSI', fontsize=9)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)
        
        # Volume
        ax5 = fig.add_subplot(3, 2, 5)
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
                 for i in range(len(df))]
        ax5.bar(df.index, df['Volume'], color=colors, alpha=0.5, width=1)
        ax5.plot(df.index, df['volume_ma'], color='blue', linewidth=1.5, label='Volume MA')
        ax5.set_title('Volume', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Volume', fontsize=9)
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(labelsize=8)
        
        # Monthly Returns
        ax6 = fig.add_subplot(3, 2, 6)
        monthly_returns = df['equity'].resample('M').last().pct_change() * 100
        if len(monthly_returns) > 1:
            monthly_returns = monthly_returns.dropna()
            colors_map = ['green' if x > 0 else 'red' for x in monthly_returns]
            ax6.bar(range(len(monthly_returns)), monthly_returns, color=colors_map, alpha=0.6)
            ax6.axhline(y=0, color='black', linewidth=0.5)
            labels = [d.strftime('%Y-%m') for d in monthly_returns.index]
            step = max(1, len(monthly_returns) // 10)
            ax6.set_xticks(range(0, len(monthly_returns), step))
            ax6.set_xticklabels(labels[::step], rotation=45, ha='right', fontsize=7)
            ax6.grid(True, alpha=0.3, axis='y')
            ax6.set_title('Monthly Returns', fontweight='bold', fontsize=10)
            ax6.set_ylabel('Return (%)', fontsize=9)
            ax6.tick_params(labelsize=8)
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.charts_canvas_widget)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = BacktestUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()