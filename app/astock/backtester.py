"""
A-share backtester — enforces Chinese market rules:

  1. T+1: shares bought today cannot be sold until the next trading day
  2. Price limits: cannot buy at limit-up, cannot sell at limit-down
  3. Lot size: must trade in multiples of 100 shares (1 lot)
  4. Trading fees:
     - Commission: 0.025% (min 5 yuan per trade)
     - Stamp duty: 0.05% (sell only, since 2023-08-28)
     - Transfer fee: 0.001% (both sides)
  5. Slippage: configurable (default 0.1%)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class TradeRecord:
    date: str
    action: str        # "BUY" or "SELL"
    price: float
    shares: int
    amount: float
    commission: float
    stamp_duty: float
    transfer_fee: float
    total_cost: float
    reason: str = ""


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.00025     # 万2.5
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005      # 千0.5 (sell only)
    transfer_fee_rate: float = 0.00001   # 万0.1
    slippage: float = 0.001              # 0.1%
    max_position_pct: float = 1.0        # max % of capital per stock
    lot_size: int = 100                  # must trade multiples of 100


def run_backtest(df: pd.DataFrame, signals: pd.Series,
                 config: BacktestConfig = None) -> dict:
    """
    Run backtest with A-share rules.

    Args:
        df: DataFrame with columns [date, open, close, high, low, volume,
            is_limit_up, is_limit_down, ...]
        signals: Series of +1 (buy), -1 (sell), 0 (hold)
        config: BacktestConfig

    Returns:
        dict with equity curve, trades, metrics
    """
    if config is None:
        config = BacktestConfig()

    n = len(df)
    dates = df["date"].values
    opens = df["open"].values
    closes = df["close"].values
    limit_up = df["is_limit_up"].values if "is_limit_up" in df else np.zeros(n, dtype=bool)
    limit_down = df["is_limit_down"].values if "is_limit_down" in df else np.zeros(n, dtype=bool)

    cash = config.initial_capital
    shares_held = 0
    buy_date_idx = -10  # index of last buy (for T+1 check)
    trades: List[TradeRecord] = []
    equity_curve = np.zeros(n)
    position_curve = np.zeros(n)
    cash_curve = np.zeros(n)

    for i in range(n):
        signal = signals.iloc[i] if i < len(signals) else 0

        # Execute at next day's open (signal generated at close, execute next open)
        if i > 0:
            exec_price = opens[i]

            # BUY signal (from previous day)
            prev_signal = signals.iloc[i - 1] if (i - 1) < len(signals) else 0
            if prev_signal > 0 and shares_held == 0:
                if limit_up[i]:
                    pass  # cannot buy at limit-up (封涨停)
                else:
                    buy_price = exec_price * (1 + config.slippage)
                    max_amount = cash * config.max_position_pct
                    max_shares = int(max_amount / buy_price / config.lot_size) * config.lot_size
                    if max_shares >= config.lot_size:
                        amount = max_shares * buy_price
                        comm = max(amount * config.commission_rate, config.min_commission)
                        transfer = amount * config.transfer_fee_rate
                        total = amount + comm + transfer
                        if total <= cash:
                            cash -= total
                            shares_held = max_shares
                            buy_date_idx = i
                            trades.append(TradeRecord(
                                date=str(dates[i])[:10], action="BUY",
                                price=round(buy_price, 2), shares=max_shares,
                                amount=round(amount, 2), commission=round(comm, 2),
                                stamp_duty=0, transfer_fee=round(transfer, 2),
                                total_cost=round(total, 2), reason="买入信号"))

            # SELL signal (from previous day) — T+1 check
            elif prev_signal < 0 and shares_held > 0:
                if i - buy_date_idx < 2:
                    pass  # T+1: cannot sell shares bought yesterday
                elif limit_down[i]:
                    pass  # cannot sell at limit-down (封跌停)
                else:
                    sell_price = exec_price * (1 - config.slippage)
                    amount = shares_held * sell_price
                    comm = max(amount * config.commission_rate, config.min_commission)
                    stamp = amount * config.stamp_duty_rate
                    transfer = amount * config.transfer_fee_rate
                    total_cost = comm + stamp + transfer
                    cash += amount - total_cost
                    trades.append(TradeRecord(
                        date=str(dates[i])[:10], action="SELL",
                        price=round(sell_price, 2), shares=shares_held,
                        amount=round(amount, 2), commission=round(comm, 2),
                        stamp_duty=round(stamp, 2), transfer_fee=round(transfer, 2),
                        total_cost=round(total_cost, 2), reason="卖出信号"))
                    shares_held = 0

        # Record equity
        market_value = shares_held * closes[i]
        equity_curve[i] = cash + market_value
        position_curve[i] = market_value
        cash_curve[i] = cash

    # Metrics
    total_return = (equity_curve[-1] / config.initial_capital - 1) * 100
    buy_hold_return = (closes[-1] / closes[0] - 1) * 100

    buy_trades = [t for t in trades if t.action == "BUY"]
    sell_trades = [t for t in trades if t.action == "SELL"]
    n_round_trips = min(len(buy_trades), len(sell_trades))
    wins = 0
    for b, s in zip(buy_trades, sell_trades):
        if s.price > b.price:
            wins += 1
    win_rate = wins / max(n_round_trips, 1) * 100

    # Daily returns for Sharpe
    daily_ret = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    sharpe = np.sqrt(252) * daily_ret.mean() / (daily_ret.std() + 1e-10) if len(daily_ret) > 1 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-10)
    max_dd = drawdown.max() * 100

    total_fees = sum(t.commission + t.transfer_fee for t in trades if t.action == "BUY") + \
                 sum(t.commission + t.stamp_duty + t.transfer_fee for t in trades if t.action == "SELL")

    return {
        "初始资金": config.initial_capital,
        "最终资金": round(equity_curve[-1], 2),
        "总收益率(%)": round(total_return, 2),
        "买入持有(%)": round(buy_hold_return, 2),
        "超额收益(%)": round(total_return - buy_hold_return, 2),
        "交易次数": len(trades),
        "完整交易": n_round_trips,
        "胜率(%)": round(win_rate, 1),
        "年化Sharpe": round(sharpe, 3),
        "最大回撤(%)": round(max_dd, 2),
        "总交易费用": round(total_fees, 2),
        "_equity": equity_curve,
        "_position": position_curve,
        "_cash": cash_curve,
        "_dates": dates,
        "_trades": trades,
        "_drawdown": drawdown * 100,
    }
