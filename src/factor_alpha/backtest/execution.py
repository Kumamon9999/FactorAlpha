from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    """Realized execution metrics from moving between rebalance weights."""

    portfolio_returns: pd.Series
    trade_costs: pd.Series
    turnover_history: pd.Series
    fill_price_history: pd.DataFrame
    cumulative_return: float = field(init=False)

    def __post_init__(self):
        r = self.portfolio_returns.dropna()
        self.cumulative_return = float(np.exp(r.sum()) - 1) if len(r) else 0.0


class IntradayExecutionSimulator:
    """
    Simulate portfolio returns from daily target weights using intraday bars.

    This is intentionally a simple portfolio-level simulator: it uses 5-minute
    VWAP/close bars to estimate fill prices, turnover, and transaction costs
    while leaving signal generation in the existing daily BacktestEngine.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        cost_bps: float = 5.0,
        fill: str = "vwap",
    ):
        if fill not in {"vwap", "close", "open"}:
            raise ValueError("fill must be one of: 'vwap', 'close', 'open'")
        required = {"datetime", "date", "ticker", "open", "close", "volume", "amount"}
        missing = required.difference(bars.columns)
        if missing:
            raise ValueError(f"bars is missing required columns: {sorted(missing)}")

        self.bars = bars.copy()
        self.bars["date"] = pd.to_datetime(self.bars["date"]).dt.normalize()
        self.bars["datetime"] = pd.to_datetime(self.bars["datetime"])
        self.bars = self.bars.sort_values(["datetime", "ticker"])
        self.cost_bps = cost_bps
        self.fill = fill

        self._prices = self._daily_execution_prices()
        self._returns = np.log(self._prices / self._prices.shift(1))

    @property
    def execution_prices(self) -> pd.DataFrame:
        return self._prices

    def simulate(self, target_weights: pd.DataFrame) -> ExecutionResult:
        if target_weights.empty:
            return ExecutionResult(
                portfolio_returns=pd.Series(dtype=float),
                trade_costs=pd.Series(dtype=float),
                turnover_history=pd.Series(dtype=float),
                fill_price_history=pd.DataFrame(),
            )

        weights = target_weights.copy()
        weights.index = pd.to_datetime(weights.index).normalize()
        dates = self._returns.index
        weights = weights.sort_index()

        port_ret = pd.Series(0.0, index=dates, dtype=float)
        costs = pd.Series(0.0, index=dates, dtype=float)
        turnover: Dict[pd.Timestamp, float] = {}
        fills: Dict[pd.Timestamp, pd.Series] = {}

        current = pd.Series(0.0, index=weights.columns, dtype=float)
        schedule = [d for d in weights.index if d in dates]
        schedule_set = set(schedule)

        for date in dates:
            if date in schedule_set:
                target = weights.loc[date].fillna(0.0)
                common = current.index.union(target.index)
                current = current.reindex(common).fillna(0.0)
                target = target.reindex(common).fillna(0.0)
                one_way_turnover = float((target - current).abs().sum() / 2)
                turnover[date] = one_way_turnover
                costs.loc[date] = one_way_turnover * self.cost_bps / 10000.0
                fills[date] = self._prices.loc[date].reindex(common)
                current = target

            day_ret = self._returns.loc[date].reindex(current.index).fillna(0.0)
            port_ret.loc[date] = float(day_ret @ current) - costs.loc[date]

        return ExecutionResult(
            portfolio_returns=port_ret,
            trade_costs=costs[costs != 0],
            turnover_history=pd.Series(turnover, name="turnover"),
            fill_price_history=pd.DataFrame(fills).T,
        )

    def _daily_execution_prices(self) -> pd.DataFrame:
        grouped = self.bars.groupby(["date", "ticker"], sort=True)
        if self.fill == "open":
            px = grouped["open"].first()
        elif self.fill == "close":
            px = grouped["close"].last()
        else:
            tmp = self.bars.copy()
            amount = tmp["amount"]
            fallback_amount = tmp["close"] * tmp["volume"]
            tmp["notional"] = amount.where(amount.notna() & (amount > 0), fallback_amount)
            grouped = tmp.groupby(["date", "ticker"], sort=True)
            px = grouped["notional"].sum() / grouped["volume"].sum()
        return px.replace([np.inf, -np.inf], np.nan).unstack("ticker").sort_index().ffill()
