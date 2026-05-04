import glob
import logging
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loader import _ASHARE_COLS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BarSpec:
    """Metadata describing a bar dataset."""

    frequency: str
    periods_per_day: int
    annualization: int


ASHARE_5MIN_SPEC = BarSpec(frequency="5min", periods_per_day=48, annualization=252 * 48)


class AShareIntradayZipLoader:
    """
    Loads A-share 5-minute bars from the project ZIP layout without collapsing
    them to daily rows.

    The returned DataFrame is in long format with columns:
    datetime, date, ticker, open, high, low, close, volume, amount.
    """

    def __init__(
        self,
        dataset_dir: str,
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        self.dataset_dir = dataset_dir
        self.tickers = tickers
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.spec = ASHARE_5MIN_SPEC

    def load(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for zip_path in self._discover_zips():
            trade_date = pd.Timestamp(os.path.basename(zip_path).split("_")[0])
            if self.start and trade_date < self.start:
                continue
            if self.end and trade_date > self.end:
                continue
            frames.extend(self._read_zip(zip_path, trade_date))

        if not frames:
            raise ValueError("No intraday bars loaded. Check date range and ticker list.")

        bars = pd.concat(frames, ignore_index=True)
        bars = bars.sort_values(["datetime", "ticker"]).reset_index(drop=True)
        return bars

    def load_panel(self, field: str = "close") -> pd.DataFrame:
        """Return a datetime x ticker matrix for one intraday field."""
        bars = self.load()
        if field not in bars.columns:
            raise ValueError(f"Unknown field '{field}'. Available columns: {list(bars.columns)}")
        return bars.pivot(index="datetime", columns="ticker", values=field).sort_index()

    def _discover_zips(self) -> List[str]:
        pattern = os.path.join(self.dataset_dir, "**", "*_5min.zip")
        return sorted(glob.glob(pattern, recursive=True))

    def _read_zip(self, zip_path: str, trade_date: pd.Timestamp) -> List[pd.DataFrame]:
        ticker_set = set(self.tickers) if self.tickers else None
        frames: List[pd.DataFrame] = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                if ticker_set:
                    names = [n for n in names if os.path.splitext(os.path.basename(n))[0] in ticker_set]

                for fname in names:
                    ticker = os.path.splitext(os.path.basename(fname))[0]
                    try:
                        with zf.open(fname) as f:
                            raw = pd.read_csv(f)
                    except Exception:
                        logger.debug("Skipping unreadable member %s in %s", fname, zip_path)
                        continue

                    df = self._normalise_member(raw, ticker, trade_date)
                    if not df.empty:
                        frames.append(df)
        except zipfile.BadZipFile:
            logger.warning("Skipping corrupt ZIP: %s", zip_path)

        return frames

    def _normalise_member(
        self, raw: pd.DataFrame, fallback_ticker: str, trade_date: pd.Timestamp
    ) -> pd.DataFrame:
        df = raw.rename(columns=_ASHARE_COLS).copy()
        if "datetime" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()

        keep = ["datetime", "ticker", "open", "high", "low", "close", "volume", "amount"]
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
        df = df[keep]

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = trade_date.normalize()
        df["ticker"] = fallback_ticker

        for col in ("open", "high", "low", "close", "volume", "amount"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["datetime", "close"])
        return df[["datetime", "date", "ticker", "open", "high", "low", "close", "volume", "amount"]]


class IntradayFeatureBuilder:
    """Build daily factor-ready matrices from long intraday bars."""

    def __init__(self, bars: pd.DataFrame):
        required = {"datetime", "date", "ticker", "open", "high", "low", "close", "volume", "amount"}
        missing = required.difference(bars.columns)
        if missing:
            raise ValueError(f"bars is missing required columns: {sorted(missing)}")
        self.bars = bars.copy()
        self.bars["datetime"] = pd.to_datetime(self.bars["datetime"])
        self.bars["date"] = pd.to_datetime(self.bars["date"]).dt.normalize()
        self.bars = self.bars.sort_values(["ticker", "datetime"])

    def daily_ohlcv(self) -> Dict[str, pd.DataFrame]:
        grouped = self.bars.groupby(["date", "ticker"], sort=True)
        daily = grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
        return {field: daily[field].unstack("ticker").sort_index() for field in daily.columns}

    def close_to_vwap_gap(self) -> pd.DataFrame:
        daily = self.daily_ohlcv()
        vwap = self.vwap()
        return (daily["close"] / vwap - 1).replace([np.inf, -np.inf], np.nan)

    def high_low_range(self) -> pd.DataFrame:
        daily = self.daily_ohlcv()
        return ((daily["high"] - daily["low"]) / daily["open"]).replace([np.inf, -np.inf], np.nan)

    def intraday_realized_vol(self) -> pd.DataFrame:
        returns = self.bars.groupby("ticker", group_keys=False)["close"].pct_change()
        tmp = self.bars[["date", "ticker"]].copy()
        tmp["sq_ret"] = returns.pow(2)
        return tmp.groupby(["date", "ticker"])["sq_ret"].sum().pow(0.5).unstack("ticker").sort_index()

    def last_period_momentum(self, periods: int = 12) -> pd.DataFrame:
        if periods <= 0:
            raise ValueError("periods must be positive")

        def _last_momentum(day: pd.DataFrame) -> float:
            if len(day) <= periods:
                return np.nan
            start = day["close"].iloc[-periods - 1]
            end = day["close"].iloc[-1]
            return np.log(end / start) if start > 0 and end > 0 else np.nan

        return (
            self.bars.groupby(["date", "ticker"], sort=True)
            .apply(_last_momentum)
            .unstack("ticker")
            .sort_index()
        )

    def morning_return(self, periods: int = 12) -> pd.DataFrame:
        if periods <= 0:
            raise ValueError("periods must be positive")

        def _morning(day: pd.DataFrame) -> float:
            if len(day) <= periods:
                return np.nan
            start = day["open"].iloc[0]
            end = day["close"].iloc[periods - 1]
            return np.log(end / start) if start > 0 and end > 0 else np.nan

        return (
            self.bars.groupby(["date", "ticker"], sort=True)
            .apply(_morning)
            .unstack("ticker")
            .sort_index()
        )

    def vwap(self) -> pd.DataFrame:
        tmp = self.bars.copy()
        amount = tmp["amount"]
        fallback_amount = tmp["close"] * tmp["volume"]
        tmp["notional"] = amount.where(amount.notna() & (amount > 0), fallback_amount)
        grouped = tmp.groupby(["date", "ticker"], sort=True)
        notional = grouped["notional"].sum()
        volume = grouped["volume"].sum()
        vwap = (notional / volume).replace([np.inf, -np.inf], np.nan)
        return vwap.unstack("ticker").sort_index()
