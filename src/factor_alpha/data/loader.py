import glob
import logging
import os
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class UniverseLoader:
    """Fetches and preprocesses OHLCV data for a stock universe."""

    def __init__(self, tickers: List[str]):
        if not tickers:
            raise ValueError("tickers must be a non-empty list")
        self.tickers = tickers
        self._prices: Optional[pd.DataFrame] = None
        self._volumes: Optional[pd.DataFrame] = None

    def fetch(self, start: str, end: str) -> "UniverseLoader":
        """Download adjusted close prices and volumes from Yahoo Finance."""
        logger.info("Fetching %d tickers from %s to %s", len(self.tickers), start, end)
        raw = yf.download(self.tickers, start=start, end=end, auto_adjust=True, progress=False)

        if isinstance(raw.columns, pd.MultiIndex):
            self._prices = raw["Close"].copy()
            self._volumes = raw["Volume"].copy()
        else:
            # single ticker returned flat DataFrame
            self._prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})
            self._volumes = raw[["Volume"]].rename(columns={"Volume": self.tickers[0]})

        self._prices = self._prices.ffill().bfill()
        self._volumes = self._volumes.ffill().bfill()

        # drop tickers with >20% missing before fill
        threshold = 0.8
        valid = self._prices.notna().mean() >= threshold
        dropped = valid[~valid].index.tolist()
        if dropped:
            logger.warning("Dropping tickers with insufficient data: %s", dropped)
        self._prices = self._prices.loc[:, valid]
        self._volumes = self._volumes.loc[:, valid]

        logger.info("Universe: %d tickers, %d trading days", self._prices.shape[1], len(self._prices))
        return self

    @property
    def prices(self) -> pd.DataFrame:
        self._check_fetched()
        return self._prices

    @property
    def volumes(self) -> pd.DataFrame:
        self._check_fetched()
        return self._volumes

    def get_returns(self, method: str = "log") -> pd.DataFrame:
        """Compute daily returns. method: 'log' or 'simple'."""
        self._check_fetched()
        if method == "log":
            return np.log(self._prices / self._prices.shift(1)).iloc[1:]
        elif method == "simple":
            return self._prices.pct_change().iloc[1:]
        raise ValueError(f"method must be 'log' or 'simple', got '{method}'")

    def get_dollar_volume(self) -> pd.DataFrame:
        """Price × volume — proxy for liquidity."""
        self._check_fetched()
        return (self._prices * self._volumes).ffill()

    def align(self, *others: pd.DataFrame) -> List[pd.DataFrame]:
        """Align multiple DataFrames to the intersection of dates and tickers."""
        frames = [self._prices] + list(others)
        common_idx = frames[0].index
        common_cols = frames[0].columns
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
            common_cols = common_cols.intersection(f.columns)
        return [f.loc[common_idx, common_cols] for f in frames]

    def _check_fetched(self):
        if self._prices is None:
            raise RuntimeError("Call .fetch(start, end) before accessing data.")


class CSVLoader:
    """
    Loads price and (optionally) volume data from local CSV files instead of
    downloading from Yahoo Finance.

    Supported CSV layouts
    ---------------------
    Wide format (one column per ticker) — default:
        date,AAPL,MSFT,GOOGL,...
        2018-01-02,172.3,85.1,1053.2,...

    Long format (date + ticker + value columns):
        date,ticker,close,volume
        2018-01-02,AAPL,172.3,22000000
        2018-01-02,MSFT,85.1,18000000

    Parameters
    ----------
    prices_path : str
        Path to the prices CSV file.
    volumes_path : str, optional
        Path to the volumes CSV file (wide or long, same layout as prices).
    date_col : str
        Name of the date column (default "date").
    ticker_col : str
        Name of the ticker column in long format (default "ticker").
    price_col : str
        Name of the price column in long format (default "close").
    volume_col : str
        Name of the volume column in long format (default "volume").
    fmt : str
        "wide" (default) or "long".
    start : str, optional
        Clip data to this start date (inclusive), e.g. "2018-01-01".
    end : str, optional
        Clip data to this end date (inclusive), e.g. "2023-12-31".
    missing_threshold : float
        Drop tickers with more than this fraction of missing values (default 0.2).
    """

    def __init__(
        self,
        prices_path: str,
        volumes_path: Optional[str] = None,
        date_col: str = "date",
        ticker_col: str = "ticker",
        price_col: str = "close",
        volume_col: str = "volume",
        fmt: str = "wide",
        start: Optional[str] = None,
        end: Optional[str] = None,
        missing_threshold: float = 0.2,
    ):
        self.prices_path = prices_path
        self.volumes_path = volumes_path
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.fmt = fmt
        self.start = start
        self.end = end
        self.missing_threshold = missing_threshold

        self._prices: Optional[pd.DataFrame] = None
        self._volumes: Optional[pd.DataFrame] = None

        self._load()

    # ------------------------------------------------------------------
    # Public interface (mirrors UniverseLoader)
    # ------------------------------------------------------------------

    @property
    def prices(self) -> pd.DataFrame:
        return self._prices

    @property
    def volumes(self) -> Optional[pd.DataFrame]:
        return self._volumes

    def get_returns(self, method: str = "log") -> pd.DataFrame:
        """Compute daily returns. method: 'log' or 'simple'."""
        if method == "log":
            return np.log(self._prices / self._prices.shift(1)).iloc[1:]
        elif method == "simple":
            return self._prices.pct_change().iloc[1:]
        raise ValueError(f"method must be 'log' or 'simple', got '{method}'")

    def get_dollar_volume(self) -> pd.DataFrame:
        """Price × volume — proxy for liquidity. Requires volumes_path."""
        if self._volumes is None:
            raise RuntimeError("No volumes file provided to CSVLoader.")
        return (self._prices * self._volumes).ffill()

    def align(self, *others: pd.DataFrame) -> List[pd.DataFrame]:
        """Align multiple DataFrames to the intersection of dates and tickers."""
        frames = [self._prices] + list(others)
        common_idx = frames[0].index
        common_cols = frames[0].columns
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
            common_cols = common_cols.intersection(f.columns)
        return [f.loc[common_idx, common_cols] for f in frames]

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self):
        self._prices = self._read_csv(self.prices_path, value_col=self.price_col)
        if self.volumes_path:
            self._volumes = self._read_csv(self.volumes_path, value_col=self.volume_col)
            # align volumes to price index/columns
            self._volumes = self._volumes.reindex(
                index=self._prices.index, columns=self._prices.columns
            )
        logger.info(
            "CSVLoader: %d tickers, %d trading days loaded from %s",
            self._prices.shape[1], len(self._prices), self.prices_path,
        )

    def _read_csv(self, path: str, value_col: str) -> pd.DataFrame:
        raw = pd.read_csv(path)

        if self.fmt == "wide":
            df = self._parse_wide(raw)
        elif self.fmt == "long":
            df = self._parse_long(raw, value_col)
        else:
            raise ValueError(f"fmt must be 'wide' or 'long', got '{self.fmt}'")

        # date range clip
        if self.start:
            df = df.loc[df.index >= self.start]
        if self.end:
            df = df.loc[df.index <= self.end]

        df = df.sort_index().ffill().bfill()

        # drop tickers with too many missing values
        valid = df.notna().mean() >= (1 - self.missing_threshold)
        dropped = valid[~valid].index.tolist()
        if dropped:
            logger.warning("Dropping tickers with insufficient data: %s", dropped)
        return df.loc[:, valid]

    def _parse_wide(self, raw: pd.DataFrame) -> pd.DataFrame:
        # first column is the date, rest are tickers
        date_col = self.date_col if self.date_col in raw.columns else raw.columns[0]
        df = raw.set_index(date_col)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.apply(pd.to_numeric, errors="coerce")

    def _parse_long(self, raw: pd.DataFrame, value_col: str) -> pd.DataFrame:
        raw[self.date_col] = pd.to_datetime(raw[self.date_col])
        raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
        df = raw.pivot(index=self.date_col, columns=self.ticker_col, values=value_col)
        df.index.name = "date"
        df.columns.name = None
        return df


# ---------------------------------------------------------------------------
# Column name mapping for A-share 5-minute data (Chinese → English)
# ---------------------------------------------------------------------------
_ASHARE_COLS = {
    "时间":  "datetime",
    "代码":  "ticker",
    "名称":  "name",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨幅":  "pct_chg",
    "振幅":  "amplitude",
}


class AShareZipLoader:
    """
    Loads A-share 5-minute bar data from per-day ZIP archives and resamples
    to daily OHLCV.

    Dataset layout expected on disk
    --------------------------------
    dataset_dir/
      YYYY-MM/
        YYYYMMDD_5min.zip      ← one ZIP per trading day
          sz000001.csv         ← one CSV per stock inside the ZIP
          sh600519.csv
          ...

    Each CSV has Chinese-header columns:
        时间, 代码, 名称, 开盘价, 收盘价, 最高价, 最低价, 成交量, 成交额, 涨幅, 振幅

    Parameters
    ----------
    dataset_dir : str
        Root directory that contains the YYYY-MM sub-folders.
    tickers : list of str, optional
        Stocks to load, e.g. ["sh600519", "sz000001"]. Loads all stocks if
        None (slow for 5000+ stocks — recommend specifying a list).
    start : str, optional
        First date to include, e.g. "2025-01-01".
    end : str, optional
        Last date to include, e.g. "2025-09-30".
    missing_threshold : float
        Drop tickers missing more than this fraction of trading days (default 0.2).
    """

    def __init__(
        self,
        dataset_dir: str,
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        missing_threshold: float = 0.2,
    ):
        self.dataset_dir = dataset_dir
        self.tickers = tickers
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.missing_threshold = missing_threshold

        self._prices: Optional[pd.DataFrame] = None
        self._volumes: Optional[pd.DataFrame] = None
        self._amounts: Optional[pd.DataFrame] = None

        self._load()

    # ------------------------------------------------------------------
    # Public interface (mirrors UniverseLoader)
    # ------------------------------------------------------------------

    @property
    def prices(self) -> pd.DataFrame:
        return self._prices

    @property
    def volumes(self) -> pd.DataFrame:
        return self._volumes

    @property
    def amounts(self) -> pd.DataFrame:
        """Total turnover (成交额) per day — more reliable liquidity proxy than volume for A-shares."""
        return self._amounts

    def get_returns(self, method: str = "log") -> pd.DataFrame:
        if method == "log":
            return np.log(self._prices / self._prices.shift(1)).iloc[1:]
        elif method == "simple":
            return self._prices.pct_change().iloc[1:]
        raise ValueError(f"method must be 'log' or 'simple', got '{method}'")

    def get_dollar_volume(self) -> pd.DataFrame:
        """Daily turnover amount (成交额) — use this for liquidity screening on A-shares."""
        return self._amounts.ffill()

    def align(self, *others: pd.DataFrame) -> List[pd.DataFrame]:
        frames = [self._prices] + list(others)
        common_idx = frames[0].index
        common_cols = frames[0].columns
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
            common_cols = common_cols.intersection(f.columns)
        return [f.loc[common_idx, common_cols] for f in frames]

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self):
        zip_paths = self._discover_zips()
        if not zip_paths:
            raise FileNotFoundError(f"No ZIP files found under {self.dataset_dir}")
        logger.info("AShareZipLoader: found %d ZIP files", len(zip_paths))

        daily_records: List[Dict] = []

        for zip_path in zip_paths:
            date_str = os.path.basename(zip_path).split("_")[0]  # "20250901"
            trade_date = pd.Timestamp(date_str)

            if self.start and trade_date < self.start:
                continue
            if self.end and trade_date > self.end:
                continue

            records = self._read_zip(zip_path, trade_date)
            daily_records.extend(records)

        if not daily_records:
            raise ValueError("No data loaded — check date range and ticker list.")

        panel = pd.DataFrame(daily_records).set_index(["date", "ticker"])

        self._prices  = panel["close"].unstack("ticker")
        self._volumes = panel["volume"].unstack("ticker")
        self._amounts = panel["amount"].unstack("ticker")

        for attr in ("_prices", "_volumes", "_amounts"):
            df = getattr(self, attr).sort_index().ffill().bfill()
            valid = df.notna().mean() >= (1 - self.missing_threshold)
            dropped = valid[~valid].index.tolist()
            if dropped:
                logger.warning("Dropping tickers with insufficient data: %s", dropped)
            setattr(self, attr, df.loc[:, valid])

        logger.info(
            "AShareZipLoader: %d tickers × %d trading days loaded",
            self._prices.shape[1], len(self._prices),
        )

    def _discover_zips(self) -> List[str]:
        pattern = os.path.join(self.dataset_dir, "**", "*_5min.zip")
        return sorted(glob.glob(pattern, recursive=True))

    def _read_zip(self, zip_path: str, trade_date: pd.Timestamp) -> List[Dict]:
        records = []
        ticker_set = set(self.tickers) if self.tickers else None

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                if ticker_set:
                    names = [n for n in names if os.path.splitext(n)[0] in ticker_set]

                for fname in names:
                    ticker = os.path.splitext(fname)[0]  # "sz000001"
                    try:
                        with zf.open(fname) as f:
                            df = pd.read_csv(f)
                    except Exception:
                        continue

                    df = df.rename(columns=_ASHARE_COLS)
                    if "close" not in df.columns:
                        continue

                    for col in ("open", "close", "high", "low", "volume", "amount"):
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    # resample to single daily bar
                    rec = {
                        "date":   trade_date,
                        "ticker": ticker,
                        "close":  df["close"].iloc[-1] if len(df) else np.nan,
                        "open":   df["open"].iloc[0]   if "open"   in df.columns and len(df) else np.nan,
                        "high":   df["high"].max()      if "high"   in df.columns else np.nan,
                        "low":    df["low"].min()       if "low"    in df.columns else np.nan,
                        "volume": df["volume"].sum()    if "volume" in df.columns else np.nan,
                        "amount": df["amount"].sum()    if "amount" in df.columns else np.nan,
                    }
                    records.append(rec)
        except zipfile.BadZipFile:
            logger.warning("Skipping corrupt ZIP: %s", zip_path)

        return records
