import numpy as np
import pandas as pd

from factor_alpha.backtest import IntradayExecutionSimulator
from factor_alpha.data import IntradayFeatureBuilder


def _bars():
    rows = []
    for date in pd.to_datetime(["2025-01-02", "2025-01-03"]):
        for ticker, base in {"AAA": 10.0, "BBB": 20.0}.items():
            for i in range(4):
                close = base + i + (1 if date.day == 3 else 0)
                volume = 100 + i
                rows.append(
                    {
                        "datetime": date + pd.Timedelta(minutes=5 * i + 570),
                        "date": date,
                        "ticker": ticker,
                        "open": close - 0.2,
                        "high": close + 0.5,
                        "low": close - 0.5,
                        "close": close,
                        "volume": volume,
                        "amount": close * volume,
                    }
                )
    return pd.DataFrame(rows)


def test_intraday_feature_builder_daily_ohlcv():
    builder = IntradayFeatureBuilder(_bars())
    daily = builder.daily_ohlcv()

    assert set(daily) == {"open", "high", "low", "close", "volume", "amount"}
    assert daily["close"].shape == (2, 2)
    assert daily["close"].loc[pd.Timestamp("2025-01-02"), "AAA"] == 13.0
    assert daily["volume"].loc[pd.Timestamp("2025-01-02"), "AAA"] == 406


def test_intraday_feature_builder_factor_matrices():
    builder = IntradayFeatureBuilder(_bars())

    realized_vol = builder.intraday_realized_vol()
    vwap_gap = builder.close_to_vwap_gap()
    morning = builder.morning_return(periods=2)
    last = builder.last_period_momentum(periods=2)

    assert realized_vol.index.equals(vwap_gap.index)
    assert realized_vol.columns.tolist() == ["AAA", "BBB"]
    assert np.isfinite(vwap_gap.to_numpy()).all()
    assert morning.loc[pd.Timestamp("2025-01-02"), "AAA"] > 0
    assert last.loc[pd.Timestamp("2025-01-02"), "AAA"] > 0


def test_intraday_execution_simulator_uses_weight_schedule_and_costs():
    bars = _bars()
    weights = pd.DataFrame(
        {"AAA": [0.6, 0.2], "BBB": [0.4, 0.8]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )

    result = IntradayExecutionSimulator(bars, cost_bps=10, fill="vwap").simulate(weights)

    assert len(result.portfolio_returns) == 2
    assert result.turnover_history.loc[pd.Timestamp("2025-01-02")] == 0.5
    assert round(result.turnover_history.loc[pd.Timestamp("2025-01-03")], 6) == 0.4
    assert result.trade_costs.loc[pd.Timestamp("2025-01-02")] == 0.0005
    assert not result.fill_price_history.empty
