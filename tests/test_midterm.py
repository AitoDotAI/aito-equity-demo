import pandas as pd

from pipeline.prices.midterm import _fwd, bucket_excess


def test_bucket_excess_bands():
    assert bucket_excess(-5.0, 5.0) == "lagged"
    assert bucket_excess(4.9, 5.0) == "inline"
    assert bucket_excess(8.0, 8.0) == "outperformed"


def test_fwd_starts_at_first_trading_day_on_or_after_t0():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    closes = pd.Series([100.0, 110.0, 120.0, 90.0], index=idx)
    ret, d0, d1 = _fwd(closes, pd.Timestamp("2020-01-01").date(), 2)
    assert d0 == idx[0] and d1 == idx[2]
    assert round(ret, 6) == 20.0
    assert _fwd(closes, pd.Timestamp("2020-01-01").date(), 4) is None  # not enough history
