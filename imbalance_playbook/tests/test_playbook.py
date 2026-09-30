"""Offline tests: engine maths, no look-ahead, full backtest and live paths on fake data.

    python -m pytest -q tests
"""
from __future__ import annotations

import datetime as dt
import math
import sys
import pathlib

import databento_dbn as d
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from imbalance import common as C  # noqa: E402
from imbalance import engine as E  # noqa: E402
import fake_databento as F  # noqa: E402

CFG = C.load_config()
T0 = pd.Timestamp("2026-09-29 19:53:00", tz="UTC")       # 15:53 ET
CLOSE = pd.Timestamp("2026-09-29 20:00:00", tz="UTC")


def hist_rows(rows):
    df = pd.DataFrame(rows, columns=E.IMB_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def uni(**over):
    base = dict(prev_close=5.0, adv_usd=10_000_000.0, sigma=0.04, venue="XNAS")
    base.update(over)
    return pd.DataFrame([base], index=pd.Index(["AAAA"], name="symbol"))


# ------------------------------------------------------------------ equations by hand
def test_features_match_hand_calculation():
    h = hist_rows([(T0, "AAAA", "XNAS", "B", 5.00, 5.06, np.nan, 200_000, 300_000)])
    snap = E.snapshot(h, T0, 60, 30)
    f = E.features(snap, uni(), CFG, "B").iloc[0]
    assert f["notional"] == pytest.approx(1_500_000)
    assert f["imb_adv"] == pytest.approx(0.15)
    assert f["imb_paired"] == pytest.approx(1.5)
    assert f["near_gap"] == pytest.approx(0.012)
    assert f["impact"] == pytest.approx(0.75 * 0.04 * math.sqrt(0.15))
    assert f["exp_move"] == pytest.approx(0.5 * 0.012)                  # Window B uses beta * near gap
    # $5 stock: spread 20 bps -> half 10 + slippage 5 = 15 bps, commission 2*0.0035/5 = 14 bps
    assert f["cost"] == pytest.approx(0.0015 + 0.0014)
    assert f["edge"] == pytest.approx(0.006 - 0.0029)
    fa = E.features(snap, uni(), CFG, "A").iloc[0]
    assert fa["exp_move"] == pytest.approx(fa["impact"])                # Window A uses impact model


def test_sizing_and_limit_by_hand():
    row = pd.Series(dict(ref=5.0, sigma=0.04, imb=300_000, spread_bps=20.0))
    secs = 420
    adverse = 2.0 * 0.04 * math.sqrt(420 / 23400)
    assert E.entry_limit(row, CFG) == 5.01                             # 5 * (1 + 15 bps) = 5.0075 -> 5.01
    expected = math.floor(min(25000 * 0.0025 / (5.01 * adverse), 0.10 * 300_000, 5000 / 5.01))
    assert E.size_shares(row, CFG, secs, 5.01) == expected == 998     # notional cap binds at the price paid
    assert 998 * 5.01 <= 5000


# ------------------------------------------------------------------ no look-ahead / stability
def test_snapshot_ignores_future_and_detects_flip_and_shrink():
    h = hist_rows([
        (T0 - pd.Timedelta(seconds=40), "AAAA", "XNAS", "A", 5, np.nan, np.nan, 1e5, 9e5),
        (T0 - pd.Timedelta(seconds=10), "AAAA", "XNAS", "B", 5, np.nan, np.nan, 1e5, 3e5),
        (T0 - pd.Timedelta(seconds=30), "ZZZZ", "XNAS", "B", 5, np.nan, np.nan, 1e5, 4e5),
        (T0 - pd.Timedelta(seconds=5), "ZZZZ", "XNAS", "B", 5, np.nan, np.nan, 1e5, 1e5),
        (T0 + pd.Timedelta(seconds=1), "AAAA", "XNAS", "B", 99, np.nan, np.nan, 1e5, 9e9),   # future
    ])
    s = E.snapshot(h, T0, 60, 30)
    assert s.loc["AAAA", "ref"] == 5 and s.loc["AAAA", "imb"] == 3e5     # future row not used
    assert not s.loc["AAAA", "side_stable"]
    assert s.loc["ZZZZ", "shrink"] == pytest.approx(0.75)


def test_universe_stats_use_only_prior_sessions_and_skip_splits():
    days = [x.date() for x in pd.bdate_range("2026-08-03", periods=25)]
    closes = [5.0] * 20 + [50.0, 50.0, 55.0, 55.0, 60.0]       # 1:10 reverse split on day 20
    bars = pd.DataFrame({"date": days, "symbol": "AAAA", "close": closes, "volume": 1e6})
    st = E.universe_stats(bars, CFG)
    assert st["prev_close"].loc[days[21], "AAAA"] == 50.0      # previous session, not same day
    assert st["adv_usd"].loc[days[20], "AAAA"] == pytest.approx(5e6)   # days 0-19 only
    assert st["sigma"].loc[days[20], "AAAA"] == pytest.approx(0.0)    # the split day is not a return
    assert st["sigma"].loc[days[22], "AAAA"] == pytest.approx(0.0)    # split ratio 10 filtered out


def test_filters_give_reasons():
    rows = []
    specs = {  # symbol: (side, ref, imb, paired)
        "OKAY": ("B", 5, 300_000, 200_000), "SELL": ("A", 5, 300_000, 200_000),
        "TINY": ("B", 5, 10_000, 200_000), "WEAK": ("B", 5, 60_000, 1_000_000), "HIGH": ("B", 12, 300_000, 200_000),
    }
    for s, (side, ref, imb, paired) in specs.items():
        rows.append((T0, s, "XNAS", side, ref, np.nan, np.nan, paired, imb))
    u = pd.DataFrame({s: dict(prev_close=5.0, adv_usd=10e6, sigma=0.04, venue="XNAS") for s in specs}).T
    u[["prev_close", "adv_usd", "sigma"]] = u[["prev_close", "adv_usd", "sigma"]].astype(float)
    out = E.apply_filters(E.features(E.snapshot(hist_rows(rows), T0, 60, 30), u, CFG, "A"), CFG, "A")
    assert out.loc["OKAY", "reason"] == ""
    assert out.loc["SELL", "reason"] == "sell side"
    assert out.loc["TINY", "reason"] == "small $"
    assert out.loc["WEAK", "reason"] == "weak vs paired"
    assert out.loc["HIGH", "reason"] == "price band"


def test_bad_data_never_trades():
    h = hist_rows([(T0, "AAAA", "XNAS", "B", 5.00, np.nan, np.nan, 200_000, np.nan)])     # undefined imbalance
    out = E.run_window(h, uni(), CFG, "A", T0, CLOSE)
    assert out.empty or (out["action"] != "TRADE").all()
    h2 = hist_rows([(T0, "AAAA", "XNAS", "B", 5.00, np.nan, np.nan, 200_000, 300_000)])
    out2 = E.run_window(h2, uni(sigma=np.nan), CFG, "A", T0, CLOSE)
    assert (out2["action"] != "TRADE").all() and out2.iloc[0]["reason"] == "bad data"


def test_session_ledger_caps_the_whole_evening():
    h = hist_rows([(T0, "AAAA", "XNAS", "B", 5.00, np.nan, np.nan, 200_000, 300_000)])
    issued: dict = {}
    first = E.run_window(h, uni(), CFG, "A", T0, CLOSE, issued)
    E.record_issued(first, issued)
    assert first.loc["AAAA", "action"] == "TRADE" and "AAAA" in issued
    again = E.run_window(h, uni(), CFG, "A", T0 + pd.Timedelta(seconds=90), CLOSE, issued)
    assert again.loc["AAAA", "action"] == "held"                      # never bought twice
    full = {f"X{i}": 1000.0 for i in range(CFG["account"]["max_positions"])}
    capped = E.run_window(h, uni(), CFG, "A", T0, CLOSE, full)
    assert capped.loc["AAAA", "reason"] == "max positions tonight"


def test_uk_times_across_dst_mismatch():
    # 27 Oct 2026: UK already on GMT, US still on EDT -> close is 20:00 UK, not 21:00
    assert C.fmt_times(C.close_time(dt.date(2026, 10, 27))) == "16:00:00 ET / 20:00:00 UK"
    assert C.fmt_times(C.close_time(dt.date(2026, 9, 29))) == "16:00:00 ET / 21:00:00 UK"


# ------------------------------------------------------------------ end to end on fake Databento
def test_backtest_end_to_end(tmp_path, monkeypatch):
    import backtest
    f = C.Fetcher(F.FakeHistorical(), tmp_path, max_usd=100, assume_yes=True)
    trades, calib = backtest.backtest(f, CFG, dt.date(2026, 9, 8), dt.date(2026, 9, 29))
    filled = trades[~trades["missed"]]
    assert set(filled.index) == {"AAAA"}                   # BBBB flipped, FFFF weak, DDDD/CCCC excluded
    assert set(filled["window"]) == {"A"}                  # ledger: AAAA is not bought again in B
    missed = trades[trades["missed"]]
    assert set(missed.index) == {"HHHH"}                   # ask ran away -> no fill, no P&L
    assert missed["pnl"].isna().all()
    t = filled.iloc[0]
    assert t["entry"] <= t["limit"]                        # paid the ask, which was within the limit
    assert t["exit"] == pytest.approx(F.official_close("AAAA", t["session"]))
    assert t["pnl"] == pytest.approx(t["shares"] * (t["exit"] - t["entry"]) - 2 * max(0.35, 0.0035 * t["shares"]))
    # calibration: close = ref * 1.008 and near gap 1.2% -> beta = 0.667 exactly for AAAA
    b = calib[(calib["window"] == "B") & (calib.index == "AAAA")]
    assert backtest.through_origin(b["realized"], b["near_gap"]) == pytest.approx(0.008 / 0.012, rel=1e-3)
    # second run is served from cache: no new data downloads
    fake2 = F.FakeHistorical()
    f2 = C.Fetcher(fake2, tmp_path, max_usd=100, assume_yes=True)
    backtest.backtest(f2, CFG, dt.date(2026, 9, 8), dt.date(2026, 9, 29))
    assert fake2.calls == []


def test_ticker_called_NA_survives_csv(tmp_path):
    import live_signals as L
    p = tmp_path / "u.csv"
    pd.DataFrame({"symbol": ["NA", "AAAA"], "prev_close": [8.0, 5.0], "adv_usd": [1e7, 1e7], "sigma": [0.03, 0.03],
                  "venue": ["XNAS", "XNAS"], "session": ["2026-09-29"] * 2}).to_csv(p, index=False)
    u = L.load_universe(str(p), dt.date(2026, 9, 29))
    assert list(u.index) == ["NA", "AAAA"] and u["prev_close"].dtype == float


def test_live_runner_with_fake_feed(tmp_path, monkeypatch, capsys):
    import live_signals as L
    day = dt.date(2026, 9, 29)
    close_at = C.close_time(day)
    universe = pd.DataFrame({s: dict(prev_close=F.ref_at_close(s, day), adv_usd=F.ref_at_close(s, day) * 2e6,
                                     sigma=0.03, venue="XNAS", session=str(day)) for s in ["AAAA", "BBBB", "FFFF"]}).T
    universe[["prev_close", "adv_usd", "sigma"]] = universe[["prev_close", "adv_usd", "sigma"]].astype(float)
    universe.index.name = "symbol"
    clock = {"t": close_at - pd.Timedelta(minutes=12)}

    class FakeLive:
        def __init__(self):
            self.symbology_map, self.cbs, self.subs = {}, [], []

        def subscribe(self, dataset, schema, symbols, start):
            self.subs.append((dataset, schema, tuple(symbols), start))

        def add_callback(self, cb):
            self.cbs.append(cb)

        def start(self):
            syms = [s for sub in self.subs for s in sub[2]]
            for s in syms:
                self.symbology_map[F.IID[s]] = s
            for r in F.imbalance_records("XNAS.ITCH", syms, close_at - pd.Timedelta(minutes=11), close_at):
                for cb in self.cbs:
                    cb(r)

        def stop(self):
            pass

        def block_for_close(self, timeout=None):
            pass

    def sleep(sec):
        clock["t"] += pd.Timedelta(seconds=sec)

    monkeypatch.setenv("DATABENTO_API_KEY", "db-test")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    L.live(None, CFG, universe, day, close_at, now=lambda: clock["t"], sleep=sleep, live_factory=FakeLive)
    out = capsys.readouterr().out
    assert out.count("WINDOW A") == 2 and out.count("WINDOW B") == 2
    assert "AAAA" in out and ">> BUY" in out and "flipped" in out
    assert "21:00:00 UK" in out
    log = pd.read_csv(tmp_path / "results" / f"ideas_{day}.csv")
    first = log[(log["window"] == "A") & (log["label"] == "signal") & (log["action"] == "TRADE")]
    assert set(first["symbol"]) == {"AAAA"}           # BBBB flipped inside the 60 s look-back
    # by the last call the flip is >60 s old, so BBBB is allowed again (look-back is configurable)
    assert "BBBB" in set(log.loc[(log["label"] == "last call") & (log["action"] == "TRADE"), "symbol"])
    # ledger: each symbol is a TRADE at most once per evening
    assert log.loc[log["action"] == "TRADE", "symbol"].is_unique
    assert "records so far" in out
