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


def test_momentum_filters_by_hand():
    cfg = {**CFG, "signal": {**CFG["signal"], "min_day_move": 0.03, "min_move_to_sigma": 1.5}}
    specs = {"FLAT": 5.00, "UP2": 5.10, "UP5": 5.25}              # ref vs prev close 5.00
    rows = [(T0, s, "XNAS", "B", ref, np.nan, np.nan, 200_000, 300_000) for s, ref in specs.items()]
    u = pd.DataFrame({s: dict(prev_close=5.0, adv_usd=10e6, sigma=0.04, venue="XNAS") for s in specs}).T
    u[["prev_close", "adv_usd", "sigma"]] = u[["prev_close", "adv_usd", "sigma"]].astype(float)
    feat = E.features(E.snapshot(hist_rows(rows), T0, 60, 30), u, cfg, "A")
    assert feat.loc["UP5", "day_move"] == pytest.approx(0.05)
    assert feat.loc["UP5", "move_sigma"] == pytest.approx(1.25)   # 5% / 4% sigma
    out = E.apply_filters(feat, cfg, "A")
    assert out.loc["FLAT", "reason"] == out.loc["UP2", "reason"] == "not up enough today"
    assert out.loc["UP5", "reason"] == "move small vs sigma"      # up 5% but sigma is 4%: only 1.25x
    loose = {**cfg, "signal": {**cfg["signal"], "min_move_to_sigma": 1.2}}
    assert E.apply_filters(feat, loose, "A").loc["UP5", "reason"] != "move small vs sigma"
    assert E.apply_filters(feat, CFG, "A").loc["FLAT", "reason"] != "not up enough today"   # off by default


def test_activity_measures_by_hand():
    day, prev = dt.date(2026, 9, 29), dt.date(2026, 9, 28)
    at = pd.Timestamp("2026-09-29 15:53", tz=C.ET).tz_convert("UTC")
    bar = lambda d, hhmm, s, v: (pd.Timestamp(f"{d} {hhmm}", tz=C.ET).tz_convert("UTC"), s, v)
    bars = pd.DataFrame([bar(prev, "10:00", "X", 100), bar(prev, "15:59", "X", 10_000),   # after 15:53: ignored
                         bar(day, "09:00", "X", 10_000),                                     # pre-market: ignored
                         bar(day, "10:00", "X", 250), bar(day, "15:54", "X", 10_000)],
                        columns=["ts_event", "symbol", "volume"])
    assert E.rel_volume(bars, day, at, 20, 1)["X"] == pytest.approx(2.5)
    assert np.isnan(E.rel_volume(bars, day, at, 20, 2)["X"])                  # too little history
    trades = pd.DataFrame({"ts_event": [at - pd.Timedelta(hours=1), at + pd.Timedelta(seconds=1)],
                           "symbol": ["X", "X"], "size": [300_000, 900_000]})
    assert E.max_block(trades, at)["X"] == 300_000                            # the later print is in the future
    opts = pd.DataFrame({"symbol": [F.osi("X", "C"), F.osi("X", "P"), F.osi("Y", "C"), "junk"],
                         "volume": [300, 100, 50, 999]})
    cp = E.call_put_ratio(opts)
    assert cp["X"] == pytest.approx(3.0) and cp["Y"] == np.inf


def test_network_timeouts_are_retried(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(C.Fetcher, "RETRY_WAITS_S", (0, 0, 0))
    client = _Flaky([])
    calls = {"n": 0}
    real = client.timeseries.get_range

    def get_range(**req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ReadTimeout("read timed out")
        return real(**req)
    client.timeseries = F.NS(get_range=get_range)
    df = C.Fetcher(client, tmp_path, max_usd=100, assume_yes=True).get(**C.req_definitions("XNAS.ITCH", dt.date(2026, 9, 4)))
    assert not df.empty and calls["n"] == 2


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
    monkeypatch.setattr(C, "ROOT", tmp_path)               # keep fake results out of the real results/
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


def test_parallel_prefetch_buys_nothing_extra(tmp_path, monkeypatch):
    import backtest
    runs = {}
    for workers in (1, 8):
        monkeypatch.setattr(C, "ROOT", tmp_path / str(workers))
        fake = F.FakeHistorical()
        f = C.Fetcher(fake, tmp_path / str(workers), max_usd=100, assume_yes=True)
        trades, _ = backtest.backtest(f, CFG, dt.date(2026, 9, 8), dt.date(2026, 9, 29), workers=workers)
        runs[workers] = (sorted(fake.calls), trades.sort_index().to_csv())
    assert runs[8][0] == runs[1][0]                                # same requests, none twice
    assert runs[8][1] == runs[1][1]                                # same trades


class _Flaky:
    """get_range fails with the given HTTP statuses, then answers like the fake API."""

    def __init__(self, statuses):
        self.statuses, self.fake = list(statuses), F.FakeHistorical()
        self.metadata, self.attempts = self.fake.metadata, 0
        self.timeseries = F.NS(get_range=self._get_range)

    def _get_range(self, **req):
        self.attempts += 1
        if self.statuses:
            raise F.db.BentoServerError(http_status=self.statuses.pop(0), message="flaky")
        return self.fake.timeseries.get_range(**req)


def test_server_errors_are_retried_then_succeed(tmp_path, monkeypatch):
    monkeypatch.setattr(C.Fetcher, "RETRY_WAITS_S", (0, 0, 0))
    client = _Flaky([504, 502])
    f = C.Fetcher(client, tmp_path, max_usd=100, assume_yes=True)
    df = f.get(**C.req_definitions("XNAS.ITCH", dt.date(2026, 9, 4)))
    assert not df.empty and client.attempts == 3
    client = _Flaky([503])
    client.metadata = F.NS(get_cost=lambda **q: client.timeseries.get_range(**q) and 0.5)   # 503 once, then 0.5
    assert C.Fetcher(client, tmp_path / "q", max_usd=100, assume_yes=True).quote(
        C.req_definitions("XNAS.ITCH", dt.date(2026, 9, 3))) == 0.5          # cost quotes are retried too


def test_client_errors_and_persistent_server_errors_are_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(C.Fetcher, "RETRY_WAITS_S", (0, 0, 0))
    req = C.req_definitions("XNAS.ITCH", dt.date(2026, 9, 4))
    client = _Flaky([400])
    with pytest.raises(F.db.BentoServerError):
        C.Fetcher(client, tmp_path / "a", max_usd=100, assume_yes=True).get(**req)
    assert client.attempts == 1                                    # a 4xx is never retried
    client = _Flaky([504] * 4)
    with pytest.raises(F.db.BentoServerError):
        C.Fetcher(client, tmp_path / "b", max_usd=100, assume_yes=True).get(**req)
    assert client.attempts == 4                                    # 3 retries, then give up


def test_listings_on_a_monday_holiday_step_back_to_friday(tmp_path, monkeypatch):
    from imbalance import build as B
    assert B.weekdays_back(dt.date(2026, 9, 7), 3) == [dt.date(2026, 9, 7), dt.date(2026, 9, 4), dt.date(2026, 9, 3)]
    labor_day = dt.date(2026, 9, 7)
    monkeypatch.setattr(F, "SESSIONS", [x for x in F.SESSIONS if x != labor_day])
    fake = F.FakeHistorical()
    f = C.Fetcher(fake, tmp_path, max_usd=100, assume_yes=True)
    listings = B.fetch_listings(f, CFG, [labor_day], "test")      # a weekend request would raise a 504
    assert listings[labor_day]["AAAA"] == "XNAS"
    assert len(fake.calls) == 2                                    # Mon (empty), then Fri


def test_backtest_with_activity_filters(tmp_path, monkeypatch):
    import backtest
    on = {"min_rel_volume": 1.5, "min_block_shares": 300_000, "min_call_put_ratio": 1.5}
    runs = {}
    for name, extra in {"on": {}, "big block": {"min_block_shares": 500_000}}.items():
        monkeypatch.setattr(C, "ROOT", tmp_path / name)
        cfg = {**CFG, "signal": {**CFG["signal"], **on, **extra}}
        fake = F.FakeHistorical()
        f = C.Fetcher(fake, tmp_path / name, max_usd=100, assume_yes=True)
        trades, _ = backtest.backtest(f, cfg, dt.date(2026, 9, 8), dt.date(2026, 9, 29))
        runs[name] = (trades[~trades["missed"]], fake)
    filled, fake = runs["on"]
    assert set(filled.index) == {"AAAA"} and set(filled["window"]) == {"A"}   # 3x volume, 400k block, calls 2:1
    asked = {c for c in fake.calls if c[1] in ("ohlcv-1m", "trades") or c[0] == "OPRA.PILLAR"}
    assert asked == {("XNAS.BASIC", "ohlcv-1m"), ("XNAS.BASIC", "trades"), ("OPRA.PILLAR", "ohlcv-1d")}
    filled, _ = runs["big block"]
    # no 500k print before 15:53, but the 900k print at 15:54:30 counts for the 15:56 window
    assert set(filled.index) == {"AAAA"} and set(filled["window"]) == {"B"}


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


def test_api_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("DATABENTO_API_KEY=db-from-dotenv\n")
    assert C.api_key() == "db-from-dotenv"


def test_api_key_missing_exits_with_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("DATABENTO_API_KEY=\n")
    with pytest.raises(SystemExit) as e:
        C.api_key()
    assert "DATABENTO_API_KEY is not set" in str(e.value) and ".env" in str(e.value)
