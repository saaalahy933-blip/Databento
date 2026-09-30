"""Buy-and-hold variant: what the backtest's entries would have made held for a % target.

Reads a backtest_trades.csv (the entry fills), and daily bars already in ./cache (nothing is bought).
The close-exit results are not touched.

    python hold_test.py --trades results/backtest_trades.csv --target 0.10 --stop 0.05 --max-days 20 \
        --fit-end 2026-06-15

Exit rules, checked each session after the entry day, in order:
  1. open at or below the stop   -> sell at the open (gap down)
  2. open at or above the target -> sell at the open (gap up; a resting limit fills there)
  3. low at or below the stop    -> sell at the stop (if the same day also reaches the target, the stop counts first)
  4. high at or above the target -> sell at the target
  5. after max-days sessions     -> sell at that day's close
Trades that run out of data first are reported as open.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import pathlib

import numpy as np
import pandas as pd

from imbalance import common as C


def load_daily_bars(cache_dir) -> pd.DataFrame:
    """Every cached EQUS.SUMMARY daily bar: date, symbol, open, high, low, close."""
    parts = [pd.read_pickle(p) for p in glob.glob(str(cache_dir / "EQUS.SUMMARY_ohlcv-1d_*.pkl"))]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit(f"No cached daily bars in {cache_dir}. Run backtest.py first.")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["ts_event"], utc=True).dt.date
    return (df[["date", "symbol", "open", "high", "low", "close"]].dropna(subset=["symbol"])
            .drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"]))


def hold_exit(entry: float, after: pd.DataFrame, target: float, stop: float | None,
              max_days: int) -> tuple[float, str, int]:
    """(exit price, reason, sessions held) for one entry; `after` = daily bars after the entry day, in order."""
    up = entry * (1 + target)
    down = entry * (1 - stop) if stop else -math.inf
    for k, bar in enumerate(after.head(max_days).itertuples(index=False), 1):
        if bar.open <= down:
            return bar.open, "stop", k
        if bar.open >= up:
            return bar.open, "target", k
        if bar.low <= down:
            return down, "stop", k
        if bar.high >= up:
            return up, "target", k
        if k == max_days:
            return bar.close, "time", k
    return np.nan, "open", len(after.head(max_days))


def run(trades: pd.DataFrame, bars: pd.DataFrame, cfg: dict, target: float, stop: float | None,
        max_days: int) -> pd.DataFrame:
    c = cfg["costs"]
    by_sym = {s: g for s, g in bars.groupby("symbol")}
    rows = []
    for sym, t in trades.iterrows():
        g = by_sym.get(sym, bars.iloc[:0])
        after = g[g["date"] > pd.Timestamp(t["session"]).date()]
        px, reason, days = hold_exit(float(t["entry"]), after, target, stop, max_days)
        comm = max(c["min_commission"], c["commission_per_share"] * t["shares"])
        pnl = t["shares"] * (px - t["entry"]) - 2 * comm if reason != "open" else np.nan
        rows.append({"symbol": sym, "session": t["session"], "window": t["window"], "entry": t["entry"],
                     "shares": t["shares"], "exit": px, "reason": reason, "days": days,
                     "ret_pct": (px / t["entry"] - 1) * 100, "pnl": pnl})
    return pd.DataFrame(rows)


def summary(h: pd.DataFrame, fit_end: dt.date) -> pd.DataFrame:
    h = h.assign(period=np.where(pd.to_datetime(h["session"]).dt.date <= fit_end, "FIT", "TEST"))
    out = []
    groups = [(key, g) for key, g in h.groupby(["period", "window"])]
    groups += [((period, "All"), g) for period, g in h.groupby("period")]
    for (period, window), g in groups:
        done = g[g["reason"] != "open"]
        out.append({"period": period, "window": window, "trades": len(g), "closed": len(done),
                    "win_%": round((done["pnl"] > 0).mean() * 100, 1) if len(done) else np.nan,
                    "avg_%": round(done["ret_pct"].mean(), 2) if len(done) else np.nan,
                    "pnl_$": round(done["pnl"].sum()), "target": int((g["reason"] == "target").sum()),
                    "stop": int((g["reason"] == "stop").sum()), "time": int((g["reason"] == "time").sum()),
                    "open": int((g["reason"] == "open").sum()),
                    "avg_days": round(done["days"].mean(), 1) if len(done) else np.nan})
    return pd.DataFrame(out).sort_values(["period", "window"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", default=str(C.ROOT / "results" / "backtest_trades.csv"))
    ap.add_argument("--cache", default=str(C.ROOT / "cache"))
    ap.add_argument("--config")
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--stop", type=float, default=0.05, help="0 = no stop")
    ap.add_argument("--max-days", type=int, default=20)
    ap.add_argument("--fit-end", help="last FIT session; use the one backtest.py printed "
                                      "(default: middle of the sessions that have trades)")
    ap.add_argument("--out")
    a = ap.parse_args()
    trades = pd.read_csv(a.trades, index_col=0, keep_default_na=False, na_values=[""])
    trades = trades[trades["missed"].astype(str) == "False"]
    sessions = sorted(pd.to_datetime(trades["session"]).dt.date.unique())
    fit_end = dt.date.fromisoformat(a.fit_end) if a.fit_end else sessions[(len(sessions) - 1) // 2]
    bars = load_daily_bars(pathlib.Path(a.cache))
    h = run(trades, bars, C.load_config(a.config), a.target, a.stop or None, a.max_days)
    if a.out:
        h.to_csv(a.out, index=False)
    print(f"HOLD TEST  target +{a.target:.0%}  stop {'-' + format(a.stop, '.0%') if a.stop else 'none'}  "
          f"max {a.max_days} sessions  FIT up to {fit_end}  (bars cached up to {bars['date'].max()})")
    print(summary(h, fit_end).to_string(index=False))


if __name__ == "__main__":
    main()
