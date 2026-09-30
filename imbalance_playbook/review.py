"""Step 3 (next day): score yesterday's TRADE ideas against the official close.

    python review.py                       # latest results/ideas_*.csv
    python review.py --date 2026-09-29 --fills fills.csv

Uses the entry limit as the assumed fill unless you pass --fills (CSV with
columns symbol,window,fill_price,shares from your broker). Compare with the
backtest: if live results are clearly worse, stop and find out why.
"""
from __future__ import annotations

import argparse
import datetime as dt

import databento as db
import numpy as np
import pandas as pd

from imbalance import build as B
from imbalance import common as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--fills")
    ap.add_argument("--close", default="16:00")
    ap.add_argument("--config")
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    cfg = C.load_config(a.config)

    files = sorted((C.ROOT / "results").glob("ideas_*.csv"))
    path = C.ROOT / "results" / f"ideas_{a.date}.csv" if a.date else (files[-1] if files else None)
    if path is None or not path.exists():
        raise SystemExit("No ideas file found. Run live_signals.py first.")
    session = dt.date.fromisoformat(path.stem.split("_")[1])
    ideas = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"symbol": str})
    t = ideas[ideas["action"] == "TRADE"].drop_duplicates(["symbol", "window"], keep="first").copy()
    if t.empty:
        raise SystemExit(f"No TRADE rows in {path.name}.")
    if a.fills:
        fills = pd.read_csv(a.fills)
        t = t.merge(fills, on=["symbol", "window"], how="left", suffixes=("", "_fill"))
        t["entry"] = t["fill_price"].fillna(t["limit"])
        t["shares"] = t["shares_fill"].fillna(t["shares"]) if "shares_fill" in t else t["shares"]
    else:
        t["entry"] = t["limit"]

    client = db.Historical(C.api_key())
    f = C.Fetcher(client, C.ROOT / "cache", a.max_cost, a.yes)
    close_at = C.close_time(session, a.close)
    by_venue = {v: sorted(g["symbol"]) for v, g in t.groupby("venue")}
    f.approve(sum(f.quote(r) for r in B.close_requests(cfg, by_venue, close_at)), "official closing prices")
    closes = B.official_closes(f, cfg, by_venue, close_at)

    c = cfg["costs"]
    t["close"] = t["symbol"].map(closes)
    comm = np.maximum(c["min_commission"], c["commission_per_share"] * t["shares"])
    t["pnl"] = t["shares"] * (t["close"] - t["entry"]) - 2 * comm
    loc_missed = (t["window"] == "B") & (t["close"] < t["ref"] * (1 - cfg["execution"]["loc_buffer"]))
    if loc_missed.any():
        print("!! LOC sell NOT filled (close below your LOC price) — you still hold: "
              + ", ".join(t.loc[loc_missed, "symbol"]) + ". P&L below is marked at the close.")
    t["ret_bps"] = (t["close"] / t["entry"] - 1) * 1e4
    t["predicted_bps"] = t["exp_move"] * 1e4
    cols = ["window", "symbol", "shares", "entry", "close", "predicted_bps", "ret_bps", "pnl"]
    print(f"\nReview of {session}\n")
    print(t[cols].round(2).to_string(index=False))
    print(f"\nTotal P&L ${t['pnl'].sum():,.2f} on {len(t)} trades; win rate {100*(t['pnl']>0).mean():.0f}%.")


if __name__ == "__main__":
    main()
