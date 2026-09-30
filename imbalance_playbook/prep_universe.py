"""Step 1 (UK afternoon, any time before 20:45): build tonight's universe.

    python prep_universe.py              # universe for today's US session
    python prep_universe.py --session 2026-09-29   # for --replay of a past day

Writes universe.csv: Nasdaq-listed stocks priced ~$1-10 with 20-day $ADV
and daily volatility (sigma), using only data from BEFORE the session.
Typical cost: cents. The exact quote is shown before anything is bought.
"""
from __future__ import annotations

import argparse
import datetime as dt

import databento as db
import pandas as pd

from imbalance import build as B
from imbalance import common as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="US session date YYYY-MM-DD (default: today in New York)")
    ap.add_argument("--out", default="universe.csv")
    ap.add_argument("--config")
    ap.add_argument("--max-cost", type=float, default=5.0, help="abort if the quote is above this ($)")
    ap.add_argument("--yes", action="store_true", help="don't ask before buying")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    session = dt.date.fromisoformat(a.session) if a.session else dt.datetime.now(C.ET).date()
    client = db.Historical(C.api_key())
    f = C.Fetcher(client, C.ROOT / "cache", a.max_cost, a.yes)

    defs_last = B.latest_available(client, cfg["datasets"]["XNAS"], "definition")
    bars_last = B.latest_available(client, "EQUS.SUMMARY", "ohlcv-1d")
    listing_date = min(session, defs_last)
    bars_end = min(session - dt.timedelta(days=1), bars_last)
    if (session - bars_end).days > 4:
        print(f"WARNING: newest daily bar available is {bars_end}; stats will be a little stale.")

    uni = B.build_universe(f, cfg, session, listing_date, bars_end)
    uni = uni.assign(session=session.isoformat()).sort_values("adv_usd", ascending=False)
    uni.to_csv(a.out)
    print(f"\nUniverse for {session}: {len(uni):,} symbols -> {a.out}")
    print(f"(listing venues as of {listing_date}, daily bars up to {bars_end})")
    with pd.option_context("display.width", 120):
        print(uni.head(10).round(4))


if __name__ == "__main__":
    main()
