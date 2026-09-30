"""Builds the inputs the engine needs from Databento historical data.

Shared by prep_universe.py (tonight's universe), backtest.py and the
--replay mode of live_signals.py. Every purchase goes through Fetcher, which
quotes the exact price first and caches the result on disk.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import common as C
from . import engine as E

BAR_LOOKBACK_CAL_DAYS = 45      # enough calendar days for 20 sessions + holidays


def latest_available(client, dataset: str, schema: str) -> dt.date:
    """Last full date the historical API can serve for this dataset/schema."""
    rng = client.metadata.get_dataset_range(dataset=dataset)
    end = pd.Timestamp(rng.get("schema", {}).get(schema, rng)["end"])
    return (end - pd.Timedelta(nanoseconds=1)).date()


def definition_dates(first: dt.date, last: dt.date) -> list[dt.date]:
    """First weekday of every month from `first` to `last` (listing venue refresh points)."""
    out, d = [], dt.date(first.year, first.month, 1)
    while d <= last:
        wd = d
        while wd.weekday() >= 5:
            wd += dt.timedelta(days=1)
        out.append(max(wd, first) if d.year == first.year and d.month == first.month else wd)
        d = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def weekdays_back(d: dt.date, n: int) -> list[dt.date]:
    """`d` (if a weekday) and the weekdays before it, newest first, `n` in total."""
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def fetch_listings(f: C.Fetcher, cfg: dict, dates: list[dt.date], what: str) -> dict[dt.date, pd.Series]:
    """symbol -> listing venue, as of each date (steps back over holidays, never onto a weekend:
    Databento answers weekend definition requests with a 504 instead of an empty result)."""
    ds = cfg["datasets"]["XNAS"]
    reqs = []
    for d in dates:
        reqs.append([C.req_definitions(ds, w) for w in weekdays_back(d, 4)])
    f.approve(sum(f.quote(r[0]) for r in reqs), what)
    out = {}
    for d, tries in zip(dates, reqs):
        for r in tries:
            df = f.get(**r)
            if not df.empty:
                out[d] = C.parse_listing(df)
                break
    return out


def fetch_bars(f: C.Fetcher, symbols: list[str], start: dt.date, end: dt.date, what: str) -> pd.DataFrame:
    reqs = [C.req_daily_bars(ch, start, end) for ch in C.chunks(sorted(symbols), 2000)]
    f.approve(sum(f.quote(r) for r in reqs), what)
    parts = [C.parse_bars(f.get(**r)) for r in reqs]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "symbol", "close", "volume"])


def stats_with_row(bars: pd.DataFrame, cfg: dict, session: dt.date) -> dict[str, pd.DataFrame]:
    """universe_stats, making sure `session` has a row even if it has no bar yet (live use)."""
    return E.universe_stats(bars, cfg, extra_dates=[session])


def listing_for(listings: dict[dt.date, pd.Series], session: dt.date) -> pd.Series:
    """Most recent listing map on or before `session`."""
    keys = [d for d in sorted(listings) if d <= session]
    return listings[keys[-1]] if keys else listings[min(listings)]


def enabled_symbols(listings: dict[dt.date, pd.Series], cfg: dict) -> list[str]:
    venues = set(cfg["universe"]["venues"])
    syms = set()
    for s in listings.values():
        syms |= set(s[s.isin(venues)].index)
    return sorted(syms)


def imbalance_history(f: C.Fetcher, cfg: dict, universe: pd.DataFrame, start: pd.Timestamp,
                      end: pd.Timestamp) -> pd.DataFrame:
    """Closing imbalance records for the universe between start and end (UTC)."""
    parts = []
    for venue, grp in universe.groupby("venue"):
        ds = cfg["datasets"][venue]
        types = cfg["closing_auction_types"][ds]
        for ch in C.chunks(grp.index, 2000):
            raw = f.get(**C.req_imbalance(ds, ch, start, end))
            parts.append(E.clean_imbalance(raw, venue, types))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True).sort_values("ts") if parts else pd.DataFrame(columns=E.IMB_COLUMNS)


def imbalance_requests(cfg: dict, universe: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    out = []
    for venue, grp in universe.groupby("venue"):
        for ch in C.chunks(grp.index, 2000):
            out.append(C.req_imbalance(cfg["datasets"][venue], ch, start, end))
    return out


def close_requests(cfg: dict, symbols_by_venue: dict[str, list[str]], close_at: pd.Timestamp) -> list[dict]:
    out = []
    for venue, syms in symbols_by_venue.items():
        for ch in C.chunks(syms, 2000):
            out.append(C.req_closes(cfg["datasets"][venue], ch, close_at))
    return out


def first_fill(f: C.Fetcher, cfg: dict, trades: pd.DataFrame, at: pd.Timestamp, wait_s: float) -> pd.Series:
    """symbol -> first Nasdaq ask <= limit within wait_s after `at` (NaN = the buy would not have filled).

    Uses 1-second Nasdaq BBO (bbo-1s), so it is a proxy for the national best offer.
    """
    out = pd.Series(np.nan, index=trades.index, dtype=float)
    for venue, grp in trades.groupby("venue"):
        raw = f.get(**C.req_bbo(cfg["datasets"][venue], list(grp.index), at, at + pd.Timedelta(seconds=wait_s)))
        if raw.empty:
            continue
        q = raw[["symbol", "ask_px_00"]].dropna()
        q = q[q["ask_px_00"] > 0].join(grp["limit"], on="symbol")
        hit = q[q["ask_px_00"] <= q["limit"] + 1e-9]
        out.update(hit.groupby("symbol")["ask_px_00"].first())
    return out


def official_closes(f: C.Fetcher, cfg: dict, symbols_by_venue: dict[str, list[str]], close_at: pd.Timestamp) -> pd.Series:
    parts = [C.parse_closes(f.get(**r)) for r in close_requests(cfg, symbols_by_venue, close_at)]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def build_universe(f: C.Fetcher, cfg: dict, session: dt.date, listing_date: dt.date,
                   bars_end: dt.date) -> pd.DataFrame:
    """Universe for one session from listings on `listing_date` and bars up to `bars_end`."""
    listings = fetch_listings(f, cfg, [listing_date], "listing venues (1 day of definitions)")
    if not listings:
        raise SystemExit(f"No definitions found around {listing_date}.")
    syms = enabled_symbols(listings, cfg)
    bars = fetch_bars(f, syms, bars_end - dt.timedelta(days=BAR_LOOKBACK_CAL_DAYS), bars_end,
                      f"daily bars for {len(syms):,} symbols")
    bars = bars[bars["date"] < session]
    stats = stats_with_row(bars, cfg, session)
    return E.universe_for(stats, session, listing_for(listings, session), cfg)


# --------------------------------------------------------------------------- activity filters (optional)
def activity_requests(cfg: dict, symbols: list[str], session: dt.date, until: pd.Timestamp,
                      prior_dates: list[dt.date]) -> list[dict]:
    """Data for the activity filters that are switched on, for `symbols` on `session` up to `until`."""
    a, sig = cfg["activity"], cfg["signal"]
    if not symbols:
        return []
    out = []
    if "min_rel_volume" in sig:
        first = (prior_dates[-a["rel_volume_days"]:] or [session])[0]
        out.append(C.req_minute_bars(a["volume_dataset"], symbols, C.session_open(first), until))
    if "min_block_shares" in sig:
        out.append(C.req_trades(a["trades_dataset"], symbols, C.session_open(session), until))
    if "min_call_put_ratio" in sig:
        out.append(C.req_option_volume(a["options_dataset"], symbols, session))
    return out


def _rejected(e: Exception) -> bool:
    return getattr(e, "http_status", None) == 422     # e.g. a stock with no listed options


def get_lenient(f: C.Fetcher, req: dict) -> pd.DataFrame:
    """f.get, but if Databento rejects the symbol list, ask symbol by symbol and skip the ones it can't resolve."""
    try:
        return f.get(**req)
    except Exception as e:
        if not _rejected(e):
            raise
    if len(req["symbols"]) == 1:
        return pd.DataFrame()
    parts = [get_lenient(f, {**req, "symbols": [s]}) for s in req["symbols"]]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def quote_lenient(f: C.Fetcher, req: dict) -> float:
    try:
        return f.quote(req)
    except Exception as e:
        if not _rejected(e):
            raise
    if len(req["symbols"]) == 1:
        return 0.0
    return sum(quote_lenient(f, {**req, "symbols": [s]}) for s in req["symbols"])


def fetch_activity(f: C.Fetcher, cfg: dict, symbols: list[str], session: dt.date, at: pd.Timestamp,
                   until: pd.Timestamp, prior_dates: list[dt.date]) -> pd.DataFrame:
    """symbol -> rel_volume, max_block, cp_ratio (only the switched-on ones), measured before `at`."""
    a = cfg["activity"]
    out = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    for req in activity_requests(cfg, symbols, session, until, prior_dates):
        raw = get_lenient(f, req)
        if req["schema"] == "ohlcv-1m":
            out["rel_volume"] = E.rel_volume(raw, session, at, a["rel_volume_days"], a["min_volume_days"]).reindex(out.index)
        elif req["schema"] == "trades":
            out["max_block"] = E.max_block(raw, at).reindex(out.index)
        else:
            out["cp_ratio"] = E.call_put_ratio(raw).reindex(out.index)
    return out
