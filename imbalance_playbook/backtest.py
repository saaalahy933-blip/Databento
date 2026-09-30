"""Step 0 (do this first): test the playbook on past data, out of sample.

    python backtest.py --start 2025-10-01 --end 2026-09-29
    python backtest.py --start 2025-10-01 --end 2026-09-29 --fit-end 2026-03-31

How it simulates, per session:
  * imbalance snapshot at each decision time -> exactly the same engine as live_signals.py
  * one ledger per session: a stock bought in Window A is not bought again in B,
    and max_positions / max_gross_usd count the whole evening
  * BUY fills only if the Nasdaq ask (1-second BBO) is at or below the limit within
    fill_wait_s; you pay that ask. No fill = no trade (these are often the winners)
  * exit at the official closing-cross price. Window B LOC: if the close is below the
    LOC price, the sell doesn't fill and the position is sold at the next session's open
  * commissions on both legs, minimum ticket included

Calibration (k_impact, beta_near) is fitted ONLY on sessions up to --fit-end
(default: the middle of the range). Judge the strategy on the TEST period.

Nothing is bought until you see the cost estimate and type y. Downloads are cached
in ./cache, so re-running with new thresholds costs nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor

import databento as db
import numpy as np
import pandas as pd

from imbalance import build as B
from imbalance import common as C
from imbalance import engine as E


def simulate(ideas, fills, closes, next_open, cfg, session, window) -> pd.DataFrame:
    """P&L for TRADE rows that would have filled."""
    c = cfg["costs"]
    t = ideas[ideas["action"] == "TRADE"].copy()
    if t.empty:
        return t
    t["entry"] = fills.reindex(t.index)
    t["close"] = closes.reindex(t.index)
    t["missed"] = t["entry"].isna()
    t["exit"], t["overnight"] = t["close"], False
    if window == "B":
        loc = t["ref"] * (1 - cfg["execution"]["loc_buffer"])
        unfilled = t["close"] < loc
        t.loc[unfilled, "exit"] = next_open.reindex(t.index)[unfilled]
        t.loc[unfilled, "overnight"] = True
    comm = np.maximum(c["min_commission"], c["commission_per_share"] * t["shares"])
    t["pnl"] = t["shares"] * (t["exit"] - t["entry"]) - 2 * comm
    t["ret_bps"] = (t["exit"] / t["entry"] - 1) * 1e4
    t["session"], t["window"] = session, window
    return t


def calib_rows(ideas: pd.DataFrame, closes: pd.Series, cfg: dict, session, window: str) -> pd.DataFrame:
    """Buy-side rows big enough to matter, with the move that actually happened (ref -> close)."""
    sig = cfg["signal"]
    r = ideas[(ideas["notional"] >= sig["min_notional_usd"]) & (ideas["imb_adv"] >= sig["min_imb_to_adv"])].copy()
    r["realized"] = closes.reindex(r.index) / r["ref"] - 1
    r["session"], r["window"] = session, window
    return r[r["realized"].notna()]


def through_origin(y: pd.Series, x: pd.Series) -> float:
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    den = float((x[m] ** 2).sum())
    return float((x[m] * y[m]).sum() / den) if den > 0 else float("nan")


def perf_line(name: str, t: pd.DataFrame, sessions: list) -> str:
    done = t[~t["missed"] & t["exit"].notna()] if not t.empty else t
    if done.empty:
        return f"  {name:<18} no filled trades"
    daily = done.groupby("session")["pnl"].sum().reindex(sessions, fill_value=0.0)   # flat days count
    eq = daily.cumsum()
    dd = float((eq - eq.cummax()).min())
    sharpe = daily.mean() / daily.std() * math.sqrt(252) if daily.std() > 0 else float("nan")
    miss = 100 * t["missed"].mean()
    ovn = int(done["overnight"].sum())
    return (f"  {name:<18} filled {len(done):>4} (missed {miss:4.0f}%)  win {100*(done['pnl']>0).mean():5.1f}%  "
            f"avg {done['ret_bps'].mean():6.1f} bps  P&L ${done['pnl'].sum():>9,.0f}  "
            f"Sharpe {sharpe:5.2f}  maxDD ${dd:,.0f}  overnight {ovn}")


def report(trades: pd.DataFrame, calib: pd.DataFrame, cfg: dict, sessions: list, fit_end) -> None:
    fit = [d for d in sessions if d <= fit_end]
    test = [d for d in sessions if d > fit_end]
    print("\n" + "=" * 110)
    print(f"BACKTEST  fit period {fit[0] if fit else '-'}..{fit_end}   TEST period {test[0] if test else '-'}..{sessions[-1]}")
    print("=" * 110)
    for per_name, per in [("FIT", fit), ("TEST (judge this)", test)]:
        print(per_name)
        for w in ["A", "B", None]:
            t = trades[trades["session"].isin(per) & ((trades["window"] == w) if w else True)] if not trades.empty else trades
            print(perf_line(f"Window {w}" if w else "All", t, per))

    print("\nCALIBRATION from the FIT period only (paste into config.toml, then re-run; cache makes it free)")
    sig = cfg["signal"]
    cf = calib[calib["session"].isin(fit)] if not calib.empty else calib
    a = cf[cf["window"] == "A"] if not cf.empty else cf
    b = cf[cf["window"] == "B"] if not cf.empty else cf
    if not a.empty:
        x = a["sigma"] * np.sqrt(a["imb_adv"])
        print(f"  k_impact  = {through_origin(a['realized'], x):.3f}   (now {sig['k_impact']}; {len(a):,} observations)")
        if len(a) >= 50:
            q = pd.qcut(x.rank(method="first"), 5, labels=False)
            tbl = pd.DataFrame({"predicted_bps": sig["k_impact"] * x * 1e4, "realized_bps": a["realized"] * 1e4}).groupby(q).mean()
            print("  Window A by quintile of sigma*sqrt(imb/ADV):\n    " + tbl.round(1).to_string().replace("\n", "\n    "))
    if not b.empty:
        g = b["near_gap"].where(b["near_gap"].abs() <= sig["max_near_gap"])
        print(f"  beta_near = {through_origin(b['realized'], g):.3f}   (now {sig['beta_near']}; {int(g.notna().sum()):,} observations)")
    print("\nIf the TEST period is not clearly positive after costs, do not trade it. Small samples mislead.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--fit-end", help="last session used for calibration (default: middle of the range)")
    ap.add_argument("--windows", default="AB")
    ap.add_argument("--close", default="16:00")
    ap.add_argument("--config")
    ap.add_argument("--max-cost", type=float, default=25.0, help="abort if any one estimate exceeds this ($)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads (1 = one at a time)")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    client = db.Historical(C.api_key())
    f = C.Fetcher(client, C.ROOT / "cache", a.max_cost, a.yes)
    fit_end = dt.date.fromisoformat(a.fit_end) if a.fit_end else None
    backtest(f, cfg, dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), a.windows, a.close, fit_end,
             a.workers)


def prefetch(f: C.Fetcher, reqs: list[dict], workers: int, get=None) -> None:
    """Download (and cache) many requests at once; the day-by-day loop then reads them from the cache."""
    todo = [r for r in reqs if not f.cached(r)]
    if workers <= 1 or not todo:
        return
    get = get or (lambda r: f.get(**r))
    print(f"  downloading {len(todo):,} requests, {workers} at a time")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(get, todo):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  {done:,}/{len(todo):,} downloaded")


def plan_activity(f: C.Fetcher, cfg: dict, plan: list, windows: str, offsets: dict, fetch_back: dict,
                  all_dates: list, workers: int) -> dict:
    """session -> (candidate symbols, data end time, prior sessions); quotes, asks once, then downloads."""
    out, reqs = {}, []
    for d, uni, close_at, *_ in plan:
        syms, last_at = set(), None
        for w in windows:
            at = close_at - pd.Timedelta(seconds=offsets[w])
            hist = B.imbalance_history(f, cfg, uni, at - pd.Timedelta(seconds=fetch_back[w]), at + pd.Timedelta(seconds=1))
            syms |= set(E.activity_candidates(hist, uni, cfg, w, at))
            last_at = at if last_at is None else max(last_at, at)
        if syms:
            prior = [x for x in all_dates if x < d]
            out[d] = (sorted(syms), last_at, prior)
            reqs += B.activity_requests(cfg, sorted(syms), d, last_at, prior)
    if not reqs:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        cost = sum(pool.map(lambda r: B.quote_lenient(f, r), reqs))
    n = sum(len(v[0]) for v in out.values())
    f.approve(cost, f"activity data (volume / block trades / options) for {n:,} candidate stock-days")
    prefetch(f, reqs, workers, get=lambda r: B.get_lenient(f, r))
    return out


def backtest(f: C.Fetcher, cfg: dict, start: dt.date, end: dt.date, windows: str = "AB",
             close_hhmm: str = "16:00", fit_end: dt.date | None = None, workers: int = 8):
    sch, sig = cfg["schedule"], cfg["signal"]
    wait = cfg["execution"]["fill_wait_s"]
    # 1. listing venue each month (keeps later-delisted stocks: no survivorship bias)
    listings = B.fetch_listings(f, cfg, B.definition_dates(start - dt.timedelta(days=40), end),
                                "listing venues (monthly definitions)")
    syms = B.enabled_symbols(listings, cfg)
    # 2. daily bars -> ADV and sigma from prior sessions only; next open for overnight exits
    bars_end = min(end + dt.timedelta(days=7), B.latest_available(f.client, "EQUS.SUMMARY", "ohlcv-1d"))
    bars = B.fetch_bars(f, syms, start - dt.timedelta(days=B.BAR_LOOKBACK_CAL_DAYS), bars_end,
                        f"daily bars for {len(syms):,} symbols")
    stats = E.universe_stats(bars[bars["date"] <= end], cfg)
    opens = bars.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    next_open_tbl = opens.shift(-1)
    sessions = [d for d in stats["prev_close"].index if start <= d <= end]
    offsets = {"A": sch["window_a_s"], "B": sch["window_b_s"]}
    fetch_back = {w: max(sig["lookback_a_s"] if w == "A" else sig["lookback_b_s"], sig["max_age_s"]) for w in "AB"}
    if "min_buy_print_share" in sig:   # "repeated" is judged over every print since Nasdaq starts publishing
        fetch_back = {w: max(b, E.CLOSING_IMBALANCE_FROM_S - offsets[w]) for w, b in fetch_back.items()}

    # 3. plan every request, quote a sample of sessions, extrapolate, ask once
    plan = []
    for d in sessions:
        uni = E.universe_for(stats, d, B.listing_for(listings, d), cfg)
        if uni.empty:
            continue
        close_at = C.close_time(d, close_hhmm)
        reqs, known = [], []       # known: exactly what the day-by-day loop will ask for (safe to prefetch)
        for w in windows:
            at = close_at - pd.Timedelta(seconds=offsets[w])
            known += B.imbalance_requests(cfg, uni, at - pd.Timedelta(seconds=fetch_back[w]), at + pd.Timedelta(seconds=1))
            top = uni.head(cfg["account"]["max_positions"])        # allowance for the fill check
            reqs += [C.req_bbo(cfg["datasets"][v], list(g.index), at, at + pd.Timedelta(seconds=wait))
                     for v, g in top.groupby("venue")]
        by_venue = {v: sorted(g.index) for v, g in uni.groupby("venue")}
        known += B.close_requests(cfg, by_venue, close_at)
        plan.append((d, uni, close_at, by_venue, reqs + known, known))
    if not plan:
        raise SystemExit("No sessions with a universe in that range.")
    sample = plan[:: max(1, len(plan) // 5)][:5]
    per_day = float(np.mean([sum(f.quote(r) for r in p[4]) for p in sample]))
    f.approve(per_day * len(plan) * 1.2, f"imbalance, quotes and closing prices for {len(plan)} sessions "
                                         f"(estimate from {len(sample)} sampled sessions, +20%)")
    # imbalance + closing prices in parallel; the fill-check quotes depend on the ideas, so they stay in the loop
    prefetch(f, [r for p in plan for r in p[5]], workers)

    # 3b. optional activity filters: buy their data only for stocks that pass every other rule
    activity = plan_activity(f, cfg, plan, windows, offsets, fetch_back, sorted(stats["prev_close"].index), workers) \
        if E.needs_activity(cfg) else {}

    # 4. run the engine day by day with one ledger per session
    trades, calib = [], []
    for i, (d, uni, close_at, by_venue, _, _) in enumerate(plan, 1):
        closes = B.official_closes(f, cfg, by_venue, close_at)
        nxt = next_open_tbl.loc[d] if d in next_open_tbl.index else pd.Series(dtype=float)
        issued: dict[str, float] = {}
        for w in windows:
            at = close_at - pd.Timedelta(seconds=offsets[w])
            hist = B.imbalance_history(f, cfg, uni, at - pd.Timedelta(seconds=fetch_back[w]), at + pd.Timedelta(seconds=1))
            if hist.empty:
                continue  # holiday / half-day / no data
            u = uni
            if d in activity:
                syms, until, prior = activity[d]
                u = uni.join(B.fetch_activity(f, cfg, syms, d, at, until, prior))
            ideas = E.run_window(hist, u, cfg, w, at, close_at, issued)
            if ideas.empty:
                continue
            new = ideas[ideas["action"] == "TRADE"]
            fills = B.first_fill(f, cfg, new, at, wait) if not new.empty else pd.Series(dtype=float)
            sim = simulate(ideas, fills, closes, nxt, cfg, d, w)
            E.record_issued(ideas[ideas.index.isin(fills.dropna().index)], issued)   # only filled buys
            trades.append(sim)
            calib.append(calib_rows(ideas, closes, cfg, d, w))
        if i % 20 == 0 or i == len(plan):
            print(f"  {i}/{len(plan)} sessions done")

    trades = pd.concat([t for t in trades if not t.empty]) if any(not t.empty for t in trades) else \
        pd.DataFrame(columns=["session", "window", "missed", "exit", "pnl", "ret_bps", "overnight"])
    calib = pd.concat([c for c in calib if not c.empty]) if any(not c.empty for c in calib) else pd.DataFrame()
    run_sessions = [p[0] for p in plan]
    fit_end = fit_end or run_sessions[(len(run_sessions) - 1) // 2]
    out = C.ROOT / "results"
    out.mkdir(exist_ok=True)
    trades.to_csv(out / "backtest_trades.csv")
    if not calib.empty:
        calib.to_csv(out / "backtest_calibration.csv")
    report(trades, calib, cfg, run_sessions, fit_end)
    return trades, calib


if __name__ == "__main__":
    main()
