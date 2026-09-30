"""Step 2 (start by 20:45 UK): live trade ideas from the closing imbalance feed.

    python live_signals.py                         # live, needs Databento live entitlement
    python live_signals.py --replay 2026-09-29     # re-run a past day from historical data (cheap)
    python live_signals.py --close 13:00           # half-day sessions

Prints ranked BUY ideas at four moments (times shown in ET and UK):
  Window A  close-7:00  and last call close-5:30  -> exit SELL MOC before close-5:00
  Window B  close-4:00  and last call close-2:30  -> exit SELL LOC before close-2:00
and saves every printed row to results/ideas_<date>.csv for review.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import threading
import time

import databento as db
import pandas as pd

from imbalance import build as B
from imbalance import common as C
from imbalance import engine as E


class Book:
    """Thread-safe store of every closing-imbalance record received."""

    def __init__(self):
        self._rows, self._lock = [], threading.Lock()

    def add(self, row: tuple) -> None:
        with self._lock:
            self._rows.append(row)

    def frame(self) -> pd.DataFrame:
        with self._lock:
            df = pd.DataFrame(self._rows, columns=E.IMB_COLUMNS)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.sort_values("ts").reset_index(drop=True)


def make_callback(client, venue: str, closing_types: list[str], allowed: set[str], book: Book):
    """Record handler for one Live client (one dataset)."""

    def qty(x: int) -> float:
        return math.nan if x == E.UNDEF_QTY else float(x)

    def cb(rec) -> None:
        if isinstance(rec, db.ImbalanceMsg):
            if rec.auction_type not in closing_types:
                return
            sym = client.symbology_map.get(rec.instrument_id)
            if sym is None or str(sym) not in allowed:
                return
            ref = rec.pretty_ref_price
            if not (ref > 0) or rec.total_imbalance_qty == E.UNDEF_QTY:   # same checks as clean_imbalance
                return
            side = getattr(rec.side, "value", str(rec.side))
            book.add((rec.ts_recv, str(sym), venue, side, ref,
                      rec.pretty_cont_book_clr_price, rec.pretty_auct_interest_clr_price,
                      qty(rec.paired_qty), qty(rec.total_imbalance_qty)))
        elif isinstance(rec, db.ErrorMsg):
            print(f"[Databento error] {rec.err}")

    return cb


def schedule(close_at: pd.Timestamp, cfg: dict) -> list[tuple[pd.Timestamp, str, str]]:
    s = cfg["schedule"]
    at = lambda sec: close_at - pd.Timedelta(seconds=sec)
    return [
        (at(s["window_a_s"]), "A", "signal"),
        (at(s["window_a_last_call_s"]), "A", "last call"),
        (at(s["window_b_s"]), "B", "signal"),
        (at(s["window_b_last_call_s"]), "B", "last call"),
    ]


def run_events(book_frame, universe, cfg, close_at, events, log, sleep_until=None) -> None:
    """Print ideas at each scheduled moment, with one ledger for the whole evening."""
    issued: dict[str, float] = {}
    for at, window, label in events:
        if sleep_until is not None:
            if not sleep_until(at):
                print(f"(skipped Window {window} {label} at {C.fmt_times(at)} — already past)")
                continue
        try:
            hist = book_frame()
            if hist.empty:
                print(f"\n!! {C.fmt_times(at)}: NO imbalance data received. Check your live entitlement "
                      f"(Nasdaq TotalView-ITCH live) and the Databento status page. Do not trade blind.")
                continue
            ideas = E.run_window(hist, universe, cfg, window, at, close_at, issued)
            C.print_ideas(ideas, window, at, close_at, cfg, n_records=int((hist["ts"] <= at).sum()))
            E.record_issued(ideas, issued)
            if ideas is not None and not ideas.empty:
                log.append(ideas.assign(decision_time=at, label=label).reset_index())
        except Exception as exc:  # one bad print must not kill the evening
            print(f"\n!! Window {window} {label} failed: {exc!r}. Continuing with the next print.")


def save_log(log: list[pd.DataFrame], session: dt.date) -> None:
    if not log:
        print("\nNothing to save.")
        return
    out = C.ROOT / "results" / f"ideas_{session}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.concat(log, ignore_index=True).to_csv(out, index=False)
    print(f"\nSaved {out}  (run review.py tomorrow to score it)")


def load_universe(path: str, session: dt.date) -> pd.DataFrame:
    # keep_default_na=False: a ticker literally called "NA" must stay a ticker
    uni = pd.read_csv(path, index_col="symbol", keep_default_na=False, na_values=[""], dtype={"symbol": str})
    built_for = str(uni["session"].iloc[0]) if "session" in uni and len(uni) else "?"
    if built_for != session.isoformat():
        print(f"WARNING: {path} was built for {built_for}, not {session}. Run prep_universe.py first.")
    return uni


def live(a, cfg: dict, universe: pd.DataFrame, session: dt.date, close_at: pd.Timestamp,
         now=lambda: pd.Timestamp.now(tz="UTC"), sleep=time.sleep, live_factory=None) -> None:
    key = C.api_key()
    live_factory = live_factory or (lambda: db.Live(key=key))
    book, log, clients = Book(), [], []
    start_at = close_at - pd.Timedelta(seconds=cfg["schedule"]["subscribe_s"])

    def sleep_until(t: pd.Timestamp) -> bool:
        if now() > t + pd.Timedelta(seconds=5):
            return False
        while (left := (t - now()).total_seconds()) > 0:
            sleep(min(left, 30))
        return True

    print(f"Session {session}. Close {C.fmt_times(close_at)}. Feed starts {C.fmt_times(start_at)}.")
    if now() < start_at:
        print("Waiting for the feed start… (leave this window open)")
        sleep_until(start_at)

    for venue, grp in universe.groupby("venue"):
        ds = cfg["datasets"][venue]
        client = live_factory()
        for ch in C.chunks(grp.index, 500):
            client.subscribe(dataset=ds, schema="imbalance", symbols=list(ch), start=start_at.isoformat())
        client.add_callback(make_callback(client, venue, cfg["closing_auction_types"][ds], set(grp.index), book))
        client.start()
        clients.append(client)
        print(f"Subscribed {len(grp):,} {venue} symbols on {ds} (replaying from {C.fmt_times(start_at)}).")

    try:
        run_events(book.frame, universe, cfg, close_at, schedule(close_at, cfg), log, sleep_until)
        sleep_until(close_at + pd.Timedelta(seconds=20))
    finally:
        for c in clients:
            c.stop()
        for c in clients:
            try:
                c.block_for_close(timeout=10)
            except Exception as exc:  # closing errors are harmless here
                print(f"(close: {exc})")
        save_log(log, session)


def replay(a, cfg: dict, universe: pd.DataFrame, session: dt.date, close_at: pd.Timestamp) -> None:
    client = db.Historical(C.api_key())
    f = C.Fetcher(client, C.ROOT / "cache", a.max_cost, a.yes)
    start = close_at - pd.Timedelta(seconds=cfg["schedule"]["subscribe_s"])
    reqs = B.imbalance_requests(cfg, universe, start, close_at)
    f.approve(sum(f.quote(r) for r in reqs), f"imbalance replay for {session}")
    hist = B.imbalance_history(f, cfg, universe, start, close_at)
    print(f"Replaying {len(hist):,} imbalance records for {session}.")
    log: list[pd.DataFrame] = []
    run_events(lambda: hist, universe, cfg, close_at, schedule(close_at, cfg), log)
    save_log(log, session)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="universe.csv")
    ap.add_argument("--close", default="16:00", help="closing auction time ET, e.g. 13:00 on half-days")
    ap.add_argument("--replay", help="YYYY-MM-DD: run a past session from historical data")
    ap.add_argument("--config")
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    cfg = C.load_config(a.config)
    session = dt.date.fromisoformat(a.replay) if a.replay else dt.datetime.now(C.ET).date()
    close_at = C.close_time(session, a.close)
    universe = load_universe(a.universe, session)
    if a.replay:
        replay(a, cfg, universe, session, close_at)
    else:
        live(a, cfg, universe, session, close_at)


if __name__ == "__main__":
    main()
