"""Config, clock, Databento access (with cache + cost guard) and printing."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import time
import tomllib
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")
UK = ZoneInfo("Europe/London")
ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- config & clock
def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def close_time(date: dt.date, hhmm: str = "16:00") -> pd.Timestamp:
    """The US closing-auction time for `date`, as a UTC timestamp (DST-correct)."""
    h, m = map(int, hhmm.split(":"))
    return pd.Timestamp(dt.datetime(date.year, date.month, date.day, h, m, tzinfo=ET)).tz_convert("UTC")


def fmt_times(ts: pd.Timestamp) -> str:
    """'15:55:00 ET / 20:55:00 UK' — both clocks, because UK/US DST dates differ."""
    return f"{ts.tz_convert(ET):%H:%M:%S} ET / {ts.tz_convert(UK):%H:%M:%S} UK"


# --------------------------------------------------------------------------- Databento access
def api_key() -> str:
    """DATABENTO_API_KEY from the environment, else from imbalance_playbook/.env (never committed)."""
    load_dotenv(ROOT / ".env", override=False)  # a key already in the environment wins
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        sys.exit(f"DATABENTO_API_KEY is not set. Copy .env.example to {ROOT / '.env'} and put your key after "
                 "DATABENTO_API_KEY=  (or export DATABENTO_API_KEY=db-... in your shell).")
    return key


class Fetcher:
    """Historical requests with an on-disk cache (never pay twice) and a spend cap."""

    def __init__(self, client, cache_dir: pathlib.Path, max_usd: float, assume_yes: bool):
        self.client, self.cache_dir = client, pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_usd, self.assume_yes = max_usd, assume_yes
        self.spent_estimate = 0.0

    def _path(self, req: dict) -> pathlib.Path:
        key = hashlib.sha1(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return self.cache_dir / f"{req['dataset']}_{req['schema']}_{key}.pkl"

    def cached(self, req: dict) -> bool:
        return self._path(req).exists()

    def quote(self, req: dict) -> float:
        """Exact $ cost of a request (free call). Cached requests cost 0."""
        if self.cached(req):
            return 0.0
        q = {k: v for k, v in req.items() if k in ("dataset", "schema", "symbols", "start", "end", "stype_in")}
        return float(self._retry(lambda: self.client.metadata.get_cost(**q)))

    def approve(self, estimate_usd: float, what: str) -> None:
        print(f"\nEstimated Databento cost for {what}: ${estimate_usd:,.2f} "
              f"(cap ${self.max_usd:,.2f}; your $125 sign-up credit is used first)")
        if estimate_usd > self.max_usd:
            sys.exit("Above --max-cost. Shorten the date range or raise the cap. Nothing was bought.")
        if estimate_usd > 0 and not self.assume_yes:
            if input("Proceed? [y/N] ").strip().lower() != "y":
                sys.exit("Cancelled. Nothing was bought.")
        self.spent_estimate += estimate_usd

    RETRY_WAITS_S = (2, 8, 30)   # a busy Databento gateway sometimes answers 502/503/504

    def _retry(self, call):
        for wait in (*self.RETRY_WAITS_S, None):
            try:
                return call()
            except Exception as e:
                if wait is None or not 500 <= (getattr(e, "http_status", None) or 0) < 600:
                    raise
                print(f"  Databento {e.http_status}; retrying in {wait}s", flush=True)
                time.sleep(wait)

    def get(self, **req) -> pd.DataFrame:
        path = self._path(req)
        if path.exists():
            return pd.read_pickle(path)
        store = self._retry(lambda: self.client.timeseries.get_range(**req))
        df = store.to_df()
        df = df.reset_index() if df.index.name else df
        # An empty answer is cached only when the period is long over (a weekend, a holiday).
        # A recent empty answer may just mean "not published yet", so it is asked again next time.
        settled = pd.Timestamp(req["end"]) < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
        if not df.empty or settled:
            df.to_pickle(path)
        return df


def chunks(seq, n: int):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------- request builders
def req_definitions(dataset: str, date: dt.date) -> dict:
    start = pd.Timestamp(date, tz="UTC")
    return dict(dataset=dataset, schema="definition", symbols="ALL_SYMBOLS",
                start=start.isoformat(), end=(start + pd.Timedelta(days=1)).isoformat())


def req_daily_bars(symbols: list[str], start: dt.date, end: dt.date) -> dict:
    # Explicit symbol lists (not ALL_SYMBOLS) so the response carries ticker mappings.
    return dict(dataset="EQUS.SUMMARY", schema="ohlcv-1d", symbols=sorted(symbols),
                start=pd.Timestamp(start, tz="UTC").isoformat(),
                end=(pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).isoformat())


def req_imbalance(dataset: str, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return dict(dataset=dataset, schema="imbalance", symbols=sorted(symbols),
                start=start.isoformat(), end=end.isoformat())


def req_bbo(dataset: str, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return dict(dataset=dataset, schema="bbo-1s", symbols=sorted(symbols),
                start=start.isoformat(), end=end.isoformat())


def req_closes(dataset: str, symbols: list[str], close_at: pd.Timestamp) -> dict:
    return dict(dataset=dataset, schema="statistics", symbols=sorted(symbols),
                start=(close_at - pd.Timedelta(minutes=1)).isoformat(),
                end=(close_at + pd.Timedelta(minutes=5)).isoformat())


# --------------------------------------------------------------------------- parsers
def parse_listing(defs: pd.DataFrame) -> pd.Series:
    """symbol -> listing venue MIC (XNAS.ITCH definitions carry the listing venue in `exchange`)."""
    if defs.empty:
        return pd.Series(dtype=str)
    d = defs.sort_values("ts_recv") if "ts_recv" in defs else defs
    return d.groupby("raw_symbol")["exchange"].last().astype(str)


def parse_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """EQUS.SUMMARY ohlcv-1d -> date, symbol, open, close, volume (consolidated, official close)."""
    if raw.empty:
        return pd.DataFrame(columns=["date", "symbol", "open", "close", "volume"])
    df = raw.copy()
    df["date"] = pd.to_datetime(df["ts_event"], utc=True).dt.date
    return df[["date", "symbol", "open", "close", "volume"]].dropna(subset=["symbol"])


def parse_closes(raw: pd.DataFrame) -> pd.Series:
    """statistics -> symbol -> official closing-cross price (stat_type CLOSE_PRICE = 11)."""
    if raw.empty:
        return pd.Series(dtype=float)
    df = raw[(raw["stat_type"] == 11) & raw["price"].notna() & (raw["quantity"] > 0)]
    return df.groupby("symbol")["price"].last()


# --------------------------------------------------------------------------- printing
def _money(x: float) -> str:
    return f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:.0f}k"


def print_ideas(ideas: pd.DataFrame, window: str, at: pd.Timestamp, close_at: pd.Timestamp, cfg: dict,
                n_records: int | None = None) -> None:
    sch = cfg["schedule"]
    title = {"A": "WINDOW A — early imbalance, impact model",
             "B": "WINDOW B — near clearing price model"}[window]
    cut = close_at - pd.Timedelta(seconds=sch["moc_cutoff_s"] if window == "A" else sch["loc_cutoff_s"])
    order = "SELL MOC" if window == "A" else "SELL LOC at the price shown"
    print("\n" + "=" * 118)
    fed = f"   ({n_records:,} imbalance records so far)" if n_records is not None else ""
    print(f"{title}   decision time {fmt_times(at)}{fed}")
    print(f"1) Send the BUY limit. 2) ONLY for shares that filled, send {order} before {fmt_times(cut)}.")
    print(f"   Not filled ~15 s before that cut-off? Cancel the buy and drop the idea. On-close orders can't be cancelled once sent.")
    if window == "B":
        print("   LOC risk: Nasdaq re-prices a lower sell LOC up to the 15:50/15:55 reference price. If the close")
        print("   prints below it, the sell does not fill and you hold overnight.")
    print("=" * 118)
    if ideas is None or ideas.empty:
        print("No buy imbalances in your universe yet.")
        return
    hdr = f"{'#':>2} {'TICKER':<7}{'VEN':<5}{'REF':>7}{'IMB SH':>11}{'IMB $':>8}{'/ADV':>7}{'/PAIR':>7}" \
          f"{'NEAR':>8}{'EXP':>7}{'COST':>7}{'EDGE':>7}  ACTION"
    print(hdr)
    print("-" * 118)
    loc_px = lambda r: r["ref"] * (1 - cfg["execution"]["loc_buffer"])
    for i, (sym, r) in enumerate(ideas.iterrows(), 1):
        near = f"{r['near']:.2f}" if pd.notna(r["near"]) else "  -"
        base = (f"{i:>2} {sym:<7}{r['venue']:<5}{r['ref']:>7.2f}{r['imb']:>11,.0f}{_money(r['notional']):>8}"
                f"{r['imb_adv']*100:>6.1f}%{r['imb_paired']:>7.2f}{near:>8}"
                f"{r['exp_move']*100:>6.2f}%{r['cost']*100:>6.2f}%{r['edge']*100:>6.2f}%  ")
        if r["action"] == "TRADE":
            exit_ = "SELL MOC" if window == "A" else f"SELL LOC {loc_px(r):.2f}"
            act = f">> BUY {int(r['shares']):,} @ <= {r['limit']:.2f}, filled qty -> {exit_}"
        else:
            act = f"{r['action'] or 'skip'}: {r['reason']}"
        print(base + act)
    n = int((ideas["action"] == "TRADE").sum())
    print("-" * 118)
    print(f"{n} new TRADE idea(s). Everything else is watch-only. Rules-based output, not advice — "
          f"size down until paper results agree with the backtest.")
