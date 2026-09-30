"""Signal maths for the closing-auction BUY-imbalance playbook.

Pure functions only: no network and no Databento client, so every number the
tool prints can be traced and tested. All returns and edges are fractions
(0.01 = 1%); *_bps values are basis points.

Pipeline for one decision time T:
    clean_imbalance -> snapshot -> features -> apply_filters -> pick_trades
"""
from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

UNDEF_QTY = 4294967295          # DBN "not provided" for u32 quantities
SECONDS_PER_SESSION = 23400     # 6.5-hour regular session
# Exit rules (MOC before 15:55, LOC before 15:58) are only defined for Nasdaq-listed
# stocks. NYSE's MOC/LOC cut-off is 15:50 — the same minute its imbalance feed
# starts — so NYSE / NYSE American names are shown as WATCH, never TRADE.
TRADABLE_VENUES = {"XNAS"}

ET = ZoneInfo("America/New_York")
CLOSING_IMBALANCE_FROM_S = 600  # Nasdaq publishes closing imbalances from 15:50 ET (600 s before the close)
IMB_COLUMNS = ["ts", "symbol", "venue", "side", "ref", "near", "far", "paired", "imb"]


# --------------------------------------------------------------------------- data shaping
def clean_imbalance(raw: pd.DataFrame, venue: str, closing_types: list[str]) -> pd.DataFrame:
    """Turn a Databento imbalance DataFrame into the tool's standard columns.

    Keeps closing-auction records only. `ref`/`near`/`far` are prices in dollars,
    `paired`/`imb` are shares; missing values become NaN.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=IMB_COLUMNS)
    df = raw.reset_index() if "ts_recv" not in raw.columns else raw.copy()
    df = df[df["auction_type"].isin(closing_types)]
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["ts_recv"], utc=True),
        "symbol": df["symbol"].astype(str),
        "venue": venue,
        "side": df["side"].astype(str),
        "ref": df["ref_price"].astype(float),
        "near": df["cont_book_clr_price"].astype(float),
        "far": df["auct_interest_clr_price"].astype(float),
        "paired": df["paired_qty"].astype(float).replace(UNDEF_QTY, np.nan),
        "imb": df["total_imbalance_qty"].astype(float).replace(UNDEF_QTY, np.nan),
    })
    out = out[(out["ref"] > 0) & out["imb"].notna()]
    return out.sort_values("ts").reset_index(drop=True)


def snapshot(hist: pd.DataFrame, at: pd.Timestamp, lookback_s: float, max_age_s: float) -> pd.DataFrame:
    """Latest record per symbol at time `at`, plus stability measures.

    Uses only records with ts <= at (no look-ahead). Adds:
      side_stable  - side unchanged over the look-back window
      shrink       - 1 - imb_now / peak imb over the look-back (same side)
      age_s        - seconds since the latest record
      buy_share    - share of today's closing-imbalance prints so far that were on the buy side
    """
    h = hist[hist["ts"] <= at]
    if h.empty:
        return pd.DataFrame(columns=IMB_COLUMNS + ["side_stable", "shrink", "age_s", "buy_share"]).set_index("symbol")
    last = h.groupby("symbol").tail(1).set_index("symbol")
    recent = h[h["ts"] > at - pd.Timedelta(seconds=lookback_s)]
    sides = recent.groupby("symbol")["side"].nunique()
    last_side = last["side"].rename("side_last").reset_index()
    same_side = recent.merge(last_side, on="symbol", how="inner")
    same_side = same_side[same_side["side"] == same_side["side_last"]]
    peak = same_side.groupby("symbol")["imb"].max()
    last["side_stable"] = sides.reindex(last.index).fillna(1).le(1)
    last["peak_imb"] = peak.reindex(last.index).fillna(last["imb"])
    last["shrink"] = (1 - last["imb"] / last["peak_imb"]).clip(lower=0).fillna(0)
    last["age_s"] = (at - last["ts"]).dt.total_seconds()
    last["buy_share"] = h.assign(buy=h["side"] == "B").groupby("symbol")["buy"].mean().reindex(last.index)
    return last


# --------------------------------------------------------------------------- the equations
def spread_bps(price: pd.Series, costs: dict) -> pd.Series:
    """Assumed full spread (bps) from the price bucket table in config."""
    return pd.Series(
        np.select(
            [price < 2, price < 5],
            [costs["spread_bps_1_2"], costs["spread_bps_2_5"]],
            default=costs["spread_bps_5_10"],
        ),
        index=price.index,
        dtype=float,
    )


def features(snap: pd.DataFrame, universe: pd.DataFrame, cfg: dict, window: str) -> pd.DataFrame:
    """Compute every quantity in the playbook for each symbol.

    notional   = imb * ref                              ($ imbalance)
    imb_adv    = notional / adv_usd                     (size vs normal trading)
    imb_paired = imb / paired                           (how one-sided the auction is)
    near_gap   = near / ref - 1                         (exchange's own clearing-price move)
    day_move   = ref / prev_close - 1                   (how far the stock is up today)
    move_sigma = day_move / sigma                       (that move measured in normal daily moves)
    impact     = k * sigma * sqrt(imb_adv)              (square-root impact law)
    exp_move   = impact              in Window A        (no near price published yet)
               = beta * near_gap     in Window B        (falls back to impact if no near price)
    cost       = spread/2 + slippage + 2 * commission/price
    edge       = exp_move - cost
    """
    sig, costs = cfg["signal"], cfg["costs"]
    df = snap.join(universe, how="inner", rsuffix="_u")
    if df.empty:
        return df
    df["notional"] = df["imb"] * df["ref"]
    df["imb_adv"] = df["notional"] / df["adv_usd"]
    df["imb_paired"] = df["imb"] / df["paired"].where(df["paired"] > 0)
    df["near_gap"] = df["near"] / df["ref"] - 1
    df["day_move"] = df["ref"] / df["prev_close"] - 1               # move since yesterday's close
    df["move_sigma"] = df["day_move"] / df["sigma"]                  # that move in "normal days"
    df["impact"] = sig["k_impact"] * df["sigma"] * np.sqrt(df["imb_adv"].clip(lower=0))
    if window == "B":
        df["exp_move"] = (sig["beta_near"] * df["near_gap"]).where(df["near"].notna(), df["impact"])
    else:
        df["exp_move"] = df["impact"]
    df["spread_bps"] = spread_bps(df["ref"], costs)
    df["cost"] = (df["spread_bps"] / 2 + costs["slippage_bps"]) / 1e4 + 2 * costs["commission_per_share"] / df["ref"]
    df["edge"] = df["exp_move"] - df["cost"]
    return df


ACTIVITY_KEYS = ("min_rel_volume", "min_block_shares", "min_block_count", "min_call_put_ratio")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].astype(float) if name in df else pd.Series(np.nan, index=df.index)


def apply_filters(df: pd.DataFrame, cfg: dict, window: str) -> pd.DataFrame:
    """Add `reason` (first rule that fails, '' if none) and `ok`."""
    if df.empty:
        return df.assign(reason=pd.Series(dtype=str), ok=pd.Series(dtype=bool))
    sig, uni = cfg["signal"], cfg["universe"]
    bad = (~np.isfinite(df["edge"].astype(float)) | ~(df["ref"] > 0) | df["imb"].isna()
           | df["sigma"].isna() | df["adv_usd"].isna())
    rules = [
        ("bad data", bad),
        ("sell side", df["side"] != sig["side"]),
        ("price band", (df["ref"] < uni["min_price"]) | (df["ref"] > uni["max_price"])),
        ("illiquid", df["adv_usd"] < uni["min_adv_usd"]),
        ("stale", df["age_s"] > sig["max_age_s"]),
        ("small $", df["notional"] < sig["min_notional_usd"]),
        ("small vs ADV", df["imb_adv"] < sig["min_imb_to_adv"]),
        ("weak vs paired", df["imb_paired"].fillna(np.inf) < sig["min_imb_to_paired"]),
        ("flipped", ~df["side_stable"].astype(bool)),
        ("shrinking", df["shrink"] > sig["max_shrink"]),
        ("not up enough today", df["day_move"] < sig.get("min_day_move", -np.inf)),
        ("move small vs sigma", df["move_sigma"] < sig.get("min_move_to_sigma", -np.inf)),
        ("imbalance not repeated", _col(df, "buy_share") < sig.get("min_buy_print_share", -np.inf)),
    ]
    if window == "B":
        rules += [
            ("no near price", df["near"].isna()),
            ("gap too wide", df["near_gap"] > sig["max_near_gap"]),
        ]
    # activity filters: need data fetched for the day (see build.fetch_activity); missing data fails,
    # except the call/put ratio, which only applies to stocks that traded options that day
    if "min_rel_volume" in sig:
        rules += [("low rel volume", _col(df, "rel_volume") < sig["min_rel_volume"])]
    if "min_block_shares" in sig:
        rules += [("no block trade", _col(df, "max_block") < sig["min_block_shares"])]
    if "min_block_count" in sig:
        rules += [("blocks not repeated", _col(df, "buy_blocks") < sig["min_block_count"])]
    if "min_call_put_ratio" in sig:
        rules += [("call/put low", _col(df, "cp_ratio").fillna(np.inf) < sig["min_call_put_ratio"])]
    rules += [("edge < min", df["edge"] * 1e4 < sig["min_edge_bps"])]
    reason = pd.Series("", index=df.index)
    for name, mask in rules:
        reason = reason.mask((reason == "") & mask.fillna(True), name)
    out = df.copy()
    out["reason"] = reason
    out["ok"] = reason == ""
    return out


def size_shares(row: pd.Series, cfg: dict, secs_to_close: float, price: float) -> int:
    """Shares for one idea = min(risk budget, participation cap, notional cap).

    price                     = the entry limit (what you may actually pay)
    adverse move to the close = z * sigma * sqrt(seconds_left / 23,400)
    risk budget shares        = equity * risk% / (price * adverse move)
    """
    acct, sz = cfg["account"], cfg["sizing"]
    adverse = sz["z_adverse"] * row["sigma"] * math.sqrt(max(secs_to_close, 1) / SECONDS_PER_SESSION)
    if not np.isfinite(adverse) or adverse <= 0:
        return 0
    risk_usd = acct["equity_usd"] * acct["risk_per_trade_pct"] / 100
    by_risk = risk_usd / (price * adverse)
    by_part = sz["max_participation"] * row["imb"]
    by_notional = acct["max_notional_per_trade_usd"] / price
    return int(max(0, math.floor(min(by_risk, by_part, by_notional))))


def entry_limit(row: pd.Series, cfg: dict) -> float:
    """Highest price to pay: reference + half spread + slippage, rounded up to the cent."""
    c = cfg["costs"]
    px = row["ref"] * (1 + (row["spread_bps"] / 2 + c["slippage_bps"]) / 1e4)
    return math.ceil(px * 100 - 1e-9) / 100


def rank_score(df: pd.DataFrame, rank_by: str) -> pd.Series:
    """Order in which ideas get the tonight's slots: `edge`, or `move_x_rvol` = day move x relative volume."""
    if rank_by == "edge":
        return df["edge"].astype(float)
    if rank_by == "move_x_rvol":
        return df["day_move"].astype(float) * _col(df, "rel_volume")
    raise ValueError(f"rank_by must be 'edge' or 'move_x_rvol', not {rank_by!r}")


def pick_trades(df: pd.DataFrame, cfg: dict, window: str, secs_to_close: float,
                issued: dict[str, float] | None = None) -> pd.DataFrame:
    """Watch list (top_n buy imbalances by $) with TRADE rows sized and capped.

    Ideas that pass every rule get the slots in `rank_by` order (default: edge). With
    max_positions = 1 that is the single best idea of the evening.

    `issued` is the session ledger {symbol: $ notional} of ideas already given
    earlier tonight. Caps (max_positions, max_gross_usd) count those too, and a
    symbol already issued is shown as 'held', never bought twice.
    """
    sig, acct, c = cfg["signal"], cfg["account"], cfg["costs"]
    issued = issued if issued is not None else {}
    if df.empty:
        return df
    watch = df[df["side"] == sig["side"]].sort_values("notional", ascending=False).head(sig["top_n"]).copy()
    watch["action"] = np.where(watch["ok"], "", "skip")
    watch["shares"] = 0
    watch["limit"] = np.nan
    held = watch.index.isin(list(issued))
    watch.loc[held, ["action", "reason"]] = ["held", "bought earlier: keep its exit"]
    gross, n = float(sum(issued.values())), len(issued)
    watch["score"] = rank_score(watch, sig.get("rank_by", "edge"))
    for sym in watch[watch["ok"] & ~held].sort_values("score", ascending=False, na_position="last").index:
        row = watch.loc[sym]
        if row["venue"] not in TRADABLE_VENUES:
            watch.loc[sym, ["action", "reason"]] = ["watch", "venue exit rules"]
            continue
        if n >= acct["max_positions"]:
            watch.loc[sym, ["action", "reason"]] = ["skip", "max positions tonight"]
            continue
        limit = entry_limit(row, cfg)
        shares = size_shares(row, cfg, secs_to_close, limit)
        shares = min(shares, int(max(0.0, acct["max_gross_usd"] - gross) // limit))
        if shares < 1:
            watch.loc[sym, ["action", "reason"]] = ["skip", "size < 1 / gross cap"]
            continue
        # edge again with the real commission for this size (minimum ticket included)
        comm = 2 * max(c["min_commission"], c["commission_per_share"] * shares) / (shares * limit)
        real_edge = row["exp_move"] - (row["spread_bps"] / 2 + c["slippage_bps"]) / 1e4 - comm
        if real_edge * 1e4 < sig["min_edge_bps"]:
            watch.loc[sym, ["action", "reason"]] = ["skip", "commission on small size"]
            continue
        watch.loc[sym, ["action", "shares", "limit"]] = ["TRADE", shares, limit]
        gross += shares * limit
        n += 1
    watch["window"] = window
    return watch


def run_window(hist: pd.DataFrame, universe: pd.DataFrame, cfg: dict, window: str,
               at: pd.Timestamp, close_at: pd.Timestamp, issued: dict[str, float] | None = None) -> pd.DataFrame:
    """Everything for one decision time: snapshot -> features -> filters -> trades."""
    sig = cfg["signal"]
    lookback = sig["lookback_a_s"] if window == "A" else sig["lookback_b_s"]
    snap = snapshot(hist, at, lookback, sig["max_age_s"])
    feat = features(snap, universe, cfg, window)
    filt = apply_filters(feat, cfg, window)
    return pick_trades(filt, cfg, window, (close_at - at).total_seconds(), issued)


def activity_candidates(hist: pd.DataFrame, universe: pd.DataFrame, cfg: dict, window: str,
                        at: pd.Timestamp) -> list[str]:
    """Symbols that pass every rule except the activity filters: the only ones worth buying activity data for."""
    sig = cfg["signal"]
    base = {**cfg, "signal": {k: v for k, v in sig.items() if k not in ACTIVITY_KEYS}}
    lookback = sig["lookback_a_s"] if window == "A" else sig["lookback_b_s"]
    filt = apply_filters(features(snapshot(hist, at, lookback, sig["max_age_s"]), universe, base, window), base, window)
    if filt.empty:
        return []
    return sorted(filt.index[filt["ok"] & (filt["side"] == sig["side"])])


def needs_activity(cfg: dict) -> bool:
    sig = cfg["signal"]
    return any(k in sig for k in ACTIVITY_KEYS) or sig.get("rank_by") == "move_x_rvol"


# --------------------------------------------------------------------------- activity measures
def rel_volume(bars: pd.DataFrame, session, at: pd.Timestamp, n_days: int, min_days: int) -> pd.Series:
    """symbol -> volume from the open to `at`'s time of day, today / average of the prior n_days sessions.

    bars: 1-minute bars with ts_event, symbol, volume (any venues; only the ratio is used).
    """
    if bars.empty:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(bars["ts_event"], utc=True).dt.tz_convert(ET)
    tod = ts - ts.dt.normalize()
    at_et = at.tz_convert(ET)
    cutoff = at_et - at_et.normalize()
    keep = (tod >= pd.Timedelta(hours=9, minutes=30)) & (tod < cutoff)
    df = pd.DataFrame({"date": ts.dt.date, "symbol": bars["symbol"], "volume": bars["volume"].astype(float)})[keep]
    daily = df.pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum").fillna(0.0)
    if session not in daily.index:
        return pd.Series(np.nan, index=daily.columns)
    prior = daily[daily.index < session].tail(n_days)
    if len(prior) < min_days:
        return pd.Series(np.nan, index=daily.columns, name="rel_volume")
    avg = prior.mean()
    return (daily.loc[session] / avg.where(avg > 0)).rename("rel_volume")


def max_block(trades: pd.DataFrame, at: pd.Timestamp, publisher_ids: list[int] | None = None) -> pd.Series:
    """symbol -> largest single print (shares) before `at`. With `publisher_ids`, count only prints from these
    publishers (e.g. the off-exchange TRFs), so the exchange's opening cross, which prints as one large trade at
    09:30, is not taken for a block."""
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades[pd.to_datetime(trades["ts_event"], utc=True) < at]
    if publisher_ids is not None:
        t = t[t["publisher_id"].isin(publisher_ids)]
    return t.groupby("symbol")["size"].max().astype(float).rename("max_block")


def buy_blocks(trades: pd.DataFrame, at: pd.Timestamp, min_shares: float,
               publisher_ids: list[int] | None = None) -> pd.Series:
    """symbol -> number of block prints (>= min_shares) before `at` that look buyer-initiated.

    Block prints don't say who initiated them, so this uses the tick test: a print above the last
    different price before it (an uptick, or a repeat of an uptick price) counts as a buy. Every print
    sets the tick; only prints from `publisher_ids` (the off-exchange TRFs) count as blocks.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.assign(ts=pd.to_datetime(trades["ts_event"], utc=True))
    t = t[t["ts"] < at].sort_values("ts", kind="stable")
    tick = np.sign(t.groupby("symbol")["price"].diff()).replace(0, np.nan)
    t["up"] = tick.groupby(t["symbol"]).ffill() > 0
    blocks = t[(t["size"] >= min_shares) & t["up"]]
    if publisher_ids is not None:
        blocks = blocks[blocks["publisher_id"].isin(publisher_ids)]
    return blocks.groupby("symbol").size().reindex(t["symbol"].unique(), fill_value=0).astype(float).rename("buy_blocks")


def call_put_ratio(options: pd.DataFrame) -> pd.Series:
    """underlying -> total call volume / total put volume, from OSI contract symbols ('ROOT  YYMMDDC00012500')."""
    if options.empty:
        return pd.Series(dtype=float)
    osi = options["symbol"].astype(str).str.extract(r"^(?P<root>.{1,6}?)\s*(?P<exp>\d{6})(?P<cp>[CP])\d{8}$")
    df = pd.DataFrame({"root": osi["root"].str.strip(), "cp": osi["cp"], "volume": options["volume"].astype(float)})
    df = df.dropna(subset=["cp"])
    if df.empty:
        return pd.Series(dtype=float)
    vol = df.pivot_table(index="root", columns="cp", values="volume", aggfunc="sum").reindex(columns=["C", "P"]).fillna(0)
    ratio = vol["C"] / vol["P"].where(vol["P"] > 0)
    ratio[(vol["P"] == 0) & (vol["C"] > 0)] = np.inf
    return ratio.rename("cp_ratio")


def record_issued(ideas: pd.DataFrame, issued: dict[str, float]) -> None:
    """Add tonight's new TRADE rows to the session ledger."""
    if ideas is None or ideas.empty:
        return
    t = ideas[ideas["action"] == "TRADE"]
    for sym, r in t.iterrows():
        issued[sym] = float(r["shares"] * r["limit"])


# --------------------------------------------------------------------------- universe stats
def universe_stats(bars: pd.DataFrame, cfg: dict, extra_dates=()) -> dict[str, pd.DataFrame]:
    """Per-session tables built ONLY from earlier sessions (no look-ahead).

    bars: columns date, symbol, close, volume (one row per session per symbol).
    Returns wide DataFrames indexed by session date: prev_close, adv_usd, sigma.
    extra_dates: sessions to include even without a bar yet (tonight, for live use).
    """
    u = cfg["universe"]
    close = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    close = close.reindex(sorted(set(close.index) | set(extra_dates)))   # e.g. tonight's session, no bar yet
    vol = bars.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last").reindex_like(close)
    ratio = close / close.shift(1)
    # |log return| above the cap is treated as a split / bad print, not a real move
    ret = (ratio - 1).where(np.log(ratio).abs() <= u["max_abs_log_return"])
    n, m = u["lookback_days"], u["min_history_days"]
    dollar = close * vol
    return {
        "prev_close": close.shift(1),
        "adv_usd": dollar.rolling(n, min_periods=m).mean().shift(1),
        "sigma": ret.rolling(n, min_periods=m).std().shift(1),
    }


def universe_for(stats: dict[str, pd.DataFrame], date, listing: pd.Series, cfg: dict) -> pd.DataFrame:
    """Tradable universe for one session: listing venue, price band (on prior close), liquidity."""
    u = cfg["universe"]
    df = pd.DataFrame({k: v.loc[date] for k, v in stats.items()}).dropna()
    df = df.join(listing.rename("venue"), how="inner")
    df = df[df["venue"].isin(u["venues"])]
    band = (df["prev_close"] >= u["min_price"] * 0.9) & (df["prev_close"] <= u["max_price"] * 1.1)
    df = df[band & (df["adv_usd"] >= u["min_adv_usd"])]
    bad = tuple(u.get("exclude_suffixes", []))
    if bad:
        df = df[~((df.index.str.len() == 5) & df.index.str.endswith(bad))]
    df.index.name = "symbol"
    return df
