"""A fake Databento world for offline tests.

Builds real DBN bytes (via databento_dbn) so the real DBNStore.to_df parsing
path is exercised; only the network is faked.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace as NS

import databento as db
import databento_dbn as d
import numpy as np
import pandas as pd

from imbalance import common as C

PX = 1_000_000_000
SESSIONS = [x.date() for x in pd.bdate_range("2026-08-03", "2026-09-29")]
LISTING = {"AAAA": "XNAS", "BBBB": "XNAS", "CCCC": "XNYS", "DDDD": "XNAS", "EEEEW": "XNAS", "FFFF": "XNAS",
           "HHHH": "XNAS", "NA": "XNAS"}
IID = {s: 100 + i for i, s in enumerate(LISTING)}
BASE = {"AAAA": 5.00, "BBBB": 3.00, "CCCC": 4.00, "DDDD": 50.0, "EEEEW": 2.0, "FFFF": 6.0, "HHHH": 7.0, "NA": 8.0}
# closing behaviour per symbol: (imb side before flip, flips?, imbalance shares, paired, near gap, close move vs ref)
PLAN = {
    "AAAA": ("B", False, 300_000, 200_000, 0.012, 0.008),    # clean buy imbalance, closes up
    "BBBB": ("B", True, 400_000, 300_000, 0.010, -0.005),    # flips to sell inside Window A look-back
    "FFFF": ("B", False, 20_000, 900_000, 0.002, 0.000),     # too small vs paired / ADV
    "DDDD": ("B", False, 500_000, 100_000, 0.02, 0.01),      # $50 stock: outside band
    "CCCC": ("B", False, 300_000, 100_000, 0.01, 0.01),      # NYSE listed
    "HHHH": ("B", False, 250_000, 150_000, 0.015, 0.012),    # good signal but the ask runs away: never fills
    "NA": ("A", False, 100_000, 100_000, -0.01, -0.01),      # ticker literally "NA", sell imbalance
}
ASK_PREMIUM = {"HHHH": 0.02}   # ask vs reference in the seconds after the decision (default +0.1%)


def ns(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value)


def daily_close(sym: str, day: dt.date) -> float:
    i = SESSIONS.index(day) if day in SESSIONS else 0
    rng = np.random.default_rng(IID[sym] * 1000 + i)
    return round(BASE[sym] * (1 + 0.02 * rng.standard_normal()), 2)


def ref_at_close(sym: str, day: dt.date) -> float:
    prev = SESSIONS[SESSIONS.index(day) - 1]
    return daily_close(sym, prev)


def official_close(sym: str, day: dt.date) -> float:
    return round(ref_at_close(sym, day) * (1 + PLAN[sym][5]), 4)


def _meta(dataset, schema, symbols, start, end):
    mappings = []
    if symbols != "ALL_SYMBOLS":
        s0, s1 = pd.Timestamp(start).date(), pd.Timestamp(end).date() + dt.timedelta(days=1)
        for s in symbols:
            if s in IID:
                mappings.append(NS(raw_symbol=s, intervals=[NS(start_date=s0, end_date=s1, symbol=str(IID[s]))]))
    return d.Metadata(dataset=dataset, start=ns(start), end=ns(end), stype_in=d.SType.RAW_SYMBOL,
                      stype_out=d.SType.INSTRUMENT_ID, schema=d.Schema.from_str(schema),
                      symbols=[] if symbols == "ALL_SYMBOLS" else list(symbols), partial=[], not_found=[],
                      mappings=mappings)


def imbalance_records(dataset: str, symbols, start: pd.Timestamp, end: pd.Timestamp):
    recs = []
    day = start.tz_convert(C.ET).date()
    close_at = C.close_time(day)
    venue = {"XNAS.ITCH": "XNAS", "XNYS.PILLAR": "XNYS"}[dataset]
    for s in symbols:
        if s not in PLAN or LISTING[s] != venue:
            continue
        side0, flips, imb, paired, gap, _ = PLAN[s]
        ref = ref_at_close(s, day)
        t = close_at - pd.Timedelta(minutes=10)
        while t < close_at:
            step = 10 if t < close_at - pd.Timedelta(minutes=5) else 1
            if start <= t < end:
                secs_left = (close_at - t).total_seconds()
                side = side0
                if flips and 430 < secs_left <= 470:        # sell for 40 s just before Window A
                    side = "A"
                near = ref * (1 + gap) if secs_left <= 300 else None   # Nasdaq: near price only from 15:55
                recs.append(d.ImbalanceMsg(
                    publisher_id=2, instrument_id=IID[s], ts_event=ns(t), ts_recv=ns(t),
                    ref_price=int(round(ref * PX)), auction_time=ns(close_at),
                    cont_book_clr_price=int(round(near * PX)) if near else d.UNDEF_PRICE,
                    auct_interest_clr_price=d.UNDEF_PRICE, paired_qty=paired, total_imbalance_qty=imb,
                    auction_type="C", side=d.Side.BID if side == "B" else d.Side.ASK, significant_imbalance="L"))
            t += pd.Timedelta(seconds=step)
    return recs


class FakeHistorical:
    def __init__(self):
        self.calls = []
        self.metadata = NS(get_cost=self._cost, get_dataset_range=self._range)
        self.timeseries = NS(get_range=self._get_range)

    def _cost(self, **q):
        return 0.01

    def _range(self, dataset):
        return {"end": "2026-09-30T00:00:00Z", "schema": {}}

    def _get_range(self, dataset, schema, symbols, start, end, **_):
        self.calls.append((dataset, schema))
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        recs = []
        if schema == "definition":
            day = start.date()
            if day.weekday() >= 5:   # the real API times out on weekend definition requests
                raise db.BentoServerError(http_status=504, message="504 gateway timed out")
            if day in SESSIONS:
                for s, v in LISTING.items():
                    recs.append(d.InstrumentDefMsg(
                        publisher_id=2, instrument_id=IID[s], ts_event=ns(start), ts_recv=ns(start),
                        min_price_increment=10_000_000, display_factor=PX, raw_symbol=s, asset=s[:4],
                        security_type="", instrument_class=d.InstrumentClass.STOCK, security_update_action=d.SecurityUpdateAction.ADD,
                        exchange=v))
        elif schema == "ohlcv-1d":
            for day in SESSIONS:
                t = pd.Timestamp(day, tz="UTC")
                if start <= t < end:
                    for s in symbols:
                        c = daily_close(s, day)
                        recs.append(d.OHLCVMsg(rtype=d.RType.OHLCV_1D, publisher_id=90, instrument_id=IID[s], ts_event=ns(t),
                                               open=int(c * PX), high=int(c * PX), low=int(c * PX), close=int(c * PX),
                                               volume=2_000_000))
        elif schema == "imbalance":
            recs = imbalance_records(dataset, symbols, start, end)
        elif schema == "bbo-1s":
            day = start.tz_convert(C.ET).date()
            for s in symbols:
                if s not in PLAN:
                    continue
                ref = ref_at_close(s, day)
                ask = ref * (1 + ASK_PREMIUM.get(s, 0.001))
                t = start
                while t < end:
                    recs.append(d.BBOMsg(rtype=d.RType.BBO_1S, publisher_id=2, instrument_id=IID[s], ts_event=ns(t),
                                         price=int(round(ref * PX)), size=100, side=d.Side.NONE, ts_recv=ns(t),
                                         levels=d.BidAskPair(bid_px=int(round((2 * ref - ask) * PX)),
                                                             ask_px=int(round(ask * PX)), bid_sz=500, ask_sz=500)))
                    t += pd.Timedelta(seconds=1)
        elif schema == "statistics":
            day = start.tz_convert(C.ET).date()
            close_at = C.close_time(day)
            for s in symbols:
                if s in PLAN:
                    recs.append(d.StatMsg(publisher_id=2, instrument_id=IID[s], ts_event=ns(close_at), ts_recv=ns(close_at),
                                          ts_ref=0, price=int(round(official_close(s, day) * PX)), quantity=50_000,
                                          stat_type=d.StatType.CLOSE_PRICE))
        m = _meta(dataset, schema, symbols, start, end)
        return db.DBNStore.from_bytes(m.encode() + b"".join(bytes(r) for r in recs))
