# Closing Auction Buy-Imbalance Playbook

Sep 30, 2026 · @Ahmed Alsalahy

Buy Nasdaq-listed stocks priced $1–10 that show a large, stable buy imbalance in the last seven minutes before the close, and sell them in the closing auction. Trade it only if the out-of-sample backtest is clearly positive after costs; until then it is an idea, not an edge.

## The evening, minute by minute

**The last 11 minutes · 2 action windows, 3 cut-offs**

| ET | UK | What happens |
| --- | --- | --- |
| 15:49:00 | 20:49:00 | `live_signals.py` connects to the imbalance feed (and replays from here if started late) |
| 15:50:00 | 20:50:00 | **Cut-off:** on-close orders can no longer be cancelled, and NYSE stops accepting them. Nasdaq's closing imbalance starts, updated every 10 s |
| 15:53:00 | 20:53:00 | **Window A** ideas print (impact model) |
| 15:54:30 | 20:54:30 | Window A last call |
| 15:55:00 | 20:55:00 | **Cut-off:** Nasdaq MOC entry. The near clearing price starts, and updates come every second |
| 15:56:00 | 20:56:00 | **Window B** ideas print (near-price model) |
| 15:57:30 | 20:57:30 | Window B last call |
| 15:58:00 | 20:58:00 | **Cut-off:** Nasdaq LOC entry |
| 16:00:00 | 21:00:00 | Closing cross: exits fill at the official close |

Each window gives you about two minutes to buy before its exit cut-off. In late October and late March the UK times are one hour earlier.

## The equations

Every trade idea is ranked by **edge**, the expected move to the close minus the round-trip cost. Window A uses an impact model because Nasdaq publishes no near price before 15:55 ET. Window B uses Nasdaq's own near clearing price.

Size of the imbalance, in dollars and against normal trading:

```math
\text{notional} = Q_{imb} \cdot P_{ref} \qquad I = \frac{\text{notional}}{\text{ADV}_{\$,20d}} \qquad R = \frac{Q_{imb}}{Q_{paired}}
```

Expected move to the close:

```math
E_A = k \cdot \sigma_{20d} \cdot \sqrt{I} \qquad\qquad E_B = \beta \cdot \left( \frac{P_{near}}{P_{ref}} - 1 \right)
```

Cost and edge (spread in basis points, commission per share):

```math
C = \frac{\text{spread}}{2} + \text{slippage} + \frac{2 \cdot \text{commission}}{P_{ref}} \qquad\qquad \text{edge} = E - C
```

| Symbol | Meaning | Default | Where it comes from |
| --- | --- | --- | --- |
| Q\_imb, Q\_paired | Unmatched and matched shares in the closing auction | — | Databento imbalance feed |
| P\_ref, P\_near | Reference price; near indicative clearing price | — | Databento imbalance feed |
| ADV$ 20d | Average daily dollar volume, previous 20 sessions | — | EQUS.SUMMARY daily bars |
| σ 20d | Standard deviation of daily returns, previous 20 sessions | — | EQUS.SUMMARY daily bars |
| k | Impact scale | 0.75 | Calibrate in the backtest |
| β | Share of the near-price gap that survives to the close | 0.50 | Calibrate in the backtest |
| spread | Assumed full spread: $1–2 / $2–5 / $5–10 | 80 / 40 / 20 bps | Replace with your own fills |
| slippage | Extra paid on entry | 5 bps | Replace with your own fills |
| commission | Per share, per leg (minimum $0.35) | $0.0035 | Your broker |

The default k and β are starting guesses, not facts. The backtest prints fitted values from the first half of your date range; use those.

## Filters

A stock becomes a TRADE only if it passes every rule below, checked in this order. The first rule it fails is printed as the reason.

| # | Rule | Default | Why |
| --- | --- | --- | --- |
| 1 | Data complete (price, imbalance, σ, ADV all present) | — | Missing data must never produce a trade |
| 2 | Buy-side imbalance | side = B | You asked for buying imbalances only |
| 3 | Reference price in band | $1–10 | Your universe |
| 4 | Liquid enough | ADV$ ≥ $2M | Wide spreads eat the edge |
| 5 | Fresh snapshot | ≤ 30 s old | A stale number is not a signal |
| 6 | Imbalance big in dollars | ≥ $100k | Small imbalances get absorbed |
| 7 | Imbalance big vs normal trading | I ≥ 1% | Size relative to the stock matters |
| 8 | Auction one-sided | R ≥ 0.20 | Deep paired books absorb pressure |
| 9 | No side flip in the look-back | 60 s (A), 15 s (B) | Flips mean offsetting orders are arriving |
| 10 | Not shrinking | ≤ 50% below its recent peak | A melting imbalance is being filled |
| 11 | Window B only: near price exists and gap ≤ 8% | — | Bigger gaps are usually thin books |
| 12 | Edge high enough | ≥ 25 bps | Must beat costs with room to spare |
| 13 | Real commission at this size still leaves edge ≥ 25 bps | — | The $0.35 minimum ticket matters on small orders |

The watch list shows the 30 largest buy imbalances by dollars, including the ones that fail. Only Nasdaq-listed stocks can be TRADE rows. NYSE's cut-off for sell-on-close orders is 15:50 ET, the same minute its imbalance feed starts, so NYSE and NYSE American names are shown as WATCH only.

## Entry, exit and sizing

Buy with a limit order, then send the sell-in-the-auction order **only for the shares that filled**. Never send the exit first. After 15:50 ET Nasdaq will not let you cancel on-close orders, so an exit sent before the buy fills can leave you short at the close.

1. Send **BUY limit** at the price shown. That price is the reference plus half the spread plus slippage, rounded up to the cent.
2. When it fills, send the exit for the filled quantity only.
   - Window A: **SELL MOC** (market-on-close) before **15:55 ET**.
   - Window B: **SELL LOC** (limit-on-close) at the reference price shown before **15:58 ET**. A lower limit gains nothing: Nasdaq re-prices a sell LOC entered after 15:55 that is below the 15:50 or 15:55 reference price. If the close prints below it, the order does not fill and you hold the stock overnight.
3. If the buy has not filled about 15 seconds before the cut-off, cancel it and drop the idea.
4. Confirm with your broker that it accepts MOC and LOC orders to Nasdaq right up to these cut-offs. Some brokers stop accepting them earlier.

Shares are the smallest of three caps:

```math
\text{shares} = \min\left( \frac{\text{equity} \cdot r}{P_{limit} \cdot z \cdot \sigma_{20d} \sqrt{t / 23{,}400}} ,\; 0.10 \cdot Q_{imb} ,\; \frac{\$5{,}000}{P_{limit}} \right)
```

Here r is the risk per trade (0.25% of equity), z = 2, and t is the seconds left to the close. The first cap means a 2-sigma move against you before the close costs 0.25% of equity. The second cap means you never take more than 10% of the imbalance itself. The third cap is a hard $5,000 per trade.

## Risk rules

The limits below are the tool's defaults in `config.toml`. They cap a bad evening at roughly 1.25% of equity if every position moves 2 sigma against you.

| Limit | Default | Applies to |
| --- | --- | --- |
| Risk per trade | 0.25% of equity at a 2-sigma move | Each TRADE row |
| Notional per trade | $5,000 | Each TRADE row |
| Positions per evening | 5 | All four prints together; a stock is suggested once |
| Gross bought per evening | $20,000 | All four prints together |
| Share of the imbalance | 10% | Each TRADE row |

Stop or step back when any of these happens:

- The tool prints **NO imbalance data received**. Do not trade that evening.
- Half-days (27 Nov and 24 Dec 2026) close at 13:00 ET. Run with `--close 13:00`, or skip.
- Index rebalance and quarterly expiry days (for example the Russell reconstitution in June, and third Fridays of Mar/Jun/Sep/Dec) bring very large professional imbalances. Halve size or sit out.
- After 20 live trades, if the average result is less than half the backtest TEST average, pause and find out why.
- A losing week beyond 2% of equity: stop for the rest of the week.

## Daily routine and commands

The close is normally 21:00 UK. Between 26 and 30 October 2026, and between 15 and 26 March 2027, the clocks are out of step and the close is at 20:00 UK. The tool prints both ET and UK times, so follow what it prints.

| When (UK) | Step | Command |
| --- | --- | --- |
| Afternoon, before 20:45 | Build tonight's universe (Nasdaq-listed, $1–10, ADV and σ) | `python prep_universe.py` |
| By 20:45 | Start the live tool, open your broker's order ticket | `python live_signals.py` |
| 20:53 to 20:58 | Act on TRADE rows as each print appears (timeline above) | — |
| Next day | Score yesterday's ideas against the official close | `python review.py` |
| Weekly | Compare live results with the backtest TEST period | `python review.py --date …` |

Keep your Databento key on your own computer as the setting `DATABENTO_API_KEY`. The README in the tool shows how to set it.

## Before real money

Work down this list in order. Each step can stop the project, and that is its purpose.

- [ ] Revoke the Databento key that was pasted into the chat, create a new one, and set a monthly budget limit in the Databento portal.
- [ ] Install the tool and run `python -m pytest -q tests` (12 offline tests should pass).
- [ ] Backtest at least 12 months: `python backtest.py --start 2025-10-01 --end 2026-09-29`. The backtest counts a buy only if the ask was at or below your limit, so it includes missed fills.
- [ ] Paste the fitted `k_impact` and `beta_near` from the FIT period into `config.toml` and re-run (cached, so free).
- [ ] Continue only if the **TEST** period is clearly positive after costs, over enough trades (aim for 100+).
- [ ] Replay three recent sessions (`python live_signals.py --replay <date>`) to see exactly what an evening looks like.
- [ ] Confirm with your broker: US stocks allowed on your UK account, MOC to Nasdaq accepted until 15:55 ET and LOC until 15:58 ET, and the commission per share.
- [ ] Buy live access: a Databento plan with live Nasdaq TotalView-ITCH (Plus or higher). Ask Databento to confirm the closing imbalance feed is included before signing.
- [ ] Paper-trade 20 sessions. Run `review.py` each morning and compare with the backtest.
- [ ] Go live at a quarter of the default size (set `max_notional_per_trade_usd = 1250`).
- [ ] Scale up only after 40 live trades agree with the backtest.

## Costs

Live data is the big cost; backtests and daily preparation are pay-per-use and show a quote before anything is bought.

| Item | Cost | Notes |
| --- | --- | --- |
| Databento historical data (backtest, prep, replay, review) | Pay per GB, quoted before each purchase | $125 free credit for new accounts, expires after 6 months; downloads are cached |
| Databento Standard plan | $199/month | Live data without licence fees, but the catalogue lists live TotalView-ITCH under Plus |
| Databento Plus plan (live Nasdaq imbalance) | About $1,500/month licence fees, annual contract | Needed for `live_signals.py` in live mode |
| Broker commission | Around $0.0035/share, $0.35 minimum, each leg | Set yours in `config.toml`; the tool includes it in every edge |

If the backtest does not justify about $1,500 a month, the cheaper live route is an Interactive Brokers account with TotalView (about $15/month for non-professionals). The engine would need a small adapter for that feed.

## Sources

- [Nasdaq Closing Cross FAQ](https://www.nasdaqtrader.com/content/productsservices/Trading/ClosingCrossfaq.pdf): order cut-offs, LOC re-pricing, NOII timing
- [SEC approval of NYSE Rule 123C change](https://sec.gov/files/rules/sro/nyse/2019/34-85021.pdf): NYSE closing-auction order cut-off at 15:50 ET
- [Databento: Nasdaq TotalView-ITCH specification](https://databento.com/docs/venues-and-datasets/xnas-itch): imbalance fields (near/far price), official closing price, listing venue
- [Databento: NYSE Integrated specification](https://databento.com/docs/venues-and-datasets/xnys-pillar): NYSE imbalance fields and auction types
- [Databento: imbalance schema](https://databento.com/docs/schemas-and-data-formats/imbalance)
- [Databento: US Equities Summary](https://databento.com/docs/venues-and-datasets/equs-summary): consolidated daily bars
- [Databento: historical API reference](https://databento.com/docs/api-reference-historical/metadata/metadata-list-unit-prices): cost quotes and metered pricing
- [Databento: pricing](https://databento.com/pricing) and [US Equities catalogue](https://databento.com/catalog/us-equities): plans and live entitlements
- [Nasdaq US equities price list](https://www.nasdaqtrader.com/content/ProductsServices/PriceList/Nasdaq_US_Equities_Price_List_2025_2026_2027.pdf): TotalView subscriber fees

Rules-based information, not financial advice. Past results do not predict future results.
