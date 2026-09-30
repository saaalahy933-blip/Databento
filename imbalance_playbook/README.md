# Closing-auction BUY-imbalance tool

Prints ranked **BUY ideas** from the live Nasdaq closing-auction imbalance for stocks priced $1–10.
Each TRADE row gives the ticker, the number of shares, the highest price to pay and the exit order.
The rules and the equations behind each column are in the [playbook document](PLAYBOOK.md).

Rules-based output, not financial advice. Backtest first, then paper-trade, then go small.

## 1. Install (once)

You need Python 3.11 or newer.

```bash
cd imbalance_playbook
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests          # 11 offline tests should pass
```

## 2. Your Databento key (once)

Create a **new** key in the Databento portal (API keys page) and revoke any key you have pasted anywhere.
Keep the key on your computer only:

```bash
export DATABENTO_API_KEY=db-xxxxxxxx      # macOS/Linux (add to ~/.zshrc or ~/.bashrc)
setx DATABENTO_API_KEY db-xxxxxxxx        # Windows (then open a new terminal)
```

In the Databento portal, set a monthly budget limit as a second safety net.

## 3. Backtest before anything else

```bash
python backtest.py --start 2025-10-01 --end 2026-09-29
```

The tool shows a cost estimate and waits for `y` before buying anything.
For a backtest the estimate is based on a sample of sessions plus 20%, and `--max-cost` is a hard cap.
Everything it downloads is cached in `./cache`, so re-runs with new settings cost nothing.

The backtest is deliberately strict:
- A buy only counts if the Nasdaq ask was at or below your limit within 60 seconds. If not, it's a miss, and misses are often the stocks that ran.
- A stock is bought at most once per evening.
- If a Window B sell-on-close order doesn't fill, the stock is sold at the next day's open.

The results are split into a **FIT** period (the first half, by default) and a **TEST** period.
The calibrated `k_impact` and `beta_near` come from the FIT period only.
1. Paste them into `config.toml`.
2. Re-run the backtest.
3. Judge the strategy on the **TEST** lines only. If those are not clearly positive after costs, don't trade it.

## 4. Every trading day (UK times; US close is normally 21:00 UK)

| When (UK) | Command | What happens |
|---|---|---|
| Afternoon, before 20:45 | `python prep_universe.py` | Builds `universe.csv`: Nasdaq-listed $1–10 stocks with 20-day $ADV and volatility. Costs cents. |
| By 20:45 | `python live_signals.py` | Waits, connects to the live feed at 20:49 and prints ideas at 20:53, 20:54:30, 20:56 and 20:57:30. |
| Next day | `python review.py` | Scores yesterday's TRADE ideas against the official close. |

Live mode needs a Databento plan that includes **live** Nasdaq TotalView-ITCH, which is the Plus plan or higher.
To see exactly what the output would have looked like on a past day, run it with historical data (costs cents):

```bash
python prep_universe.py --session 2026-09-29
python live_signals.py --replay 2026-09-29
```

**Clock changes:** twice a year the UK and US change their clocks on different dates, and the close then moves to 20:00 UK.
In 2026 that is 26–30 October, and in 2027 it is 15–26 March.
The tool always prints both ET and UK times, so follow what it prints.

**Half-days** (for example 27 Nov and 24 Dec 2026) close at 13:00 ET. On those days run `python live_signals.py --close 13:00`.

## 5. Reading the output

```
 # TICKER VEN  REF   IMB SH   IMB $  /ADV  /PAIR  NEAR   EXP   COST  EDGE  ACTION
 1 AAAA   XNAS 4.94 300,000 $1.48M 14.7%  1.50     -  0.88% 0.39% 0.49%  >> BUY 1,012 @ <= 4.96  then SELL MOC
```

| Column | Meaning |
|---|---|
| REF | Exchange reference price for the auction |
| IMB SH / IMB $ | Unmatched buy shares, and those shares × REF |
| /ADV | IMB $ as a share of the stock's 20-day average $ volume |
| /PAIR | Imbalance shares ÷ shares already matched |
| NEAR | Nasdaq's near indicative clearing price, published from 15:55 ET |
| EXP | Expected move to the close |
| COST | Half the spread + slippage + commissions |
| EDGE | EXP − COST |

**How to act on a TRADE row:**
1. Send the BUY limit order.
2. **Only for shares that actually filled**, send the exit order.
   - Window A: **SELL MOC** before 15:55 ET.
   - Window B: **SELL LOC** at the price shown, before 15:58 ET.
3. If the buy hasn't filled about 15 seconds before that cut-off, cancel it and drop the idea.

Never send the exit first: on-close orders can't be cancelled after 15:50 ET, so an unfilled buy would leave you short.
Check that your broker accepts MOC and LOC orders right up to these times, because some brokers stop earlier.

Each stock is suggested at most once per evening. At the later prints, a stock you already bought shows as `held`.
The size limits in `config.toml` (`max_positions`, `max_gross_usd`) count the whole evening, not each print.

## Files

| File | Purpose |
|---|---|
| `config.toml` | Every threshold, cost and size limit |
| `imbalance/engine.py` | The equations (pure maths, tested) |
| `imbalance/common.py` | Clock, cached Databento access with cost guard, printing |
| `imbalance/build.py` | Turns Databento data into engine inputs |
| `prep_universe.py`, `live_signals.py`, `backtest.py`, `review.py` | The four daily/one-off steps |
| `results/` | Idea logs and backtest output |
