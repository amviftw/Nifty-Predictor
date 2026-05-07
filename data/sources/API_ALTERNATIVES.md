# Data source alternatives to yfinance

Notes from the May-2026 audit. Captured here so the next person who looks at
the data layer doesn't have to re-do the research.

## Why look beyond yfinance?

`yfinance` scrapes Yahoo's public site. Pros: free, no auth, broad ticker
coverage, decent history. Cons:

- **Latency on Indian tickers**: the daily candle for `*.NS` symbols often
  doesn't roll forward until well after the 15:30 IST close. We already paper
  over this with the `fast_info` live-quote overlay in `data_loader.py`, but
  it's a recurring source of bugs.
- **Rate limits / 429s**: bursty, undocumented. Batch downloads silently drop
  Indian indices during peak hours.
- **Schema drift**: column casing and MultiIndex shape changed across
  yfinance 0.2.x → 1.x; the fetchers carry a lot of compatibility code.
- **No corporate-action hygiene**: split / bonus adjustments are inconsistent.

For a dashboard that's supposed to feel "live", a more reliable upstream is
worth a look.

## Candidates evaluated

### NSE-direct community libraries (Python)

| Library         | What it does                                    | Verdict |
| --------------- | ----------------------------------------------- | ------- |
| `jugaad-data`   | Hits the modern NSE site; bhavcopies + history. | **Recommended**: most actively maintained Indian-market lib. Future-proof per its own docs. Good for OHLCV and bhavcopy ingest. |
| `nselib`        | Same idea, smaller scope.                       | Reasonable backup. |
| `nsepython`     | Wraps NSE endpoints.                            | Works, but fragile when NSE changes endpoints. |
| `nsetools`      | Real-time-ish quotes.                           | Limited history. |
| `nsepy`         | Older, popular.                                 | Stalled — depends on the legacy NSE site. |

### Hosted commercial APIs

| Provider       | Free tier                                      | Notes |
| -------------- | ---------------------------------------------- | ----- |
| Alpha Vantage  | 25 req/day, free key.                          | OK for spot quotes; insufficient for a multi-stock dashboard. |
| Finnhub        | 60 req/min free.                               | Strong global coverage; Indian tickers patchy. |
| EOD Historical Data | 20 req/day free, paid scales fine.        | Excellent India coverage incl. corporate actions. Paid. |
| FMP            | 250 req/day free.                              | Good fundamentals; thin on intraday for India. |
| Marketstack    | 100 req/month free.                            | Too tight for our use. |

### Indian broker APIs

| Provider             | Notes |
| -------------------- | ----- |
| Upstox API           | Free with brokerage account. Reliable WebSocket quotes; good for live ticks. Auth = OAuth. |
| ICICI Direct Breeze  | Free with brokerage account. 3y of second-level LTP history. |
| Zerodha Kite Connect | ₹2k/mo subscription. Industry standard for live-quote infra; not free. |

## Recommendation

For this project, the pragmatic next step is a **hybrid**:

1. **Keep yfinance as the default** for non-time-critical fetches (long
   histories for EMA chart and monthly heatmap, sectoral indices,
   global tickers, commodities). It's free, no auth, and we've already
   invested heavily in retry / fallback logic.
2. **Add `jugaad-data` as a Yahoo fallback** for two specific surfaces
   where Yahoo's daily-candle lag stings:
   - Today's stock-level OHLCV across the expanded universe (currently
     papered over with `fast_info`).
   - FII/DII flows (NSE is the source of truth anyway).
3. **Skip the commercial APIs unless we hit a real reliability wall.** None
   of the free tiers fit the volume we need (~250 stocks × multiple horizons
   × intraday refresh). Going paid (EOD Historical, Alpha Vantage Premium)
   would solve it but isn't justified for the current dashboard scale.

The `data/sources/yahoo_fetcher.py` interface is already abstracted — adding
a `JugaadFetcher` parallel class with the same surface would let us swap
sources per-call without disturbing callers.

## References (audit links)

- jugaad-data: https://github.com/jugaad-py/jugaad-data
- nselib: https://pypi.org/project/nselib/
- nsepython: https://pypi.org/project/nsepython/
- Alpha Vantage: https://www.alphavantage.co/
- Finnhub: https://finnhub.io/
- Upstox API: https://upstox.com/trading-api/
- ICICI Breeze: https://www.icicidirect.com/futures-and-options/api/breeze
- Free finance API roundup: https://noteapiconnector.com/best-free-finance-apis
