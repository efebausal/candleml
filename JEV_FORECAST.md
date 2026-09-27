# Local SPY / BTC forecasts

This tool makes an on-demand, signal-only forecast from Twelve Data candles and
Jev through OpenRouter. SPY remains the default; `--market crypto` selects BTC/USD
and its 24/7 UTC calendar. SPY is an S&P 500 ETF proxy, not the cash index or a
broker CFD. BTC/USD is an exchange-market series and may differ from FTMO's BTCUSD
CFD quote. The tool does not submit orders or calculate account-specific risk.

## Requirements

- Python 3.10 or newer; the forecast tool uses only the Python standard library.
- A Twelve Data API key with access to SPY or BTC/USD 15-minute candles. Check
  your plan's request limits and real-time access in your Twelve Data account.
- An OpenRouter API key with access to Jev. Jev inference is billed to your
  OpenRouter account separately from Twelve Data.
- The HTTP client verifies TLS certificates. It uses `certifi` when installed;
  `SSL_CERT_FILE` / `SSL_CERT_DIR` can point to a custom trusted CA bundle.

Create a local `.env` file from the example and replace the placeholders with
your keys. `.env` is excluded from Git; don't add real keys to source files:

```bash
cp .env.example .env
# Edit .env and replace both placeholder values with your real keys.
```

The CLI loads `.env` automatically. Non-empty variables already set in the
shell take precedence. Jev requests use OpenRouter's Decisions API at
`https://openrouter.ai/api/alpha/decisions` with the
`~typesafe/jev-latest` model alias. Create an OpenRouter key at
`openrouter.ai/settings/keys`.

## Commands

Preview the market state without calling Jev:

```bash
python3 jev_forecast.py forecast --dry-run
```

Generate or retrieve the latest forecast:

```bash
python3 jev_forecast.py forecast
```

Preview or generate the default BTC/USD forecast (15-minute candles, one-hour
horizon):

```bash
python3 jev_forecast.py forecast --market crypto --dry-run
python3 jev_forecast.py forecast --market crypto
```

Twelve Data returns BTC timestamps in UTC. To pin a specific venue, add
`--exchange Binance` (or an exchange supported by your Twelve Data plan). The
default market returned for `BTC/USD` depends on Twelve Data's symbol mapping.

See stored forecasts and any resolved one-hour outcomes:

```bash
python3 jev_forecast.py history --limit 50
python3 jev_forecast.py history --market crypto --limit 50
```

The forecast uses the latest completed 15-minute bar and predicts the class of
the return over the next four bars (one hour). The default `FLAT` band is +/-10
basis points (0.10%); change it with `--neutral-band-bps`.

The SPY ledger is stored at `~/.candleml/jev_forecasts.sqlite3`; the BTC ledger
is separate at `~/.candleml/btc_forecasts.sqlite3`. Pass `--db` to either
command to use a different local SQLite file.

Repeated runs for the same bar, symbol, horizon, neutral band, lookback, and
model alias reuse the cached forecast. Use `--force` to make another Jev call.
Use `--allow-stale` only for research when an equity bar is outside the regular
session or either market's latest bar is older than the freshness limit.

## Evaluation notes

- Inputs contain only completed candles; the forming candle is discarded.
- Crypto mode uses UTC and continuous-market freshness checks; it does not apply
  U.S. equity session rules. FTMO instrument availability and maintenance hours
  can still differ from the Twelve Data market feed.
- The forecast deadband is not automatically adjusted for trading costs. FTMO's
  published crypto trading update lists BTCUSD commission at 0.0325% per side;
  spreads and slippage are additional. Verify current terms in your platform and
  choose `--neutral-band-bps` accordingly. See
  [FTMO's crypto trading update](https://ftmo.com/en/blog/ftmo-enhances-crypto-trading-new-instruments-and-better-spreads/).
- Existing `runs/15m_sweep` BTC baseline experiments used `--fee_bps 1.0`; that
  is below the published FTMO BTCUSD commission alone. For comparable offline
  tests, `baseline.py` accepts `--fee_bps` as a per-side cost. Add your estimated
  per-side spread/slippage to the commission assumption when choosing that value.
- Forecasts are recorded locally and scored when the matching four future bars
  are later available. Gaps/early closes may leave a forecast unresolved rather
  than fabricate a result.
- Jev's Choice probabilities and confidence are stored, but neither is treated
  as proof of calibration or predictive edge. The default neutral band is a
  research label, not a broker-cost estimate.
- The API call happens only when the command is run. There is no background
  poller, dashboard, broker connection, account-risk calculation, or trade
  execution.
