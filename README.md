# Alpaca Russell 2000 1-Minute Downloader

Production-oriented Python tooling for downloading approximately one year of 1-minute OHLCV bars from Alpaca for a Russell 2000 ticker list. The downloader is built for research and backtesting datasets, not live trading.

It uses Alpaca's historical stock bars endpoint directly:

```text
GET https://data.alpaca.markets/v2/stocks/bars
```

The implementation explicitly handles batching, pagination, retries, rate limiting, resumability, and Parquet output.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
```

Credentials are read with `python-dotenv`. Keys are never hardcoded.

## Symbol CSV

Put your Russell 2000 ticker list here:

```text
data/symbols/russell2000.csv
```

The CSV must include at least one column named `symbol`:

```csv
symbol
SMCI
CAVA
DUOL
```

The included `data/symbols/russell2000.csv` is only a template. Replace it with your actual Russell 2000 universe before running a full download.

## Example Download

```bash
python scripts/download_bars.py \
  --symbols data/symbols/russell2000.csv \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --timeframe 1Min \
  --feed sip \
  --adjustment raw \
  --out data/raw/r2000_1min \
  --batch-size 300 \
  --requests-per-minute 180 \
  --resume
```

You can also start from `config.example.yaml`:

```bash
python scripts/download_bars.py --config config.example.yaml
```

Command-line flags override values from the YAML config.

## How It Works

The downloader uses NYSE trading days from `pandas-market-calendars`, so it avoids weekends and exchange holidays.

For each trading day:

1. Split the symbols into batches, defaulting to 300 symbols per batch (~7 batches for the full Russell 2000 universe).
2. Request regular-session bars from 09:30 to 16:00 America/New_York.
3. Convert request timestamps to UTC.
4. Call Alpaca with `timeframe=1Min`, `limit=10000`, configurable `feed`, and configurable `adjustment`.
5. Paginate with `next_page_token` until no more pages remain.
6. Write the completed day/batch immediately to Parquet.
7. Mark the day/batch completed in SQLite.

The Alpaca `limit` is treated as a total page size across the request, not per symbol. The code never assumes all requested symbols appear on the first page.

## Output Layout

Parquet files are written as:

```text
data/raw/r2000_1min/
  year=2025/
    month=01/
      date=2025-01-02/
        batch_000.parquet
        batch_001.parquet
```

Each row includes:

```text
symbol
timestamp
open
high
low
close
volume
trade_count
vwap
date
source_feed
adjustment
```

Alpaca JSON fields are mapped as:

```text
t  -> timestamp
o  -> open
h  -> high
l  -> low
c  -> close
v  -> volume
n  -> trade_count
vw -> vwap
```

Files are written to a temporary path first, then atomically renamed after a successful Parquet write.

## Resume Behavior

The manifest database is stored at:

```text
logs/download_manifest.sqlite
```

It tracks one row per `date`, `batch_index`, `feed`, and `adjustment`, with status `pending`, `completed`, or `failed`.

When `--resume` is enabled, completed batches are skipped only if:

- the manifest row is marked `completed`
- the stored symbols hash matches the current batch
- the output Parquet file still exists

Failed batches are retried on the next run.

## Rate Limiting And Retries

The default limit is `180` requests per minute to stay below a common `200/min` ceiling.

The downloader retries:

- HTTP `429`
- HTTP `500`, `502`, `503`, `504`
- request timeouts and connection errors

Retries use exponential backoff with jitter and respect `Retry-After` when Alpaca provides it. If a batch still fails after max retries, the batch is marked `failed` and the downloader continues.

If Alpaca returns HTTP `403` for the selected feed, the script prints a clear message suggesting `--feed iex` or checking SIP subscription access.

## Validate Output

```bash
python scripts/validate_dataset.py \
  --out data/raw/r2000_1min \
  --manifest logs/download_manifest.sqlite \
  --check-sorted
```

The validation script prints:

- number of Parquet files
- total rows
- min/max timestamp
- number of unique symbols
- failed or pending manifest rows
- completed manifest rows whose output file is missing
- optional sampled timestamp sort checks

## Backtest Jump Follow-Through

You can test whether sharp intraday jumps continue or revert with:

```bash
python scripts/backtest_jump_followthrough.py \
  --data data/raw/r2000_1min \
  --jump-pct 3 \
  --window-minutes 5 \
  --horizons 1m,5m,15m,30m,1h,5h,eod
```

The default hypothesis is:

```text
signal when close / close.shift(5) - 1 >= 3%
```

Signals are evaluated within each `symbol` and `date`. Each horizon is scored
only when the future bar exists later in the same trading day. For example,
`1h` requires a bar 60 minutes after the signal, `5h` requires a bar 300
minutes after the signal, and `eod` uses the final available close for that
symbol/date. Late-day signals can therefore be eligible for `eod` while being
ineligible for `5h`.

The summary includes:

- eligible signal count by horizon
- follow-through count
- revert count
- flat count
- follow-through/revert ratio
- win rate, where follow-through is the win condition
- signal frequency per trading day
- average and median future return

By default, the script applies a 5-minute cooldown per symbol/day so repeated
qualifying bars from the same jump do not dominate the counts. Use
`--allow-overlap` to count every qualifying bar, or `--cooldown-minutes N` to
choose a different cooldown.

You can export signal-level and summary results:

```bash
python scripts/backtest_jump_followthrough.py \
  --data data/raw/r2000_1min \
  --jump-pct 3 \
  --window-minutes 5 \
  --horizons 1m,5m,15m,30m,1h,5h,eod \
  --signals-out results/jump_signals.csv \
  --summary-out results/jump_summary.json
```

## Alpaca Feed Notes

`sip` provides broader consolidated market coverage if your Alpaca account has access.

`iex` is more limited, but may be available on free or basic plans. If `sip` returns HTTP `403`, try:

```bash
python scripts/download_bars.py ... --feed iex
```

## Storage Notes

One year of 1-minute bars for roughly 2,000 stocks can produce tens of GB of data depending on compression, symbol liquidity, and feed coverage. Use a local disk with plenty of free space.

Default compression is `zstd`. You can switch to `snappy` with:

```bash
python scripts/download_bars.py ... --compression snappy
```

## Survivorship Bias

If you use today's Russell 2000 tickers to download past data, your research dataset has survivorship bias. It excludes companies that were in the index historically but later delisted, merged, went bankrupt, or left the index.

True historical Russell 2000 membership requires historical constituent data from another provider. This project only downloads bars for the symbols you provide.

## Project Layout

```text
alpaca_r2000_candle/
  README.md
  requirements.txt
  .env.example
  config.example.yaml
  data/
    symbols/
      russell2000.csv
      test_symbols.csv
    raw/
  logs/
  src/
    __init__.py
    config.py
    alpaca_client.py
    rate_limiter.py
    downloader.py
    follow_through_backtest.py
    storage.py
    manifest.py
    utils.py
  scripts/
    backtest_jump_followthrough.py
    download_bars.py
    validate_dataset.py
```
