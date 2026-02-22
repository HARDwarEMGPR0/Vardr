# Snapshot Pipeline Script

## Run (Live)

```bash
python scripts/fetch_snapshots.py --mode live --allow-fallback --n-markets 5
```

Strict live-only:

```bash
python scripts/fetch_snapshots.py --mode live --no-fallback --n-markets 5
```

## Run (Fixture / Offline)

```bash
python scripts/fetch_snapshots.py --mode fixture --n-markets 5
```

Use this mode when outbound HTTPS is blocked (for example WinError 10013).

## Save Fixtures From Live

```bash
python scripts/fetch_snapshots.py --mode live --allow-fallback --save-fixtures
```

Core fixture files:
- `fixtures/polymarket_markets_open.json`
- `fixtures/kalshi_markets_open.json`
- `fixtures/kalshi_orderbook.json`
- `fixtures/kalshi_trades.json`

## Provenance Fields

Output includes:
- `run_status`: `live` | `fixture` | `mixed`
- `venue_status.polymarket`: `{ ok, source, error }`
- `venue_status.kalshi`: `{ ok, source, error }`

## Cross-Venue Output

Output includes:
- `cross_venue.matched_pairs` (up to 2)
- `cross_venue.kalshi_only_examples` (exactly 2)
- `cross_venue.polymarket_only_examples` (exactly 2)

## WinError 10013

If live calls fail with `[WinError 10013]`, run fixture mode or live with fallback:

```bash
python scripts/fetch_snapshots.py --mode fixture
python scripts/fetch_snapshots.py --mode live --allow-fallback
```

## Tests

```bash
pytest scripts/test_fetch_snapshots.py
```
