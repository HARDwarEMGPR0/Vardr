# Market Fetcher Project

A runnable Python project that wraps your fetch modules (`fetch_data`, `fetch_kalshi`, `fetch_polymarket`) into a package with a single CLI entrypoint.

## Project Structure

```text
market_fetcher_project/
  main.py
  requirements.txt
  .env.example
  README.md
  market_fetcher/
    __init__.py
    config.py
    logging_utils.py
    fetch_data.py
    fetch_kalshi.py
    fetch_polymarket.py
```

## Installation

1. Create and activate a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Set up config.

```bash
copy .env.example .env
```

## Usage

Run all sources:

```bash
python main.py --source all --pretty
```

Run only Polymarket (top 10 ranked markets):

```bash
python main.py --source polymarket --top-n 10 --pretty
```

Run only Kalshi:

```bash
python main.py --source kalshi --pretty
```

## CLI Arguments

- `--source {all,kalshi,polymarket}`: Select source to fetch.
- `--top-n INT`: Number of Polymarket markets to include (used by `all` and `polymarket`).
- `--pretty`: Print formatted JSON output.

## Notes

- Logging level is controlled by `LOG_LEVEL` in `.env`.
- API/network errors are handled in `main.py` with non-zero exit codes.
- All modules are imported and invoked from `main.py`.

## FastAPI Resolver

Start server:

```powershell
python -m uvicorn api.server:app --reload --port 8000 --log-level debug
```

Call resolver from PowerShell:

```powershell
$body = @{
  query = "Will Argentina win the 2026 FIFA World Cup?"
  side = "yes"
  notional_usd = 500
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/resolve_market?mode=fixture" -ContentType "application/json" -Body $body
```
