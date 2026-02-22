import json
import pandas as pd
from pathlib import Path

RESULT_PATH = Path("result.json")  # change if your file name differs

def main():
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    # Polymarket snapshots -> dataframe
    pm = data.get("polymarket_snapshots", [])
    df_pm = pd.json_normalize(pm)
    print("\n=== POLYMARKET DF ===")
    print(df_pm.head())
    print(f"rows: {len(df_pm)} | cols: {len(df_pm.columns)}")

    # Kalshi snapshots -> dataframe (optional)
    ks = data.get("kalshi_snapshots", [])
    df_ks = pd.json_normalize(ks)
    print("\n=== KALSHI DF ===")
    print(df_ks.head())
    print(f"rows: {len(df_ks)} | cols: {len(df_ks.columns)}")

if __name__ == "__main__":
    main()