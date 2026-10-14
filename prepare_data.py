"""Builds the SQLite database from the Perfect Store CSV. Runs automatically on first use, or by hand:

    python prepare_data.py                              # uses data/perfect_store_enriched.csv
    python prepare_data.py path/to/perfect_store.csv    # the RAW 16-column file also works:
                                                        # derived and simulated columns are added for you
"""
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from enrich_perfect_store import ORDER, SEED, add_derived, add_simulated

ROOT = Path(__file__).parent
DEFAULT_CSV = ROOT / "data" / "perfect_store_enriched.csv"
DB_PATH = ROOT / "data" / "stores.db"

def load_stores(csv_path=DEFAULT_CSV):
    df = pd.read_csv(csv_path)
    if "score_band" not in df.columns:                       # raw file: add the 17 extra columns
        original = list(df.columns)
        df = add_derived(add_simulated(df, np.random.default_rng(SEED)))[original + ORDER]
    return df

def build_database(csv_path=DEFAULT_CSV, db_path=DB_PATH):
    df = load_stores(csv_path)
    Path(db_path).parent.mkdir(exist_ok=True)
    Path(db_path).unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    df.to_sql("stores", con, index=False)
    con.commit()
    con.close()
    return len(df)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Built {DB_PATH.name} with {build_database(src)} stores from {Path(src).name}")
