"""Adds 17 columns to perfect_store.csv. Your original 16 columns are never modified.

Usage: python enrich_perfect_store.py perfect_store.csv perfect_store_enriched.csv

Two kinds of new column:
  DERIVED    computed only from existing columns (nothing invented)
  SIMULATED  generated with a fixed random seed (NOT observed data). Label them as synthetic in your README.
Every rule is below, so a reader can see exactly what is real and what is generated."""
import sys
import numpy as np
import pandas as pd

SEED = 42
AS_OF = pd.Timestamp("2026-09-30")                       # snapshot date for audit_date
PERFECT_THRESHOLD = 80                                   # matches the existing perfect_store flag
KPIS = ["on_shelf_availability", "share_of_shelf", "planogram_compliance", "price_compliance", "promo_execution"]

CHAINS = {"Hypermarket": (["MegaMart", "CityHyper"], [0.6, 0.4]),
          "Supermarket": (["FreshWay", "GreenGrocer", "PrimeMart"], [0.4, 0.35, 0.25]),
          "Convenience": (["QuickStop", "CornerPlus", "OpenNow"], [0.4, 0.35, 0.25]),
          "Traditional": (["Independent"], [1.0])}
TARGET_MEDIAN = {"Hypermarket": 400_000, "Supermarket": 120_000, "Convenience": 30_000, "Traditional": 12_000}

def add_simulated(df, rng):
    # who and where
    df["retailer_chain"] = [rng.choice(CHAINS[c][0], p=CHAINS[c][1]) for c in df["channel"]]
    df["field_rep_id"] = [f"{r[:3].upper()}-{rng.integers(1, 13):02d}" for r in df["region"]]   # 12 reps per region
    df["store_age_years"] = np.clip(rng.gamma(3, 3.5, len(df)).round(), 1, 25).astype(int)
    days = pd.bdate_range("2026-07-01", AS_OF)                                                    # audits happen on weekdays
    df["audit_date"] = days[rng.integers(0, len(days), len(df))].strftime("%Y-%m-%d")
    # money: the target is simulated; the actual uses the EXISTING sales_vs_target_pct, so they stay consistent
    base = df["channel"].map(TARGET_MEDIAN) * (df["store_size_sqm"] / 1200) ** 0.5 * rng.lognormal(0, 0.25, len(df))
    df["monthly_sales_target"] = ((base / 100).round() * 100).astype(int)
    df["monthly_sales_actual"] = (df["monthly_sales_target"] * df["sales_vs_target_pct"] / 100).round().astype(int)
    df["sales_gap"] = df["monthly_sales_actual"] - df["monthly_sales_target"]
    # previous audit: pulled toward the average, plus noise (so some stores improve and some decline)
    prev = 0.9 * df["perfect_store_score"] + 0.1 * df["perfect_store_score"].mean() + rng.normal(0, 3.0, len(df))
    df["prev_audit_score"] = prev.clip(40, 100).round(2)
    df["score_change"] = (df["perfect_store_score"] - df["prev_audit_score"]).round(2)
    df["score_trend"] = np.select([df["score_change"] > 1.5, df["score_change"] < -1.5], ["Improving", "Declining"], "Stable")
    return df

def add_derived(df):
    s = df["perfect_store_score"]
    df["score_band"] = np.select([s >= 80, s >= 70, s >= 60], ["Perfect", "Good", "At risk"], "Critical")
    df["gap_to_perfect"] = (PERFECT_THRESHOLD - s).clip(lower=0).round(2)
    pct = df[KPIS].rank(pct=True)                          # each KPI ranked against all other stores
    df["weakest_kpi"] = pct.idxmin(axis=1)                 # lowest relative to peers (KPI scales differ, so raw values mislead)
    df["strongest_kpi"] = pct.idxmax(axis=1)
    df["kpis_below_median"] = (df[KPIS] < df[KPIS].median()).sum(axis=1)
    df["est_oos_skus"] = (df["sku_count_listed"] * (100 - df["on_shelf_availability"]) / 100).round().astype(int)
    perfect, hit = df["perfect_store"] == 1, df["hit_sales_target"] == 1
    df["execution_outcome"] = np.select([perfect & hit, perfect & ~hit, ~perfect & hit],
                                        ["Perfect & on target", "Perfect but missed target", "Not perfect but on target"],
                                        "Not perfect & missed target")
    return df

ORDER = ["retailer_chain", "field_rep_id", "store_age_years", "audit_date",
         "monthly_sales_target", "monthly_sales_actual", "sales_gap",
         "score_band", "gap_to_perfect", "weakest_kpi", "strongest_kpi", "kpis_below_median", "est_oos_skus",
         "prev_audit_score", "score_change", "score_trend", "execution_outcome"]

def validate(orig, out):
    assert len(out) == len(orig) == out["store_id"].nunique()
    assert out[orig.columns].equals(orig), "original columns changed!"
    assert not out[ORDER].isna().any().any()
    ratio = out["monthly_sales_actual"] / out["monthly_sales_target"] * 100
    assert (ratio - out["sales_vs_target_pct"]).abs().max() < 0.1, "actual/target no longer matches sales_vs_target_pct"
    assert ((out["score_band"] == "Perfect") == (out["perfect_store"] == 1)).all(), "score_band disagrees with perfect_store"
    assert out["execution_outcome"].value_counts().sum() == len(out)

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    orig = pd.read_csv(src)
    out = add_derived(add_simulated(orig.copy(), np.random.default_rng(SEED)))[list(orig.columns) + ORDER]
    validate(orig, out)
    out.to_csv(dst, index=False)
    print(f"{len(orig.columns)} columns -> {len(out.columns)} columns, {len(out)} rows. Validation passed.")
