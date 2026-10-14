# Perfect Store dataset: data dictionary

2,000 stores, one row per store (a snapshot as of 2026-09-30). Columns 1-16 are your original data, unchanged.
Columns 17-33 were added by `enrich_perfect_store.py` (fixed random seed 42, so the file is reproducible).

**Type key:** ORIGINAL = your file · DERIVED = computed only from existing columns · SIMULATED = generated, not observed

## Original columns
store_id, region, channel, store_size_sqm, visits_per_month, on_shelf_availability, share_of_shelf, planogram_compliance,
price_compliance, promo_execution, extra_displays, sku_count_listed, perfect_store_score, sales_vs_target_pct,
perfect_store (1 if score >= 80), hit_sales_target (1 if sales_vs_target_pct >= 100).

## New columns

| Column | Type | Meaning and rule |
|---|---|---|
| retailer_chain | SIMULATED | Fictional chain name, assigned within each channel (Traditional = "Independent") |
| field_rep_id | SIMULATED | Field rep responsible for the store; 12 reps per region (e.g. NOR-04), assigned at random |
| store_age_years | SIMULATED | Years since the store opened (1-25) |
| audit_date | SIMULATED | Date of the latest audit, a weekday between 2026-07-01 and 2026-09-30 |
| monthly_sales_target | SIMULATED | Monthly sales target in currency-neutral units; scales with channel and store size, with noise |
| monthly_sales_actual | DERIVED from simulated target | monthly_sales_target x sales_vs_target_pct / 100 (consistent with your original column) |
| sales_gap | DERIVED | monthly_sales_actual - monthly_sales_target (negative = shortfall) |
| score_band | DERIVED | Perfect (>= 80), Good (70 to <80), At risk (60 to <70), Critical (< 60) |
| gap_to_perfect | DERIVED | Points needed to reach 80; 0 if already perfect |
| weakest_kpi | DERIVED | KPI where the store ranks lowest versus all other stores (percentile rank, because KPI scales differ) |
| strongest_kpi | DERIVED | KPI where the store ranks highest versus all other stores |
| kpis_below_median | DERIVED | How many of the 5 KPIs are below the all-store median (0-5) |
| est_oos_skus | DERIVED (estimate) | sku_count_listed x (100 - on_shelf_availability) / 100, rounded; approximate SKUs out of stock |
| prev_audit_score | SIMULATED | Score at the previous audit: close to the current score, pulled slightly toward the average, plus noise |
| score_change | DERIVED from simulated | perfect_store_score - prev_audit_score |
| score_trend | DERIVED from simulated | Improving (change > +1.5), Declining (< -1.5), otherwise Stable |
| execution_outcome | DERIVED | Perfect & on target / Perfect but missed target / Not perfect but on target / Not perfect & missed target |

## Using the data honestly
- **Simulated columns are not findings.** Their relationships to the KPIs are not real. If an analysis "finds" that a rep or chain
  matters, that comes from the random assignment. State this in the README: "Columns marked SIMULATED are synthetic."
- **Leakage when predicting outcomes.** To predict `hit_sales_target`, do not use `sales_vs_target_pct`, `monthly_sales_actual`,
  `sales_gap`, `execution_outcome`, `perfect_store_score`, `perfect_store`, `score_band` or `gap_to_perfect`. They contain the answer or
  are built from it. Use the five KPIs, `extra_displays`, `visits_per_month`, `sku_count_listed`, `channel`, `region` and store attributes.
- **Original data quirks.** `store_size_sqm` does not vary by channel (convenience stores up to about 2,500 sqm) and region has almost no
  effect on score. This is typical of synthetic data. Leave the columns as they are, but do not draw conclusions about store size by channel.
