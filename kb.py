"""The knowledge base: three small RAG sources the agent searches before it answers.
  GLOSSARY     what each metric means and which column to use  -> keeps definitions consistent
  MODEL_CARDS  what each ML model needs and where it fails      -> grounds model choices
  PLAYBOOK     generic field-execution guidance per KPI         -> turns findings into actions
Rewrite these in your own words: the quality of the answers depends on them."""
import hashlib
from pathlib import Path
import chromadb

GLOSSARY = [
    {"id": "on_shelf_availability", "term": "on-shelf availability", "text": "On-shelf availability (OSA, availability, in-stock rate): percentage of required SKUs found on the shelf, "
     "column on_shelf_availability (0-100). A group average is a simple AVG across stores, not weighted by size or sales."},
    {"id": "share_of_shelf", "term": "share of shelf", "text": "Share of shelf (SoS, shelf share): our share of the shelf space in the category, as a percentage, column share_of_shelf. "
     "It is naturally lower than the other KPIs, so compare it with other stores, not with the other KPIs."},
    {"id": "planogram_compliance", "term": "planogram compliance", "text": "Planogram compliance (planogram, shelf layout compliance): percentage of products placed as the planogram "
     "specifies, column planogram_compliance (0-100)."},
    {"id": "price_compliance", "term": "price compliance", "text": "Price compliance (price check, correct pricing): percentage of products at the correct shelf price, column price_compliance (0-100)."},
    {"id": "promo_execution", "term": "promo execution", "text": "Promo execution (promotion compliance, promo compliance): percentage of agreed promotions executed correctly, "
     "column promo_execution (0-100)."},
    {"id": "perfect_store_score", "term": "perfect store score", "text": "Perfect store score (PS score, store score, audit score): a weighted combination of the five KPIs, column "
     "perfect_store_score (0-100). The exact weights are not in the data. A perfect store has a score of 80 or more (perfect_store = 1). "
     "Perfect store rate = AVG(perfect_store) * 100."},
    {"id": "score_band", "term": "score band", "text": "Score bands, column score_band: Perfect (score >= 80), Good (70 to under 80), At risk (60 to under 70), Critical (under 60). "
     "'At-risk stores' in everyday language means score_band is At risk or Critical, that is a score below 70. gap_to_perfect = points needed to reach 80."},
    {"id": "sales_vs_target", "term": "sales vs target", "text": "Sales versus target: sales_vs_target_pct, where 100 means exactly on target. hit_sales_target = 1 when it is 100 or more. "
     "Hit rate = AVG(hit_sales_target) * 100. Sales shortfall = sales_gap = monthly_sales_actual - monthly_sales_target (negative = below target). "
     "Amounts are currency-neutral and synthetic."},
    {"id": "weakest_kpi", "term": "weakest kpi", "text": "Weakest KPI (biggest weakness, what to fix first), column weakest_kpi: the KPI where the store ranks lowest compared with all "
     "other stores. strongest_kpi is the opposite. kpis_below_median counts how many of the 5 KPIs are below the all-store median."},
    {"id": "execution_outcome", "term": "execution outcome", "text": "Execution outcome, column execution_outcome: Perfect & on target, Perfect but missed target, Not perfect but on target, "
     "Not perfect & missed target. 'Perfect stores that missed target' = Perfect but missed target."},
    {"id": "priority_stores", "term": "priority stores", "text": "Priority stores (stores to visit first, stores to fix first): stores that are at risk AND missed their sales target, "
     "that is perfect_store_score < 70 AND hit_sales_target = 0. Rank them by sales_gap (most negative first) or gap_to_perfect."},
    {"id": "score_trend", "term": "score trend", "text": "Score trend (improving, declining, momentum), column score_trend: Improving when score_change is above +1.5, Declining when "
     "below -1.5, otherwise Stable. score_change = perfect_store_score - prev_audit_score. Only one previous audit exists and it is synthetic."},
    {"id": "store_groups", "term": "store groups", "text": "Questions by region, channel, retailer chain or field rep: group the stores table by that column. Report the number of stores "
     "alongside averages. Field rep ids, chains and store age are synthetic demo columns; mention that in the caveats."},
    {"id": "visits_and_displays", "term": "visits and displays", "text": "visits_per_month: field visits to the store per month (1-8). extra_displays: number of additional displays "
     "set up in the store (0-7). sku_count_listed: SKUs listed for the store. These are original audit columns."},
    {"id": "estimated_oos", "term": "estimated out of stock", "text": "Estimated out-of-stock SKUs, column est_oos_skus: sku_count_listed x (100 - on_shelf_availability) / 100, rounded. "
     "An approximation, not an audited count."},
    {"id": "relative_dates", "term": "relative dates", "text": "Relative dates (today, last month, this quarter, latest, recent): resolve against data_as_of from get_data_schema (the last audit_date), "
     "not the real calendar date. 'Last month' means the previous full calendar month before the month of data_as_of. Always state the dates used."},
    {"id": "growth", "term": "growth", "text": "Change versus a previous period or group: compute the same metric for both and report the percent change (new - old) / old * 100 "
     "along with both raw values. Percentage-point differences should be called percentage points."},
]

MODEL_CARDS = [
    {"id": "logistic_regression_store_target", "task": "target_prediction", "runnable": True,
     "text": "Logistic regression to predict whether a store hits its sales target (binary classification) from execution KPIs, extra displays, visit "
             "frequency, SKU count, size, channel and region. Strength: simple, explainable coefficients. Weakness: linear effects only. Must exclude "
             "columns that leak the answer: sales_vs_target_pct, sales amounts, perfect_store_score and perfect_store. Metric: PR-AUC."},
    {"id": "random_forest_store_target", "task": "target_prediction", "runnable": True,
     "text": "Random forest to predict whether a store hits its sales target. Handles nonlinear effects and interactions between KPIs, robust on a couple "
             "of thousand stores. Weakness: less explainable, so use permutation importance to show key drivers. Exclude leaky columns. Metric: PR-AUC."},
    {"id": "gradient_boosting_store_target", "task": "target_prediction", "runnable": True,
     "text": "Gradient boosting classifier to predict store sales-target success. Often the most accurate on tabular data. Weakness: can overfit about "
             "2,000 rows and is harder to explain. Metric: PR-AUC."},
    {"id": "kmeans_store_segmentation", "task": "segmentation", "runnable": True,
     "text": "K-Means to segment stores by their five execution KPIs (scaled). Strength: produces store archetypes managers can act on, for example strong "
             "on availability but weak on promo. Weakness: when KPIs are mostly independent the groups overlap, so always check the silhouette. Metric: silhouette."},
    {"id": "hierarchical_store_segmentation", "task": "segmentation", "runnable": True,
     "text": "Hierarchical (Ward) clustering to segment stores by KPI profile. No random start, fine for a few thousand stores. Weakness: memory grows "
             "quickly with many stores. Metric: silhouette."},
    {"id": "isolation_forest_store_anomaly", "task": "anomaly", "runnable": True,
     "text": "Isolation Forest to find stores with unusual KPI combinations (suspicious or mis-keyed audits, exceptional performers). No labels needed. "
             "Weakness: it only says a store is rare, not wrong, so use it to choose stores for a re-audit. Metric: none without labels; review the top-ranked stores."},
    {"id": "store_score_forecast", "task": "forecasting_stores", "runnable": False,
     "text": "Predicting which stores will decline or improve next quarter needs several audits per store over time (a panel of monthly scores). The data is "
             "a single snapshot with at most one previous audit, which is not enough history. With monthly history, use gradient boosting on lagged scores."},
    {"id": "kpi_impact_estimation", "task": "causal", "runnable": False,
     "text": "Estimating how much sales or score would improve if a KPI improved (or the ROI of doing so) is a causal question. A snapshot only shows "
             "association. It needs pilot or before-and-after data with control stores (difference-in-differences) plus cost data for ROI."},
]

_DISCLAIMER = " (Generic industry practice written for this demo; validate it against your own operation.)"
PLAYBOOK = [
    {"id": "on_shelf_availability", "title": "Improving on-shelf availability", "text":
     "Typical causes: late or missed replenishment, small back rooms, forecasting errors, delivery gaps on busy days, products left in storage. "
     "Actions: set a minimum shelf-fill check at every visit, agree a replenishment cadence with the store, flag the top out-of-stock SKUs to supply, "
     "and re-check the worst stores within two weeks. Track: availability on the same SKU list at each audit." + _DISCLAIMER},
    {"id": "share_of_shelf", "title": "Improving share of shelf", "text":
     "Typical causes: competitors holding more facings, planogram resets not executed, shelf space given to other categories. "
     "Actions: confirm the agreed facings with the store manager, reclaim facings from slow-moving lines, and show the manager the sales per facing. "
     "Track: share of shelf per category, compared with similar stores." + _DISCLAIMER},
    {"id": "planogram_compliance", "title": "Improving planogram compliance", "text":
     "Typical causes: planogram changes not communicated, staff turnover, products placed where they fit, space constraints. "
     "Actions: share a photo of the target layout, walk the store manager through it at the visit, simplify layouts that cannot fit the shelf, "
     "and repeat photo checks until two audits in a row pass. Track: items placed correctly out of items audited." + _DISCLAIMER},
    {"id": "price_compliance", "title": "Improving price compliance", "text":
     "Typical causes: shelf labels not updated after price changes, missing labels, mismatched promotional tickets. "
     "Actions: send price-change lists before the effective date, check labels for the top-selling SKUs first, replace missing tickets on the spot, "
     "and agree who owns label changes in the store. Track: share of audited SKUs with the correct shelf price." + _DISCLAIMER},
    {"id": "promo_execution", "title": "Improving promotion execution", "text":
     "Typical causes: promotion materials arriving late, stock not ready for the promotion, displays not built, staff unaware of start dates. "
     "Actions: confirm stock and materials a week before launch, brief the store manager on dates and mechanics, build the display during the visit, "
     "and audit on the first day of the promotion. Track: agreed promotions executed correctly out of agreed promotions." + _DISCLAIMER},
    {"id": "extra_displays", "title": "Using extra displays", "text":
     "Extra displays tend to matter where there is space and traffic. Actions: place them on high-traffic routes, pair them with promotions, "
     "keep them stocked, and remove displays that are empty or off-strategy. Track: displays present and stocked at each audit. "
     "The data only shows association between displays and results, not that adding displays causes sales." + _DISCLAIMER},
    {"id": "visit_cadence", "title": "Visit cadence and audit quality", "text":
     "More visits help only if each visit fixes something. Actions: give high-gap stores more frequent visits and stable stores fewer, "
     "use a standard checklist so audits are comparable, and re-audit stores whose results look unusual. "
     "Track: issues found and issues closed per visit." + _DISCLAIMER},
    {"id": "store_prioritization", "title": "Prioritizing which stores to visit first", "text":
     "Rank stores by how much there is to gain and how fixable it is. Steps: start with at-risk stores that also missed their sales target, "
     "sort by the size of the sales shortfall, then group them by their weakest KPI so one rep can fix the same issue in several stores on one route. "
     "Leave stores that already hit target and pass the audit on a lighter visit schedule." + _DISCLAIMER},
    {"id": "perfect_store_missed_target", "title": "Perfect stores that still miss their sales target", "text":
     "Good execution does not guarantee sales. Possible causes: the target is set too high for the location, low footfall, range or assortment gaps "
     "that the audit does not measure, local competition, or pricing. Actions: compare the target with similar stores, check traffic and range, "
     "and review whether the audit measures what drives sales in that store. Treat these as a signal to investigate, not as proof execution does not matter." + _DISCLAIMER},
    {"id": "small_format_stores", "title": "Convenience and traditional stores", "text":
     "Small formats have little shelf and back-room space and rely on frequent small deliveries. Actions: focus on a short list of must-have SKUs, "
     "agree smaller more frequent orders, use compact displays, and keep layouts simple. Track: availability of the must-have list." + _DISCLAIMER},
    {"id": "root_cause_checklist", "title": "Root-cause checklist for weak stores", "text":
     "Before prescribing a fix, separate the causes: supply (stock not delivered), store process (staff, time, space), information (instructions not received) "
     "and measurement (audit errors). Ask why at least three times, confirm with the store manager, and fix the cause that repeats across stores first." + _DISCLAIMER},
]

_client = None
def _collection(name, items, meta_keys):
    """Builds the collection and rebuilds it automatically whenever the text in this file changes."""
    global _client
    root = Path(__file__).parent / "kb_store"
    root.mkdir(exist_ok=True)
    _client = _client or chromadb.PersistentClient(path=str(root))
    digest = hashlib.sha256("".join(i["id"] + i["text"] for i in items).encode()).hexdigest()
    stamp = root / f"{name}.hash"
    if not (stamp.exists() and stamp.read_text() == digest):
        try:
            _client.delete_collection(name)
        except Exception:
            pass
        col = _client.create_collection(name)
        col.add(ids=[i["id"] for i in items], documents=[i["text"] for i in items],
                metadatas=[{k: i[k] for k in meta_keys} for i in items])
        stamp.write_text(digest)
    return _client.get_collection(name)

def get_glossary():
    return _collection("metric_definitions", GLOSSARY, ["term"])

def get_model_cards():
    return _collection("model_cards", MODEL_CARDS, ["task", "runnable"])

def get_playbook():
    return _collection("playbook", PLAYBOOK, ["title"])

if __name__ == "__main__":
    print(f"Ready: {get_glossary().count()} definitions, {get_model_cards().count()} model cards, {get_playbook().count()} playbook entries")
