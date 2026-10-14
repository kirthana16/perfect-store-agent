"""Functions the agent can call. The LLM decides WHICH to run; this code computes the numbers."""
import re
import sqlite3
import time
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prepare_data import DB_PATH, build_database

MAX_ROWS = 50
_BLOCKED = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum)\b", re.I)
STORE_KPIS = ["on_shelf_availability", "share_of_shelf", "planogram_compliance", "price_compliance", "promo_execution"]
# ML features use ORIGINAL columns only. Leaky columns (score, perfect_store, sales_vs_target_pct, sales amounts)
# and synthetic columns are excluded on purpose.
STORE_FEATURES = STORE_KPIS + ["extra_displays", "visits_per_month", "sku_count_listed", "store_size_sqm"]

def _connect():
    if not DB_PATH.exists():
        build_database()                                      # builds from data/perfect_store_enriched.csv
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)   # read-only at the database level

def _load_stores():
    con = _connect()
    try:
        return pd.read_sql("SELECT * FROM stores", con)
    finally:
        con.close()

# ---------------------------------------------------------------- retrieval (RAG over three knowledge sources)
def lookup_metric_definition(term: str, n: int = 3):
    from kb import get_glossary
    res = get_glossary().query(query_texts=[term], n_results=n)
    return [{"id": i, "term": m["term"], "definition": d} for i, m, d in zip(res["ids"][0], res["metadatas"][0], res["documents"][0])]

def retrieve_model_cards(query: str, n: int = 4):
    from kb import get_model_cards
    res = get_model_cards().query(query_texts=[query], n_results=n)
    return [{"id": i, "task": m["task"], "runnable_by_tools": m["runnable"], "card": d}
            for i, m, d in zip(res["ids"][0], res["metadatas"][0], res["documents"][0])]

def retrieve_playbook(query: str, n: int = 3):
    from kb import get_playbook
    res = get_playbook().query(query_texts=[query], n_results=n)
    return [{"id": i, "title": m["title"], "guidance": d} for i, m, d in zip(res["ids"][0], res["metadatas"][0], res["documents"][0])]

# ---------------------------------------------------------------- data access
def get_data_schema():
    """Table, columns, date range and notes. The agent reads this before writing SQL."""
    con = _connect()
    try:
        cols = [f"{r[1]} {r[2]}" for r in con.execute("PRAGMA table_info(stores)")]
        first, last = con.execute("SELECT MIN(audit_date), MAX(audit_date) FROM stores").fetchone()
        distinct = lambda c: [r[0] for r in con.execute(f"SELECT DISTINCT {c} FROM stores ORDER BY 1")]
        return {"table": "stores", "rows": con.execute("SELECT COUNT(*) FROM stores").fetchone()[0], "columns": cols,
                "first_audit_date": first, "data_as_of": last,
                "regions": distinct("region"), "channels": distinct("channel"), "retailer_chains": distinct("retailer_chain"),
                "notes": [
                    "One row per store: a snapshot, not a time series. KPI columns are percentages (0-100).",
                    "perfect_store = 1 when perfect_store_score >= 80. hit_sales_target = 1 when sales_vs_target_pct >= 100.",
                    "Sales are monthly_sales_actual vs monthly_sales_target in currency-neutral units.",
                    "retailer_chain, field_rep_id, store_age_years, audit_date, monthly_sales_target/actual, sales_gap, prev_audit_score, "
                    "score_change and score_trend are synthetic demo columns. Mention this in CAVEATS when an answer depends on them.",
                    "audit_date is text 'YYYY-MM-DD'. Relative dates (last month, latest) are relative to data_as_of, not today."]}
    finally:
        con.close()

def query_stores(sql: str):
    """Run ONE read-only SELECT. Guardrails: SELECT/WITH only, single statement, read-only connection,
    10-second time limit, 50-row cap."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return {"error": "Only one SQL statement is allowed."}
    if not re.match(r"^(select|with)\b", s, re.I) or _BLOCKED.search(s):
        return {"error": "Only read-only SELECT queries are allowed."}
    con = _connect()
    deadline = time.time() + 10
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
    try:
        cur = con.execute(s)
        rows = cur.fetchmany(MAX_ROWS + 1)
        clean = [[round(v, 2) if isinstance(v, float) else v for v in r] for r in rows[:MAX_ROWS]]
        return {"sql": s, "columns": [d[0] for d in cur.description], "rows": clean, "truncated": len(rows) > MAX_ROWS}
    except sqlite3.Error as e:
        return {"error": f"SQL error: {e}"}
    finally:
        con.close()

def profile_stores():
    df = _load_stores()
    return {"rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()},
            "missing_pct": (df.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict(),
            "kpi_summary": df[STORE_KPIS].describe().round(1).loc[["min", "mean", "max"]].to_dict(),
            "hit_target_rate_pct": round(df["hit_sales_target"].mean() * 100, 1),
            "perfect_store_rate_pct": round(df["perfect_store"].mean() * 100, 1),
            "audit_date_range": [df["audit_date"].min(), df["audit_date"].max()],
            "note": "One row per store (a snapshot, not a time series). Only original columns are used as ML features."}

# ---------------------------------------------------------------- ML tools
def run_store_target_benchmark():
    """Predict hit_sales_target for stores from execution KPIs and store attributes."""
    df = _load_stores()
    X = pd.get_dummies(df[STORE_FEATURES + ["channel", "region"]], columns=["channel", "region"]).astype(float)
    y = df["hit_sales_target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
              "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=0, n_jobs=-1),
              "gradient_boosting": GradientBoostingClassifier(random_state=0)}
    results = {"baseline (always predict average rate)": round(float(average_precision_score(yte, np.zeros(len(yte)))), 4)}
    roc, fitted = {}, {}
    for name, m in models.items():
        fitted[name] = m.fit(Xtr, ytr)
        p = fitted[name].predict_proba(Xte)[:, 1]
        results[name] = round(float(average_precision_score(yte, p)), 4)
        roc[name] = round(float(roc_auc_score(yte, p)), 4)
    best = max(fitted, key=lambda n: results[n])
    imp = permutation_importance(fitted[best], Xte, yte, scoring="average_precision", n_repeats=5, random_state=0, n_jobs=-1)
    top = sorted(zip(X.columns, imp.importances_mean), key=lambda kv: -kv[1])[:6]
    return {"task": "target_prediction", "metric": "PR-AUC (higher is better)", "higher_is_better": True,
            "results": dict(sorted(results.items(), key=lambda kv: -kv[1])), "roc_auc_extra": roc,
            "hit_rate_test_pct": round(float(yte.mean()) * 100, 1), "n_train": len(Xtr), "n_test": len(Xte),
            "best_model": best, "feature_importance": {k: round(float(v), 4) for k, v in top},
            "feature_importance_note": f"Permutation importance for {best} on the test set: the drop in PR-AUC when that feature is shuffled. Shows association, not causation.",
            "excluded_columns": "perfect_store_score, perfect_store, sales_vs_target_pct, sales amounts (they leak the answer or are built from the KPIs) and all synthetic columns"}

def run_store_segmentation_benchmark():
    """Group stores by their five execution KPIs. Returns scores and a profile of each K-Means segment."""
    df = _load_stores()
    X = StandardScaler().fit_transform(df[STORE_KPIS])
    results, km_scores = {}, {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        km_scores[k] = float(silhouette_score(X, km.labels_))
        results[f"kmeans_k{k}"] = round(km_scores[k], 4)
        results[f"hierarchical_k{k}"] = round(float(silhouette_score(X, AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X).labels_)), 4)
    best_k = max(km_scores, key=km_scores.get)
    df["segment"] = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X).labels_ + 1
    prof = df.groupby("segment").agg(stores=("store_id", "count"), hit_target_pct=("hit_sales_target", lambda s: s.mean() * 100),
                                     perfect_store_pct=("perfect_store", lambda s: s.mean() * 100),
                                     **{c: (c, "mean") for c in STORE_KPIS}).round(1).reset_index()
    return {"task": "segmentation", "metric": "silhouette score (higher is better, above 0.5 is strong, below 0.25 is weak)",
            "higher_is_better": True, "results": dict(sorted(results.items(), key=lambda kv: -kv[1])),
            "best_kmeans_k": best_k, "cluster_profiles": prof.to_dict("records"), "n_stores": len(df)}

def find_unusual_stores(top_n: int = 10):
    """Isolation Forest on the five KPIs. No labels exist, so this flags stores worth re-auditing."""
    df = _load_stores()
    Z = StandardScaler().fit_transform(df[STORE_KPIS])
    score = -IsolationForest(n_estimators=300, random_state=0).fit(Z).score_samples(Z)      # higher = more unusual
    out = []
    for i in np.argsort(-score)[:top_n]:
        j = int(np.argmax(np.abs(Z[i])))
        out.append({"store_id": df.at[i, "store_id"], "region": df.at[i, "region"], "channel": df.at[i, "channel"],
                    "anomaly_score": round(float(score[i]), 3), "most_unusual_kpi": STORE_KPIS[j],
                    "kpi_value": round(float(df.at[i, STORE_KPIS[j]]), 1), "kpi_z_score": round(float(Z[i, j]), 2)})
    return {"task": "anomaly", "metric": "Isolation Forest anomaly score (higher = more unusual)",
            "unusual_stores": out, "n_stores": len(df),
            "note": "There are no labels, so this cannot say which stores are errors. It flags stores whose KPI mix is rare, as candidates for a re-audit."}

# ---------------------------------------------------------------- dispatcher
TOOL_FUNCTIONS = {"lookup_metric_definition": lookup_metric_definition, "retrieve_model_cards": retrieve_model_cards,
                  "retrieve_playbook": retrieve_playbook, "get_data_schema": get_data_schema, "query_stores": query_stores,
                  "profile_stores": profile_stores, "run_store_target_benchmark": run_store_target_benchmark,
                  "run_store_segmentation_benchmark": run_store_segmentation_benchmark, "find_unusual_stores": find_unusual_stores}

def run_tool(name, args):
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:                                    # return the error so the agent can react to it
        return {"error": f"{type(e).__name__}: {e}"}
