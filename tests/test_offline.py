"""Offline tests: no API key, internet or ChromaDB needed. They guard the parts that can break silently:
SQL safety, the gold queries, the answer matcher, the ML tools and the wiring between agent, tools and knowledge base.
Run:  pytest -q"""
import ast
import re
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools
from eval_cases import TESTS
from eval_helpers import answer_matches, gold_cells

LEAKY = {"perfect_store_score", "perfect_store", "sales_vs_target_pct", "monthly_sales_actual", "sales_gap",
         "score_band", "gap_to_perfect", "execution_outcome", "hit_sales_target"}
SIMULATED = {"retailer_chain", "field_rep_id", "store_age_years", "audit_date", "monthly_sales_target",
             "prev_audit_score", "score_change", "score_trend"}

# ---------------------------------------------------------------- SQL safety
@pytest.mark.parametrize("sql", ["DROP TABLE stores", "SELECT 1; DELETE FROM stores", "UPDATE stores SET region = 'x'",
                                 "PRAGMA table_info(stores)", "WITH x AS (SELECT 1) INSERT INTO stores (store_id) VALUES ('z')"])
def test_sql_guardrails_block_writes(sql):
    assert "error" in tools.query_stores(sql)

def test_sql_row_cap():
    out = tools.query_stores("SELECT * FROM stores")
    assert len(out["rows"]) == 50 and out["truncated"]

def test_sql_errors_are_returned_not_raised():
    assert "SQL error" in tools.query_stores("SELECT * FROM nope")["error"]

def test_the_database_is_really_read_only():
    import sqlite3
    con = tools._connect()
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM stores")
    con.close()

def test_schema_and_profile():
    s = tools.get_data_schema()
    assert s["rows"] == 2000 and s["data_as_of"] and {"North", "South", "East", "West"} <= set(s["regions"])
    assert tools.profile_stores()["rows"] == 2000

# ---------------------------------------------------------------- evaluation set
GOLD = [(t["q"], t["gold"]) for t in TESTS if t["gold"]]

@pytest.mark.parametrize("question,sql", GOLD, ids=[q[:45] for q, _ in GOLD])
def test_gold_query_runs_and_returns_data(question, sql):
    assert len(gold_cells(sql)) > 0

def test_eval_set_covers_every_task():
    assert {t["task"] for t in TESTS} == {"operational", "target_prediction", "segmentation", "anomaly", "advisory", "unsupported"}

def test_answer_matcher():
    assert answer_matches([13190, 12789], [12789, 13190, 1])
    assert not answer_matches([13190], [13191])                  # ids must match exactly
    assert answer_matches([1000.0], [1003.0]) and not answer_matches([1000.0], [1010.0])
    assert answer_matches(["West"], ["west"])

# ---------------------------------------------------------------- ML tools
def test_store_target_benchmark_beats_baseline_and_has_no_leaky_drivers():
    r = tools.run_store_target_benchmark()
    best = max(v for k, v in r["results"].items() if "baseline" not in k)
    assert best > r["results"]["baseline (always predict average rate)"]
    assert r["feature_importance"] and not LEAKY & set(r["feature_importance"])

def test_features_exclude_leaky_and_synthetic_columns():
    assert not set(tools.STORE_FEATURES) & (LEAKY | SIMULATED)

def test_segmentation_profiles_match_k():
    r = tools.run_store_segmentation_benchmark()
    assert len(r["cluster_profiles"]) == r["best_kmeans_k"] and sum(p["stores"] for p in r["cluster_profiles"]) == r["n_stores"]

def test_unusual_stores_are_ranked_and_explained():
    r = tools.find_unusual_stores(5)
    scores = [s["anomaly_score"] for s in r["unusual_stores"]]
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)
    assert all(s["most_unusual_kpi"] in tools.STORE_KPIS for s in r["unusual_stores"])

def test_raw_file_is_enriched_automatically(tmp_path):
    import pandas as pd
    from prepare_data import build_database, load_stores
    raw = pd.read_csv(ROOT / "data" / "perfect_store_enriched.csv").iloc[:, :16]
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    assert len(load_stores(raw_path).columns) == 33
    assert build_database(raw_path, tmp_path / "t.db") == len(raw)

# ---------------------------------------------------------------- wiring
def _literal(name):
    for node in ast.parse((ROOT / "kb.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)

@pytest.mark.parametrize("name", ["GLOSSARY", "MODEL_CARDS"])
def test_knowledge_entries_are_unique_and_filled(name):
    items = _literal(name)
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)) and all(i["text"].strip() for i in items)

PLAYBOOK_IDS = re.findall(r'\{"id": "(\w+)", "title"', (ROOT / "kb.py").read_text())

def test_playbook_has_unique_ids():
    assert len(PLAYBOOK_IDS) >= 8 and len(PLAYBOOK_IDS) == len(set(PLAYBOOK_IDS))

def test_every_expected_playbook_id_exists():
    expected = {t["playbook"] for t in TESTS if t["playbook"]}
    assert expected <= set(PLAYBOOK_IDS)

def test_every_agent_tool_is_implemented():
    declared = set(re.findall(r'\{"name": "(\w+)"', (ROOT / "agent.py").read_text()))
    assert declared == set(tools.TOOL_FUNCTIONS)
