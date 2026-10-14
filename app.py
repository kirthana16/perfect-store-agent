import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Perfect Store Agent", page_icon="🏬", layout="wide")

try:                                                    # on Streamlit Cloud the key lives in Secrets
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

from agent import ask_agent                             # imported after the key is set
from tools import _load_stores

STEP_LABELS = {"lookup_metric_definition": "Looking up the metric definition (RAG)", "get_data_schema": "Reading the schema",
               "query_stores": "Running a SQL query", "profile_stores": "Profiling the data",
               "retrieve_model_cards": "Searching model cards (RAG)", "retrieve_playbook": "Searching the action playbook (RAG)",
               "run_store_target_benchmark": "Benchmarking sales-target models", "run_store_segmentation_benchmark": "Benchmarking store segmentation",
               "find_unusual_stores": "Scanning for unusual stores"}
EXAMPLES = ["Which region has the lowest on-shelf availability?",
            "Which field reps have the most at-risk stores?",
            "Which KPIs best predict whether a store hits its sales target?",
            "Which stores have unusual KPI patterns we should re-audit?",
            "Which stores should reps visit first, and what should they fix?",
            "Which stores will decline in score next quarter?"]

st.title("🏬 Perfect Store Agent")
st.caption("Ask what is true (SQL), what predicts it (ML), what to do about it (playbook), or something the data cannot answer. "
           "The agent shows its work: definitions retrieved, SQL run, models benchmarked, guidance used.")

with st.sidebar:
    st.header("How it works")
    st.markdown("**Operational:** definition (RAG) → SQL → answer\n\n**Predictive:** model cards (RAG) → benchmarks or anomaly scan\n\n"
                "**Advisory:** SQL evidence → playbook (RAG) → actions\n\n**Unsupported:** declined, with what would be needed\n\n"
                "**Data:** 2,000-store Perfect Store audit. Rep, chain, store age, audit date, sales amounts and previous score are synthetic. "
                "Playbook guidance is generic and written for this demo.")
    with st.expander("Preview the data"):
        st.dataframe(_load_stores().head(8), use_container_width=True)

if "questions" not in st.session_state:
    st.session_state.questions = EXAMPLES[0]

st.subheader("1. Pick or write your questions")
for row in (EXAMPLES[:3], EXAMPLES[3:]):
    for col, ex in zip(st.columns(3), row):
        if col.button(ex, use_container_width=True):
            st.session_state.questions = ex
st.text_area("One question per line (max 4 per run)", key="questions", height=110)

def show_sql(steps):
    ok = [s for s in steps if s["tool"] == "query_stores" and "rows" in s["output"]]
    if not ok:
        return
    out = ok[-1]["output"]
    st.markdown("**SQL used**")
    st.code(out["sql"], language="sql")
    df = pd.DataFrame(out["rows"], columns=out["columns"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    if out["truncated"]:
        st.caption("Showing the first 50 rows.")
    if len(df) > 1 and len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
        st.bar_chart(df.set_index(df.columns[0]))

def show_ml(steps):
    for s in steps:
        o = s["output"]
        if not isinstance(o, dict):
            continue
        if s["tool"].startswith("run_") and "results" in o:
            st.markdown(f"**Benchmark evidence** · {o['metric']}")
            df = pd.DataFrame({"model": list(o["results"]), "score": list(o["results"].values())}).set_index("model")
            c1, c2 = st.columns([1, 2])
            c1.dataframe(df, use_container_width=True)
            c2.bar_chart(df)
            if o.get("feature_importance"):
                st.markdown("**Top drivers** · drop in PR-AUC when the feature is shuffled (association, not causation)")
                st.bar_chart(pd.Series(o["feature_importance"], name="importance"))
            if o.get("cluster_profiles"):
                st.markdown(f"**Store segments** · K-Means, k = {o['best_kmeans_k']}")
                st.dataframe(pd.DataFrame(o["cluster_profiles"]), use_container_width=True, hide_index=True)
        if s["tool"] == "find_unusual_stores" and "unusual_stores" in o:
            st.markdown("**Most unusual stores** · candidates for a re-audit, not proven errors")
            st.dataframe(pd.DataFrame(o["unusual_stores"]), use_container_width=True, hide_index=True)

def show_knowledge(steps):
    """Make the RAG visible: which definitions, model cards and playbook entries were retrieved."""
    found = {"lookup_metric_definition": "Definitions", "retrieve_model_cards": "Model cards", "retrieve_playbook": "Playbook"}
    lines = []
    for tool, label in found.items():
        ids = sorted({i["id"] for s in steps if s["tool"] == tool and isinstance(s["output"], list) for i in s["output"]})
        if ids:
            lines.append(f"**{label}:** " + ", ".join(f"`{i}`" for i in ids))
    if lines:
        st.caption("Knowledge retrieved · " + "  ·  ".join(lines))

st.subheader("2. Ask the agent")
if st.button("Run", type="primary"):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("No ANTHROPIC_API_KEY found. Set it as an environment variable or in Streamlit secrets.")
        st.stop()
    for q in [x.strip() for x in st.session_state.questions.splitlines() if x.strip()][:4]:
        with st.container(border=True):
            st.markdown(f"### {q}")
            with st.status("Agent working...", expanded=True) as status:
                def on_step(name, args):
                    st.write(f"▸ {STEP_LABELS.get(name, name)}")
                try:
                    out = ask_agent(q, on_step=on_step)
                    status.update(label="Done", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.error(f"{type(e).__name__}: {e}")
                    continue
            st.markdown(out["answer"].replace("\n", "  \n"))
            show_knowledge(out["steps"])
            show_sql(out["steps"])
            show_ml(out["steps"])
            with st.expander("Full agent trace (every tool call and raw output)"):
                for s in out["steps"]:
                    st.markdown(f"`{s['tool']}` {s['input']}")
                    st.json(s["output"], expanded=False)
