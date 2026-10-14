"""The agent loop: Claude decides which tool to call, our code runs it, Claude reads the result."""
import json, os
import anthropic
from tools import run_tool

MODEL = os.environ.get("MODEL", "claude-sonnet-5-5")   # try "claude-haiku-5-5" for a cheaper deployed demo
MAX_STEPS = 12                                          # hard cap so the agent can never loop forever
client = anthropic.Anthropic()

SYSTEM = """You are a Perfect Store analytics assistant for a retail execution team. The data is one table, stores: 2,000 stores, one row each,
with five KPIs (on_shelf_availability, share_of_shelf, planogram_compliance, price_compliance, promo_execution), a perfect_store_score,
sales vs target, region, channel, field rep and more. It is a snapshot, not a time series.

You handle five kinds of question:
A) OPERATIONAL: what is true in the data? (counts, rankings, averages, comparisons). Answer by running SQL.
B) PREDICTIVE: which stores will hit target and why, how stores group together, which stores look unusual. Use the ML tools.
C) ADVISORY: what should we do about weak stores or a weak KPI? Combine data evidence with the playbook.
D) UNSUPPORTED: needs data we do not have (costs, profit, ROI, competitors, stock levels, monthly history, predicting future declines, causal impact).
   Say so, and explain what data or work would be needed. Use retrieve_model_cards if it helps explain why.

OPERATIONAL:
1. lookup_metric_definition for every business term (perfect store, at-risk, priority stores, last month, ...). Follow the definitions exactly.
2. get_data_schema once, to see the columns and data_as_of.
3. query_stores. If it returns an error, fix the SQL and retry (at most 3 attempts). Never do arithmetic in your head that SQL could do.
4. State the definition used and the dates covered. Relative dates are relative to data_as_of, not today.

PREDICTIVE:
1. profile_stores. 2. retrieve_model_cards to see candidates, requirements and leaky columns.
3. Call the matching tool: run_store_target_benchmark (who hits target, which KPIs drive it), run_store_segmentation_benchmark (store groups),
   find_unusual_stores (re-audit candidates).
4. Recommend the best-scoring model; if the baseline wins or the gain is tiny, say so. Feature importance shows association, not causation.
   For segments, describe each in one short line and report the silhouette honestly: below 0.25 means the groups overlap and are only rough.
   For unusual stores, say they are candidates for a re-audit, not proven errors.

ADVISORY:
1. lookup_metric_definition for the terms involved, and get_data_schema if needed.
2. query_stores to find the evidence: which stores, region, channel or KPI are weakest, with numbers.
3. retrieve_playbook for the weak KPI or situation (you may call it more than once).
4. Recommend 2-4 actions from the playbook, each tagged with its playbook id in square brackets, tied to the evidence you found.
   Do not invent impact numbers. The playbook is generic guidance, not proven for this data.

RULES: never invent numbers; quote only numbers from tool results. If an answer depends on synthetic columns (see the schema notes), say so in CAVEATS.
Write for a field operations manager, not a data scientist. Be concise.

Your final answer MUST start with a TASK line and follow one of these formats exactly.

Operational:
TASK: operational
ANSWER: <direct answer with the key numbers>
DEFINITION USED: <metric definition and the dates or filters applied>
CAVEATS: <1-2 short bullets, or "none">

Predictive:
TASK: <target_prediction | segmentation | anomaly>
RECOMMENDED MODEL: <name>
WHY: <2-3 plain-English sentences>
EVIDENCE: <key scores or findings from the tool, including the baseline where there is one>
RISKS: <1-3 short bullets>

Advisory:
TASK: advisory
FINDINGS: <the data evidence, with numbers>
RECOMMENDED ACTIONS: <numbered list, each action ending with [playbook id]>
CAVEATS: <1-2 short bullets>

Unsupported:
TASK: unsupported
REASON: <why it cannot be answered>
WHAT WOULD BE NEEDED: <data or work required>"""

NO_ARGS = {"type": "object", "properties": {}}
TOOLS = [
    {"name": "lookup_metric_definition", "description": "Look up the business definition of a metric or term (perfect store, at-risk stores, priority stores, last month, OSA...). Search before writing SQL.",
     "input_schema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}},
    {"name": "get_data_schema", "description": "Get the stores table columns, notes, regions, channels and the data_as_of date (the latest audit date).", "input_schema": NO_ARGS},
    {"name": "query_stores", "description": "Run ONE read-only SQLite SELECT against the stores table. Returns up to 50 rows. On a SQL error, fix the query and retry.",
     "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "profile_stores", "description": "Profile the stores table for ML: rows, KPI ranges, hit rate, perfect-store rate, missing values.", "input_schema": NO_ARGS},
    {"name": "retrieve_model_cards", "description": "Search the knowledge base of ML model descriptions, requirements and weaknesses.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "e.g. 'predict stores that miss sales target'"}}, "required": ["query"]}},
    {"name": "retrieve_playbook", "description": "Search the field-execution playbook for causes, actions and tracking advice for a KPI or situation.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string", "description": "e.g. 'low on-shelf availability' or 'which stores to visit first'"}}, "required": ["query"]}},
    {"name": "run_store_target_benchmark", "description": "Compare classifiers that predict whether a store hits its sales target from KPIs and store attributes. Also returns top drivers.", "input_schema": NO_ARGS},
    {"name": "run_store_segmentation_benchmark", "description": "Compare clustering models that group stores by KPI profile. Also returns a profile of each segment.", "input_schema": NO_ARGS},
    {"name": "find_unusual_stores", "description": "Find stores with unusual KPI combinations (re-audit candidates) using Isolation Forest.", "input_schema": NO_ARGS},
]

def ask_agent(question, on_step=None):
    """Returns {"answer": str, "steps": [{"tool", "input", "output"}, ...]}"""
    messages = [{"role": "user", "content": question}]
    steps = []
    for _ in range(MAX_STEPS):
        resp = client.messages.create(model=MODEL, max_tokens=2000, system=SYSTEM, tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            return {"answer": "".join(b.text for b in resp.content if b.type == "text"), "steps": steps}
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                if on_step:
                    on_step(b.name, b.input)
                out = run_tool(b.name, b.input)
                steps.append({"tool": b.name, "input": b.input, "output": out})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    return {"answer": "TASK: unsupported\nREASON: The agent hit its step limit.", "steps": steps}

if __name__ == "__main__":
    for q in ["Which region has the lowest on-shelf availability?",
              "Which KPIs best predict whether a store hits its sales target?",
              "What should we do about stores with poor planogram compliance?"]:
        r = ask_agent(q, on_step=lambda n, a: print("->", n, a))
        print(r["answer"], "\n")
