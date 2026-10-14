"""Helpers that check answers. We compare the numbers in the agent's SQL results with the numbers from a gold query,
and check what the agent retrieved."""
from tools import query_stores

def gold_cells(sql):
    out = query_stores(sql)
    if "error" in out:
        raise RuntimeError(f"Gold SQL failed: {out['error']}")
    return [c for row in out["rows"] for c in row]

def agent_cells(steps):
    cells = []
    for s in steps:
        o = s["output"]
        if s["tool"] == "query_stores" and isinstance(o, dict) and "rows" in o:
            cells += [c for row in o["rows"] for c in row]
    return cells

def retrieved_ids(steps, tool):
    ids = []
    for s in steps:
        if s["tool"] == tool and isinstance(s["output"], list):
            ids += [item["id"] for item in s["output"]]
    return ids

def ran_sql(steps):
    return any(s["tool"] == "query_stores" and isinstance(s["output"], dict) and "rows" in s["output"] for s in steps)

def _match(g, c):
    if isinstance(g, str) or isinstance(c, str):
        return str(g).strip().casefold() == str(c).strip().casefold()
    if g is None or c is None:
        return g is c
    if isinstance(g, int) and isinstance(c, int):
        return g == c                                       # ids and counts must match exactly
    return abs(g - c) <= max(0.01, 0.005 * abs(g))          # decimals: within 0.5%

def answer_matches(gold, found):
    """True if every gold value appears somewhere in the agent's query results."""
    return all(any(_match(g, c) for c in found) for g in gold)
