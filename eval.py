"""Measures the agent on 27 questions. Run: python eval.py   (uses API credits: about 150 calls)
 - task accuracy:      did it pick the right kind of question?
 - answer accuracy:    (operational) do the numbers in its SQL results match a hand-written gold query?
 - retrieval accuracy: (advisory) did it run SQL for evidence AND retrieve the right playbook entry?
Put the three numbers in your README, plus what failed and what you fixed."""
import re
import pandas as pd
from agent import ask_agent
from eval_cases import TESTS
from eval_helpers import agent_cells, answer_matches, gold_cells, ran_sql, retrieved_ids

rows = []
for t in TESTS:
    answer_ok = retrieval_ok = None
    try:
        out = ask_agent(t["q"])
        m = re.search(r"TASK:\s*([a-z_]+)", out["answer"], re.I)
        got = m.group(1).lower() if m else "no-format"
        if t["gold"]:
            answer_ok = answer_matches(gold_cells(t["gold"]), agent_cells(out["steps"]))
        if t["playbook"]:
            retrieval_ok = ran_sql(out["steps"]) and t["playbook"] in retrieved_ids(out["steps"], "retrieve_playbook")
    except Exception as e:
        got = f"error: {type(e).__name__}"
        answer_ok = False if t["gold"] else None
        retrieval_ok = False if t["playbook"] else None
    rows.append({"question": t["q"], "expected": t["task"], "got": got, "task_ok": got == t["task"],
                 "answer_ok": answer_ok, "retrieval_ok": retrieval_ok})
    mark = lambda v: {True: "✓", False: "✗", None: "-"}[v]
    print("✓" if got == t["task"] else "✗", "task |", mark(answer_ok), "answer |", mark(retrieval_ok), "retrieval |", t["q"], "->", got)

df = pd.DataFrame(rows)
df.to_csv("eval_results.csv", index=False)
ops, adv = df[df["answer_ok"].notna()], df[df["retrieval_ok"].notna()]
print(f"\nTask accuracy:      {df['task_ok'].sum()}/{len(df)}")
print(f"Answer accuracy:    {int(ops['answer_ok'].sum())}/{len(ops)}  (operational, vs gold SQL)")
print(f"Retrieval accuracy: {int(adv['retrieval_ok'].sum())}/{len(adv)}  (advisory: SQL evidence + right playbook entry)")
