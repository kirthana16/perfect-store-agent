"""The evaluation questions.
  task      the TASK label the agent should give
  gold      (operational only) a hand-written SQL query whose result the agent's SQL must match
  playbook  (advisory only) a playbook id the agent should retrieve
Shared by eval.py (live agent) and tests/ (offline checks)."""
LAST_MONTH = ("strftime('%Y-%m', audit_date) = strftime('%Y-%m', (SELECT MAX(audit_date) FROM stores), 'start of month', '-1 month')")

def case(q, task, gold=None, playbook=None):
    return {"q": q, "task": task, "gold": gold, "playbook": playbook}

TESTS = [
    # operational, with gold SQL
    case("How many stores are perfect stores?", "operational", "SELECT COUNT(*) FROM stores WHERE perfect_store = 1"),
    case("Which region has the lowest average on-shelf availability?", "operational",
         "SELECT region FROM stores GROUP BY region ORDER BY AVG(on_shelf_availability) ASC LIMIT 1"),
    case("What percentage of stores hit their sales target?", "operational", "SELECT ROUND(100.0 * AVG(hit_sales_target), 2) FROM stores"),
    case("Which 5 stores have the largest sales shortfall against target?", "operational", "SELECT store_id FROM stores ORDER BY sales_gap ASC LIMIT 5"),
    case("Which field rep has the most at-risk stores?", "operational",
         "SELECT field_rep_id FROM stores WHERE perfect_store_score < 70 GROUP BY field_rep_id ORDER BY COUNT(*) DESC LIMIT 1"),
    case("Which channel has the highest perfect store rate?", "operational",
         "SELECT channel FROM stores GROUP BY channel ORDER BY AVG(perfect_store) DESC LIMIT 1"),
    case("What is the most common weakest KPI among stores that missed their sales target?", "operational",
         "SELECT weakest_kpi FROM stores WHERE hit_sales_target = 0 GROUP BY weakest_kpi ORDER BY COUNT(*) DESC LIMIT 1"),
    case("How many perfect stores still missed their sales target?", "operational",
         "SELECT COUNT(*) FROM stores WHERE perfect_store = 1 AND hit_sales_target = 0"),
    case("How many stores were audited last month?", "operational", f"SELECT COUNT(*) FROM stores WHERE {LAST_MONTH}"),
    case("How many priority stores do we have?", "operational",
         "SELECT COUNT(*) FROM stores WHERE perfect_store_score < 70 AND hit_sales_target = 0"),
    case("What is the total sales gap across all stores?", "operational", "SELECT SUM(sales_gap) FROM stores"),
    case("Which retailer chain has the highest average perfect store score?", "operational",
         "SELECT retailer_chain FROM stores GROUP BY retailer_chain ORDER BY AVG(perfect_store_score) DESC LIMIT 1"),
    # predictive
    case("Which KPIs best predict whether a store hits its sales target?", "target_prediction"),
    case("Can we predict which stores will miss their sales target?", "target_prediction"),
    case("What natural groups of stores do we have based on their KPIs?", "segmentation"),
    case("Segment our stores so we can tailor field visits.", "segmentation"),
    case("Which stores have unusual KPI patterns that we should re-audit?", "anomaly"),
    case("Are there stores whose audit results look suspicious?", "anomaly"),
    # advisory: must run SQL for evidence AND retrieve the right playbook entry
    case("What should we do to improve on-shelf availability in our weakest region?", "advisory", playbook="on_shelf_availability"),
    case("How can we improve promo execution in the stores that missed target?", "advisory", playbook="promo_execution"),
    case("Which stores should field reps visit first, and what should they fix?", "advisory", playbook="store_prioritization"),
    case("Why do some perfect stores still miss their sales target and what can we do?", "advisory", playbook="perfect_store_missed_target"),
    # unsupported
    case("Which stores will decline in score next quarter?", "unsupported"),
    case("What is our profit margin?", "unsupported"),
    case("How much does each extra display cost us?", "unsupported"),
    case("What is the ROI of improving planogram compliance?", "unsupported"),
    case("Which competitor is gaining share in our stores?", "unsupported"),
]
