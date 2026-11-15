# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4", "fairlearn>=0.12"]
# ///
# %% [markdown]
# # Lab 4.2 · Attributes, audit and a constraint
#
# Day 4 · Session 2 (Fair allocation) · 25 min.
#
# Before the rule goes to Wijdan, Shuruq Al-Obaikan, Fursa's data protection officer, asks three questions: which attributes decide who gets a place, how the rule treats each region, men and women, and seekers with a disability, and what it would cost to guarantee every region its share of places. You sort the attributes, compare the rule with one allowed to use everything, audit both the rule and the forest's benefit ranking with Fairlearn, and price a regional floor.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/pilot.csv`, `data/reference/dr_pilot.csv` (doubly robust scores of the training split), `data/reference/rule_tree.json` (the reference rule of Lab 4.1), and the logs without RUM-2 with `data/reference/cate_logged.csv.gz`. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Drive or audit: sort the attributes (6')
# - [ ] 2 · Learn the rule on the allowed attributes (5')
# - [ ] 3 · Audit by region, gender and disability (7')
# - [ ] 4 · Add a regional floor and price it (7')
# - [ ] ✱ Stretch · The same audit of the true gains with Fairlearn

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ds-402.git"
    if not Path("/content/ds-402").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ds-402"], check=True)
    os.chdir("/content/ds-402/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("fursa.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import json
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
import fursa as U
print("mode:", U.MODE)

P = U.pilot()
Ptr = P[P["split"] == "train"].reset_index(drop=True)
g = U.supplied("dr_pilot.csv").set_index("seeker_id").loc[Ptr["seeker_id"], "dr_score"].to_numpy()
L = U.logged(drop_override=True)
R = U.supplied("cate_logged.csv.gz").set_index("decision_id").loc[L["decision_id"]].reset_index()
y, a, p = L["employed_6m"].to_numpy(), L["offered"].to_numpy(), L["p_offer"].to_numpy()
mu0, mu1, cate = R["mu0"].to_numpy(), R["mu1"].to_numpy(), R["cate_hat"].to_numpy()
for d in (Ptr, L):
    d["female"] = (d["gender"] == "female").astype(int)
ATTRS = ["region", "age", "gender", "disability", "education", "months_unemployed", "prior_wage_k", "digital_skills",
         "prior_jobs", "distance_km", "has_licence", "dependants", "motivation", "risk_score"]
print(f"{len(Ptr):,} training seekers · {len(L):,} logged decisions · {len(ATTRS)} attributes")

# %% [markdown]
# ## 1 · Drive or audit: sort the attributes (6')
#
# Every attribute gets one of three roles.
# - **drive**: the rule may use it to decide.
# - **audit**: the rule must not use it, but every result is checked across its groups.
# - **exclude**: not used to decide and not needed to audit.
#
# Three sources guide the choice. The Personal Data Protection Law limits processing to the purposes data were collected for (Article 10) and their content to the minimum the purpose needs (Article 11) and treats health data, which includes disability, as sensitive (Article 1). The national AI Bias Reference Guide warns of historical bias: an attribute that encodes past unequal treatment carries it into the rule. And Fursa's caseworkers must be able to explain any attribute that drives a decision.
#
# ✏️ **Exercise** · Return a dict from each attribute to `"drive"`, `"audit"` or `"exclude"`, with a one-line reason for each in `REASONS`.

# %%
def register():
    # BEGIN SOLUTION
    roles = {"region": "audit", "age": "audit", "gender": "audit", "disability": "audit",
             "months_unemployed": "drive", "digital_skills": "drive", "education": "drive", "prior_jobs": "drive",
             "has_licence": "drive", "distance_km": "drive",
             "prior_wage_k": "exclude", "dependants": "exclude", "motivation": "exclude", "risk_score": "exclude"}
    reasons = {
        "region": "a protected place of residence; checked so no region is left out",
        "age": "deciding by age would be direct discrimination; audited",
        "gender": "deciding by gender would be direct discrimination; audited",
        "disability": "health data, sensitive under the PDPL; audited, never used to decide",
        "months_unemployed": "registered fact the service exists to act on; caseworkers can explain it",
        "digital_skills": "assessed at registration for the service; explains need for coaching",
        "education": "registered fact relevant to job search",
        "prior_jobs": "registered fact relevant to job search",
        "has_licence": "registered fact that affects which jobs are reachable",
        "distance_km": "distance to the office, relevant to attending coaching",
        "prior_wage_k": "carries past pay gaps by gender and region (historical bias)",
        "dependants": "family data collected for benefits, not for coaching (purpose limitation)",
        "motivation": "a caseworker's rating: subjective, and seekers would learn to game it",
        "risk_score": "a model output built partly from age and disability; it would bring them back in",
    }
    return roles, reasons
    # END SOLUTION


try:
    ROLES, REASONS = register()
except NotImplementedError:
    ROLES, REASONS = U.reference("d4_2", "register")
U.keep("d4_2", "register", [ROLES, REASONS])
print(pd.DataFrame({"role": ROLES, "reason": REASONS}).sort_values("role").to_string())
U.check(set(ROLES) == set(ATTRS) and set(ROLES.values()) <= {"drive", "audit", "exclude"}, "every attribute has one role")
U.check(all(ROLES[k] == "audit" for k in ["region", "gender", "disability", "age"]), "region, gender, disability and age only audit the rule")
U.check(ROLES["months_unemployed"] == "drive", "months unemployed may drive the rule")
DRIVE = [k for k, v in ROLES.items() if v == "drive"]

# %% [markdown]
# **Look at** the excluded rows. Each attribute there predicts employment, and some predict who gains from coaching: excluding them can cost value. The next task measures that cost instead of guessing it.
#
# ## 2 · Learn the rule on the allowed attributes (5')
#
# The reference rule (`rule_tree.json`) was learned on six allowed attributes. Learn a second tree, under the same budget, on every numeric attribute: your drive list plus age, `female`, disability, prior wage, dependants, motivation and risk score. Value both on the logs.
#
# ✏️ **Exercise** · Return the free tree's offers on the logs. `U.budget_tree(X, g)` learns a depth-two tree under the budget.

# %%
FREE = DRIVE + ["age", "female", "disability", "prior_wage_k", "dependants", "motivation", "risk_score"]
rule = U.predict_tree(json.loads((U.REF / "rule_tree.json").read_text(encoding="utf-8")), L)


def free_rule():
    # BEGIN SOLUTION
    t, _ = U.budget_tree(Ptr[FREE], g, n_thresholds=U.pick(12, 6), iters=U.pick(14, 10))
    print(U.tree_text(t))
    return U.predict_tree(t, L)
    # END SOLUTION


try:
    free = free_rule()
except NotImplementedError:
    free = np.array(U.reference("d4_2", "free"))
U.keep("d4_2", "free", free.astype(int))
v_rule, s_rule = U.dr_value(y, a, p, rule, mu0, mu1)
v_free, s_free = U.dr_value(y, a, p, free, mu0, mu1)
print(f"allowed attributes: offers {rule.mean():.1%}, value {v_rule:.1%} (se {s_rule:.2%})")
print(f"every attribute:    offers {free.mean():.1%}, value {v_free:.1%} (se {s_free:.2%}) · the two agree on {(rule == free).mean():.1%} of seekers")
U.check(v_free - v_rule < 0.005, "using every attribute buys less than half a point")

# %% [markdown]
# **Look at** the agreement. The search was free to use age, gender, disability and the risk score; its only extra split (on the risk score, among the long-term unemployed) offers almost no one new, and the two rules agree on practically every seeker. The effect is driven by months unemployed. Here the register costs nothing measurable. That is a finding, not a rule: in another programme the excluded attributes could carry value, and the register would then have a price you report.
#
# ## 3 · Audit by region, gender and disability (7')
#
# Audit two rules: the depth-two rule, and the top fifth by the causal forest's estimated benefit (Lab 3.2's benefit rule). For each, Fairlearn's `MetricFrame` gives the selection rate (share offered) by group, and the ratio of the lowest group's rate to the highest.
#
# ✏️ **Exercise** · Return a table with one row per rule and audit attribute: the lowest group, its rate, the highest rate and the ratio.

# %%
benefit = U.top_share(cate)
AUDIT = ["region", "gender", "disability"]


def audit(rules):
    # BEGIN SOLUTION
    rows = []
    for name, r in rules.items():
        for col in AUDIT:
            mf = MetricFrame(metrics=selection_rate, y_true=y, y_pred=r, sensitive_features=L[col])
            bg = mf.by_group
            rows.append({"rule": name, "attribute": col, "lowest": str(bg.idxmin()), "lowest_rate": bg.min(),
                         "highest_rate": bg.max(), "ratio": mf.ratio()})
    return pd.DataFrame(rows)
    # END SOLUTION


try:
    aud = audit({"depth-two rule": rule, "benefit, top fifth": benefit})
except NotImplementedError:
    aud = U.reference("d4_2", "audit")
U.keep("d4_2", "audit", aud.round(5))
print(aud.round(3).to_string(index=False))
r_rule = aud[aud.rule == "depth-two rule"].set_index("attribute")
r_ben = aud[aud.rule == "benefit, top fifth"].set_index("attribute")
U.check((r_rule["ratio"] > 0.8).all(), "the depth-two rule offers every group within 20% of the highest rate")
U.check(r_ben.loc["region", "ratio"] < 0.5 and r_ben.loc["region", "lowest"] == "Nijad", "the benefit ranking offers Nijad less than half the top region's rate")
print(L.assign(over_15_months=L["months_unemployed"] > 15, over_60_km=L["distance_km"] > 60).groupby("region")[["over_15_months", "over_60_km"]].mean().round(3).to_string())

# %% [markdown]
# **Look at** Nijad under the benefit ranking. Half of Nijad's seekers have been unemployed for more than 15 months, and three in four live more than 60 km from an office: both lower coaching's effect (Day 2), so a ranking by estimated benefit offers them few places. The ranking never saw the region; it reached it through months unemployed and distance. The depth-two rule happens to treat regions evenly because its band, 11 to 15 months, cuts across them. An audit is how you find out which case you are in.
#
# ## 4 · Add a regional floor and price it (7')
#
# The depth-two rule offers about 16% of seekers; the budget allows 20%. Spend the remaining places in two ways, both by the forest's estimated benefit among seekers the rule does not already offer:
# - **national**: the top-ranked seekers anywhere, until 20% overall;
# - **regional floor**: in each region, until 20% of that region's seekers.
#
# The price of the floor is the national rule's value minus the floor's. Estimate it with a paired interval (the difference of the two rules' per-decision augmented scores, as in Lab 3.2's stretch).
#
# ✏️ **Exercise** · Return both rules and the price `(estimate, se)`.

# %%
def fill(rule, per_region):
    # BEGIN SOLUTION
    r = rule.copy()
    reg = L["region"].to_numpy()
    groups = U.REGIONS if per_region else [None]
    for gname in groups:
        m = (reg == gname) if per_region else np.ones(len(r), bool)
        need = int(round(U.BUDGET * m.sum())) - r[m].sum()
        idx = np.flatnonzero(m & (r == 0))
        r[idx[np.argsort(-cate[idx], kind="stable")[:max(need, 0)]]] = 1
    return r
    # END SOLUTION


def psi(r):
    w = U.policy_weights(a, p, r)
    return np.where(r == 1, mu1, mu0) + w * (y - np.where(a == 1, mu1, mu0))


try:
    national, floor = fill(rule, False), fill(rule, True)
except NotImplementedError:
    national, floor = (np.array(x) for x in U.reference("d4_2", "fills"))
U.keep("d4_2", "fills", [national.astype(int), floor.astype(int)])
d = psi(national) - psi(floor)
price, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
tr = U.truth_logged().set_index("decision_id").loc[L["decision_id"]]
true_price = float(np.mean((tr["p1"] - tr["p0"]).to_numpy() * (national - floor)))
for name, r in [("national", national), ("regional floor", floor)]:
    rates = pd.Series(r).groupby(L["region"].to_numpy()).mean()
    print(f"{name:15s} offers {r.mean():.1%} · Nijad {rates['Nijad']:.1%} · value {psi(r).mean():.1%}")
print(f"price of the floor: {U.pp(price)} (95% interval {U.pp(price - 1.96 * se)} to {U.pp(price + 1.96 * se)}) · true price {U.pp(true_price)}")
print(f"in jobs: {price * len(L):+.0f} over these {len(L):,} seekers (true {true_price * len(L):+.0f})")
U.check(abs(national.mean() - U.BUDGET) < 0.005 and abs(floor.mean() - U.BUDGET) < 0.005, "both rules spend the whole budget")
U.check(price - 1.96 * se < 0 < price + 1.96 * se, "the floor's price is not distinguishable from zero")

# %% [markdown]
# **Look at** the price. A floor that gives every region its fifth of places costs nothing you can measure here: the last places go to seekers whose gains are similar wherever they live. Report it that way: "the floor costs between X and Y jobs", not "the floor is free". Had the rule been the benefit ranking, the same floor would move many places into Nijad, where coaching helps less, and its price would be real; the decision would then be Wijdan's, with the number in front of her.

# %% [markdown]
# ## ✱ Stretch · The same audit of the true gains with Fairlearn
#
# Selection rates say who is offered, not who gains. With the generator's true effects (never available in real data), use `MetricFrame` with a custom metric: the mean true effect among the seekers the rule offers, by region and by disability.

# %%
def stretch():
    # BEGIN SOLUTION
    tau = (tr["p1"] - tr["p0"]).to_numpy()
    gain = lambda y_true, y_pred: float(np.mean(y_true[y_pred == 1])) if (y_pred == 1).any() else np.nan
    for col in ["region", "disability"]:
        mf = MetricFrame(metrics=gain, y_true=tau, y_pred=rule, sensitive_features=L[col])
        print(f"true gain per offered seeker by {col}:", {k: U.pp(v) for k, v in mf.by_group.items()})
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is disability an audit attribute and not a driver, even if disabled seekers gain more?
#    <details><summary>Answer</summary>It is sensitive health data collected for another purpose, and deciding by it needs a legal basis and a policy decision, not a model's finding. Auditing shows whether the rule serves disabled seekers; a decision to prioritise them belongs to the agency.</details>
# 2. The benefit ranking never used region. How did it under-select Nijad?
#    <details><summary>Answer</summary>Through months unemployed, which differs by region: a proxy. Leaving an attribute out does not stop a rule from depending on it.</details>
# 3. Why price a constraint with a paired interval rather than two separate intervals?
#    <details><summary>Answer</summary>The two rules agree on most seekers, so their values are strongly correlated; the paired difference cancels the shared noise.</details>
#
# ## Carry forward
#
# Lab 4.3 sizes a trial of the rule against current practice. Lab 4.4 puts this audit on one page.
