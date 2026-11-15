# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 4.1 · A depth-two rule against the alternatives
#
# Day 4 · Session 1 (Policy learning under a budget) · 25 min.
#
# Wijdan's second question: next year there is one coaching place for every five job seekers; who should get them? You learn a rule from the randomised pilot that a caseworker can read: an exhaustive tree of depth two on the attributes allowed to drive it, held within the budget. Then you value it on the logs against treat-all, risk targeting and current practice.
#
# **AI policy.** AI-off: this is the day's first estimator. Assistants closed; the slides, the documentation and the trainer are open.
#
# **Reference state.** `data/pilot.csv` (training split) and `data/logged.csv.gz` without office RUM-2, with `data/reference/cate_logged.csv.gz` for the outcome predictions `mu0`, `mu1` on the logs. Nothing from another lab is needed. The attributes allowed to drive the rule are fixed here; Lab 4.2 discusses the choice.
#
# **Tasks**
# - [ ] 1 · Doubly robust scores on the training split (6')
# - [ ] 2 · An exhaustive depth-two tree under the budget (8')
# - [ ] 3 · Value against treat-all, risk and current practice (7')
# - [ ] 4 · Read the tree (4')
# - [ ] ✱ Stretch · Depth one against depth two

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
import fursa as U
print("mode:", U.MODE)

ALLOWED = ["months_unemployed", "digital_skills", "education", "prior_jobs", "has_licence", "distance_km"]
P = U.pilot()
Ptr = P[P["split"] == "train"].reset_index(drop=True)
L = U.logged(drop_override=True)
R = U.supplied("cate_logged.csv.gz").set_index("decision_id").loc[L["decision_id"]].reset_index()
y, a, p = L["employed_6m"].to_numpy(), L["offered"].to_numpy(), L["p_offer"].to_numpy()
mu0, mu1 = R["mu0"].to_numpy(), R["mu1"].to_numpy()
print(f"pilot training split {len(Ptr):,} seekers · logs {len(L):,} decisions · budget {U.BUDGET:.0%} of seekers")

# %% [markdown]
# ## 1 · Doubly robust scores on the training split (6')
#
# A tree needs a gain for each seeker. The individual effect is never seen, but the doubly robust score is unbiased for it on average:
#
# `score = mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)`, with `e = 0.5` in the pilot.
#
# The outcome predictions are cross-fitted, so no seeker's own outcome predicts itself.
#
# ✏️ **Exercise** · Return the scores for the training split.

# %%
m0_tr, m1_tr = U.outcome_models_crossfit(Ptr["employed_6m"], Ptr["coached"], U.design(Ptr), model=U.pilot_classifier(3), seed=3)


def dr_scores(y, t, e, mu0, mu1):
    # BEGIN SOLUTION
    y, t = np.asarray(y, float), np.asarray(t, float)
    return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    # END SOLUTION


try:
    g = dr_scores(Ptr["employed_6m"], Ptr["coached"], 0.5, m0_tr, m1_tr)
except NotImplementedError:
    g = U.supplied("dr_pilot.csv")["dr_score"].to_numpy()
    U.check(not U.STRICT, "the solution run used its own scores")
print(f"mean score {U.pp(g.mean())} (the average effect on the training split) · standard deviation {U.pp(g.std())}")
ref = U.supplied("dr_pilot.csv")["dr_score"].to_numpy()
U.check(np.corrcoef(g, ref)[0, 1] > 0.999, "your scores match the supplied ones")
U.check(0.05 < g.mean() < 0.13, "the scores average to a plausible effect")

# %% [markdown]
# **Look at** the spread: a single seeker's score has a standard deviation near 90 points, ten times the average effect. Each score is noise around the effect; only sums over many seekers, which is what a tree uses, are informative.
#
# ## 2 · An exhaustive depth-two tree under the budget (8')
#
# `U.policy_tree(X, g, cost)` searches every pair of splits and returns the tree that maximises the sum of `g - cost` over the seekers it offers. With `cost = 0` it offers every group with a positive average score, far more than the budget. A cost per place stands for the budget: raise it until the tree offers at most 20%.
#
# ✏️ **Exercise** · Find the smallest cost (by bisection, about 12 steps between 0 and 0.5) at which the tree's share of offers on the training split is at most `U.BUDGET`. Return the tree and the cost.

# %%
X_tr = Ptr[ALLOWED]
NT = 12  # thresholds per attribute; the search takes about a minute on a laptop


def tree_under_budget(X, g, budget=U.BUDGET, steps=12):
    # BEGIN SOLUTION
    lo, hi, best = 0.0, 0.5, None
    for _ in range(steps):
        cost = (lo + hi) / 2
        t = U.policy_tree(X, g, cost=cost, n_thresholds=NT)
        if U.predict_tree(t, X).mean() <= budget:
            best, hi = (t, cost), cost
        else:
            lo = cost
    return best
    # END SOLUTION


try:
    tree, cost = tree_under_budget(X_tr, g)
except NotImplementedError:
    tree = json.loads((U.REF / "rule_tree.json").read_text(encoding="utf-8"))
    cost = tree["cost"]
    U.check(not U.STRICT, "the solution run learned its own tree")
print(U.tree_text(tree))
rule = U.predict_tree(tree, L)
print(f"cost per place {U.pp(cost)} · offers {U.predict_tree(tree, X_tr).mean():.1%} of the training split, {rule.mean():.1%} of the logs")
U.check(rule.mean() <= U.BUDGET + 0.01, "the rule stays within one place per five seekers")
ref_rule = U.predict_tree(json.loads((U.REF / "rule_tree.json").read_text(encoding="utf-8")), L)
U.check((rule == ref_rule).mean() > 0.95, "your rule agrees with the reference rule on at least 95% of logged seekers")

# %% [markdown]
# ## 3 · Value against treat-all, risk and current practice (7')
#
# Value each rule on the logs with the augmented estimator of Lab 3.2 (`U.dr_value`). Current practice is the logs' own mean outcome. Treat-all coaches everyone and breaks the budget five times over: it is there to show what places are worth, not as a candidate.
#
# ✏️ **Exercise** · Return a table with one row per rule: `share` offered, `value`, `lo`, `hi`, and `jobs_per_100_places`: the extra seekers employed per 100 places compared with offering no one, `100 * (value - value_none) / share`.

# %%
def compare(rule):
    # BEGIN SOLUTION
    rules = {"depth-two rule": rule, "risk, top fifth": U.top_share(L["risk_score"].to_numpy()),
             "treat-all": np.ones(len(L), int), "no one": np.zeros(len(L), int)}
    rows = {}
    for k, r in rules.items():
        v, s = U.dr_value(y, a, p, r, mu0, mu1)
        rows[k] = {"share": r.mean(), "value": v, "lo": v - 1.96 * s, "hi": v + 1.96 * s}
    s = y.std(ddof=1) / np.sqrt(len(y))
    rows["current practice"] = {"share": a.mean(), "value": y.mean(), "lo": y.mean() - 1.96 * s, "hi": y.mean() + 1.96 * s}
    t = pd.DataFrame(rows).T
    none = t.loc["no one", "value"]
    t["jobs_per_100_places"] = np.where(t["share"] > 0, 100 * (t["value"] - none) / t["share"].replace(0, np.nan), np.nan)
    return t.sort_values("value", ascending=False)
    # END SOLUTION


try:
    tab = compare(rule)
except NotImplementedError:
    tab = U.reference("d4_1", "table")
U.keep("d4_1", "table", tab.round(5))
tr = U.truth_logged().set_index("decision_id").loc[L["decision_id"]]
truth = {"depth-two rule": rule, "risk, top fifth": U.top_share(L["risk_score"].to_numpy()), "treat-all": np.ones(len(L), int),
         "no one": np.zeros(len(L), int), "current practice": a}
tab["truth"] = [float(np.where(truth[k] == 1, tr["p1"], tr["p0"]).mean()) for k in tab.index]
show = tab.copy()
for c in ["share", "value", "lo", "hi", "truth"]:
    show[c] = (100 * show[c]).round(1)
print(show.round(1).to_string(), "\n(% of seekers or % employed within six months; truth from the generator)")
U.check(tab.loc["depth-two rule", "lo"] > tab.loc["current practice", "hi"], "the rule beats current practice beyond both intervals")
U.check(tab.loc["depth-two rule", "value"] > tab.loc["risk, top fifth", "hi"], "the rule beats risk targeting")
U.check(tab.loc["depth-two rule", "jobs_per_100_places"] > 1.5 * tab.loc["treat-all", "jobs_per_100_places"],
        "per place, the rule finds far more jobs than coaching everyone")

# %% [markdown]
# **Look at** two columns. Treat-all has the highest value, because it uses five times the places; per place it is the least effective. And current practice gains less per place than the rule, although it spends about as many: the caseworkers offer coaching mostly to the hardest cases (Day 1), where it helps less.
#
# ## 4 · Read the tree (4')
#
# ✏️ **Exercise** · Write the rule as one sentence a caseworker could apply at the counter, and list the attributes it uses.

# %%
def describe(tree):
    # BEGIN SOLUTION
    sentence = "Offer a coaching place to job seekers who have been unemployed for more than 10 and at most 15 months."
    used = sorted({tree["feature"]} | {n["feature"] for n in (tree["left"], tree["right"]) if "feature" in n})
    return sentence, used
    # END SOLUTION


try:
    sentence, used = describe(tree)
except NotImplementedError:
    sentence, used = U.reference("d4_1", "describe")
U.keep("d4_1", "describe", [sentence, used])
print(sentence, "\nattributes used:", used)
U.check(set(used) <= set(ALLOWED), "the rule uses only allowed attributes")
U.check("months" in sentence.lower(), "the sentence names the attribute the rule turns on")

# %% [markdown]
# **Look at** what the exhaustive search chose: every split on the same attribute, which together make one band of months. Day 2 found months unemployed the strongest driver of the effect, peaking around a year; Lab 3.2's hand-written duration rule was a guess at this band. The tree offers 16% rather than 20%: a band of whole months cannot hit the budget exactly. Lab 4.2 spends the remaining places.

# %% [markdown]
# ## ✱ Stretch · Depth one against depth two
#
# Learn the best single split under the same budget: `U.policy_tree` is depth two, so search one split yourself (for each allowed attribute and threshold, offer the side whose score sum minus cost is positive). Value it on the logs.

# %%
def stretch():
    # BEGIN SOLUTION
    def stump(cost):
        best = (-np.inf, None)
        r = g - cost
        for c in ALLOWED:
            x = X_tr[c].to_numpy(float)
            for th in U._thresholds(x, NT):
                left = x <= th
                v = max(r[left].sum(), 0) + max(r[~left].sum(), 0)
                if v > best[0]:
                    best = (v, {"feature": c, "threshold": float(th), "left": {"leaf": int(r[left].sum() > 0)}, "right": {"leaf": int(r[~left].sum() > 0)}})
        return best[1]
    lo, hi, t1 = 0.0, 0.5, None
    for _ in range(12):
        mid = (lo + hi) / 2
        t = stump(mid)
        if U.predict_tree(t, X_tr).mean() <= U.BUDGET:
            t1, hi = t, mid
        else:
            lo = mid
    r1 = U.predict_tree(t1, L)
    v1, s1 = U.dr_value(y, a, p, r1, mu0, mu1)
    print(U.tree_text(t1))
    print(f"depth one offers {r1.mean():.1%}, value {v1:.1%} (se {s1:.2%}) · depth two {tab.loc['depth-two rule', 'value']:.1%}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does the tree learn from doubly robust scores rather than from the causal forest's estimates?
#    <details><summary>Answer</summary>The scores are unbiased for each seeker's effect on average, so sums over a leaf estimate the leaf's gain without the forest's shrinkage (Lab 2.3 found it too timid).</details>
# 2. What does the cost per place stand for?
#    <details><summary>Answer</summary>The shadow price of a place: the gain a seeker must exceed to deserve one when places are limited. Raising it until the rule fits the budget is the Lagrangian form of the budget constraint.</details>
# 3. Why is the tree learned on the pilot but valued on the logs?
#    <details><summary>Answer</summary>The pilot is randomised, so its scores are clean for learning; the logs are a separate, larger sample of the population the rule will meet, with recorded probabilities, so the value is not measured on the data that chose the rule.</details>
#
# ## Carry forward
#
# Lab 4.2 asks which attributes may drive this rule and audits it. Lab 4.3 sizes the trial that would confirm it; Lab 4.4 summarises it from `data/reference/rule_tree.json`.
