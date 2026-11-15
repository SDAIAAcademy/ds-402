# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 3.2 · Three rules offline
#
# Day 3 · Session 2 (Weighted and augmented estimators) · 25 min.
#
# Wijdan wants three candidate rules valued before any is tried: places to the top fifth by risk score, to the top fifth by estimated benefit, and a simple rule a caseworker can apply by hand. Each would offer about one place per five seekers. You value them on the logs with inverse propensity weighting, its self-normalised version and the augmented (doubly robust) estimator, with intervals, against current practice.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/logged.csv.gz` without office RUM-2 (Lab 3.1), and `data/reference/cate_logged.csv.gz`: for every decision the causal forest's estimated benefit (`cate_hat`, fitted on the pilot) and cross-fitted outcome predictions `mu0`, `mu1`. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Write the three rules (5')
# - [ ] 2 · Inverse propensity values with intervals (6')
# - [ ] 3 · Self-normalised and augmented values (8')
# - [ ] 4 · One value table and a ranking (6')
# - [ ] ✱ Stretch · An interval for the difference between two rules

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

import numpy as np
import pandas as pd
import fursa as U
print("mode:", U.MODE)

L = U.logged(drop_override=True)
R = U.supplied("cate_logged.csv.gz").set_index("decision_id").loc[L["decision_id"]].reset_index()
y, a, p = L["employed_6m"].to_numpy(), L["offered"].to_numpy(), L["p_offer"].to_numpy()
mu0, mu1 = R["mu0"].to_numpy(), R["mu1"].to_numpy()
print(f"{len(L):,} decisions (RUM-2 excluded) · current practice {y.mean():.1%} employed")

# %% [markdown]
# ## 1 · Write the three rules (5')
#
# A rule is a decision, 0 or 1, for every seeker.
# - **Risk**: the top fifth by `risk_score`.
# - **Benefit**: the top fifth by `cate_hat`.
# - **Duration**: offer when months unemployed is between 10 and 15, inclusive.
#
# ✏️ **Exercise** · Return a dict from rule name to its 0/1 array. `U.top_share(score, 0.2)` does a top fifth.

# %%
def rules(L, R):
    # BEGIN SOLUTION
    return {"risk": U.top_share(L["risk_score"].to_numpy(), 0.2),
            "benefit": U.top_share(R["cate_hat"].to_numpy(), 0.2),
            "duration": L["months_unemployed"].between(10, 15).astype(int).to_numpy()}
    # END SOLUTION


try:
    RULES = rules(L, R)
except NotImplementedError:
    RULES = {k: np.array(v) for k, v in U.reference("d3_2", "rules").items()}
U.keep("d3_2", "rules", {k: v.astype(int) for k, v in RULES.items()})
for k, v in RULES.items():
    print(f"{k:9s} offers {v.mean():.1%}")
U.check(all(0.15 < v.mean() < 0.25 for v in RULES.values()), "each rule offers about one place in five")

# %% [markdown]
# ## 2 · Inverse propensity values with intervals (6')
#
# A decision counts when the logs did what the rule would do; it is weighted by one over the probability of what the logs did. `U.ipw_value(y, a, p, rule)` returns the value and its standard error; write your own here.
#
# ✏️ **Exercise** · Return `(value, se)` for one rule.

# %%
def ipw(y, a, p, rule):
    # BEGIN SOLUTION
    p_a = np.where(a == 1, p, 1 - p)
    v = (a == rule) / p_a * y
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
    # END SOLUTION


try:
    ipw_vals = {k: ipw(y, a, p, r) for k, r in RULES.items()}
except NotImplementedError:
    ipw_vals = {k: U.ipw_value(y, a, p, r) for k, r in RULES.items()}
    U.check(not U.STRICT, "the solution run used its own estimator")
for k, (v, s) in ipw_vals.items():
    print(f"{k:9s} IPW {v:.1%} (95% interval {v - 1.96 * s:.1%} to {v + 1.96 * s:.1%})")
U.check(all(abs(ipw_vals[k][0] - U.ipw_value(y, a, p, r)[0]) < 1e-9 for k, r in RULES.items()), "your IPW matches the course's")

# %% [markdown]
# ## 3 · Self-normalised and augmented values (8')
#
# **Self-normalised** IPW divides by the sum of the weights instead of n: a little bias, much less variance. **Augmented** (doubly robust): start from the outcome model's prediction under the rule, `mu_rule = rule * mu1 + (1 - rule) * mu0`, and add the weighted residual of the decisions the logs and the rule agree on.
#
# ✏️ **Exercise** · Return `(value, se)` of the augmented estimator for one rule; the self-normalised value is `U.snipw_value`.

# %%
def augmented(y, a, p, rule, mu0, mu1):
    # BEGIN SOLUTION
    p_a = np.where(a == 1, p, 1 - p)
    w = (a == rule) / p_a
    psi = np.where(rule == 1, mu1, mu0) + w * (y - np.where(a == 1, mu1, mu0))
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))
    # END SOLUTION


try:
    dr_vals = {k: augmented(y, a, p, r, mu0, mu1) for k, r in RULES.items()}
except NotImplementedError:
    dr_vals = {k: U.dr_value(y, a, p, r, mu0, mu1) for k, r in RULES.items()}
    U.check(not U.STRICT, "the solution run used its own estimator")
sn_vals = {k: U.snipw_value(y, a, p, r) for k, r in RULES.items()}
for k in RULES:
    print(f"{k:9s} SNIPW {sn_vals[k]:.1%} · augmented {dr_vals[k][0]:.1%} (se {dr_vals[k][1]:.2%}, IPW se {ipw_vals[k][1]:.2%})")
U.check(all(dr_vals[k][1] < ipw_vals[k][1] for k in RULES), "the augmented estimator is more precise than IPW for every rule")

# %% [markdown]
# ## 4 · One value table and a ranking (6')
#
# ✏️ **Exercise** · Return one row per rule plus `current practice`: `ipw`, `snipw`, `augmented`, `lo`, `hi` (the augmented 95% interval) and `gain` (augmented minus current practice), sorted by `augmented`.

# %%
def value_table():
    # BEGIN SOLUTION
    rows = []
    for k in RULES:
        v, s = dr_vals[k]
        rows.append({"rule": k, "ipw": ipw_vals[k][0], "snipw": sn_vals[k], "augmented": v, "lo": v - 1.96 * s, "hi": v + 1.96 * s})
    cur = y.mean(); s = y.std(ddof=1) / np.sqrt(len(y))
    rows.append({"rule": "current practice", "ipw": cur, "snipw": cur, "augmented": cur, "lo": cur - 1.96 * s, "hi": cur + 1.96 * s})
    t = pd.DataFrame(rows).set_index("rule")
    t["gain"] = t["augmented"] - cur
    return t.sort_values("augmented", ascending=False)
    # END SOLUTION


try:
    tab = value_table()
except NotImplementedError:
    tab = U.reference("d3_2", "table")
U.keep("d3_2", "table", tab.round(5))
tr = U.truth_logged().set_index("decision_id").loc[L["decision_id"]]
tab["truth"] = [float(np.where(RULES[k] == 1, tr["p1"], tr["p0"]).mean()) if k in RULES else float(np.where(a == 1, tr["p1"], tr["p0"]).mean()) for k in tab.index]
print((tab * 100).round(1).to_string(), "\n(% employed within six months; truth from the generator)")
U.check(tab.loc["benefit", "lo"] > tab.loc["current practice", "augmented"] and tab.loc["duration", "lo"] > tab.loc["current practice", "augmented"],
        "benefit targeting and the duration rule both beat current practice")
U.check(tab.loc["risk", "augmented"] < tab.loc["benefit", "augmented"], "risk targeting is worth less than benefit targeting")
U.check(all(tab.loc[k, "lo"] <= tab.loc[k, "truth"] <= tab.loc[k, "hi"] for k in RULES), "every augmented interval covers the true value")

# %% [markdown]
# **Look at** two rows. The risk rule is no better than current practice: caseworkers already offer coaching mostly to high-risk seekers. And the simple duration rule matches or beats the causal forest's ranking: Lab 2.3 showed the forest is too timid, and the effect's strongest driver is months unemployed. A rule a caseworker can apply by hand is a serious candidate, which Day 4 learns on purpose.

# %% [markdown]
# ## ✱ Stretch · An interval for the difference between two rules
#
# The difference between the augmented values of two rules on the same decisions has a paired standard error: the standard deviation of the difference of their per-decision scores. Compute it for benefit minus duration.

# %%
def stretch():
    # BEGIN SOLUTION
    def psi(rule):
        w = (a == rule) / np.where(a == 1, p, 1 - p)
        return np.where(rule == 1, mu1, mu0) + w * (y - np.where(a == 1, mu1, mu0))
    d = psi(RULES["benefit"]) - psi(RULES["duration"])
    print(f"benefit minus duration: {U.pp(d.mean())} (95% interval {U.pp(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)))} "
          f"to {U.pp(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)))})")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does IPW use only the decisions where the logs agree with the rule?
#    <details><summary>Answer</summary>Only for them do we see the outcome the rule would have produced; the weights make those decisions stand in for the ones that disagree.</details>
# 2. What does the augmented estimator need to be consistent?
#    <details><summary>Answer</summary>Either correct propensities (here recorded, and checked in Lab 3.1) or a correct outcome model. With both close, it is also the most precise.</details>
# 3. Why is current practice's interval the narrowest?
#    <details><summary>Answer</summary>Every decision follows current practice, so no decision is discarded or reweighted.</details>
#
# ## Carry forward
#
# Lab 3.3 values a rule that leaves the support of the logs. Day 4 learns a rule and values it the same way.
