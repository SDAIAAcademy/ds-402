# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 3.3 · A rule resting on heavy weights
#
# Day 3 · Session 3 (Untrustworthy estimates) · 25 min.
#
# An advocacy group asks Fursa to "reach the unreached": offer coaching to every disabled seeker in Rumh and Nijad, every seeker unemployed for more than 36 months and every seeker over 55. Day 1 showed these groups were almost never offered coaching. You value the rule on the logs, then show with effective sample sizes, the heaviest weights and clipping why the number cannot carry a decision.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/logged.csv.gz` without office RUM-2, and `data/reference/cate_logged.csv.gz` (outcome predictions `mu0`, `mu1`). Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · The rule and its inverse propensity value (5')
# - [ ] 2 · Effective sample size and the heaviest weights (7')
# - [ ] 3 · Where the rule leaves the logs (6')
# - [ ] 4 · Clip, compare and write the diagnosis (7')
# - [ ] ✱ Stretch · The augmented value, and why it disagrees

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
tr = U.truth_logged().set_index("decision_id").loc[L["decision_id"]]
print(f"{len(L):,} decisions")

# %% [markdown]
# ## 1 · The rule and its inverse propensity value (5')
#
# ✏️ **Exercise** · Return the rule as a 0/1 array: offer to disabled seekers in Rumh or Nijad, to anyone unemployed more than 36 months and to anyone over 55; no one else.

# %%
def unreached(L):
    # BEGIN SOLUTION
    m = (((L["disability"] == 1) & L["region"].isin(["Rumh", "Nijad"])) | (L["months_unemployed"] > 36) | (L["age"] > 55))
    return m.astype(int).to_numpy()
    # END SOLUTION


try:
    rule = unreached(L)
except NotImplementedError:
    rule = np.array(U.reference("d3_3", "rule"))
U.keep("d3_3", "rule", rule.astype(int))
v, s = U.ipw_value(y, a, p, rule)
print(f"the rule offers {rule.mean():.1%} · IPW value {v:.1%} (95% interval {v - 1.96 * s:.1%} to {v + 1.96 * s:.1%}) · current practice {y.mean():.1%}")
U.check(0.05 < rule.mean() < 0.2, "the rule offers fewer places than the budget")

# %% [markdown]
# ## 2 · Effective sample size and the heaviest weights (7')
#
# Kish's effective sample size, `(sum w)^2 / sum w^2`, says how many equally weighted decisions the estimate is worth. Compute it for the rule's weights **among the seekers it would offer coaching**, where the logs rarely did.
#
# ✏️ **Exercise** · Return the number of rule-offered seekers, how many of them the logs actually offered, the effective sample size of the weights within that group, and the share of that group's total weight carried by its ten heaviest decisions.

# %%
def weight_diagnostics(y, a, p, rule):
    # BEGIN SOLUTION
    w = U.policy_weights(a, p, rule)
    g = rule == 1
    wg = w[g]
    top10 = np.sort(wg)[-10:].sum() / wg.sum()
    return int(g.sum()), int((g & (a == 1)).sum()), U.ess(wg), float(top10)
    # END SOLUTION


try:
    n_g, n_agree, ess_g, top10 = weight_diagnostics(y, a, p, rule)
except NotImplementedError:
    n_g, n_agree, ess_g, top10 = U.reference("d3_3", "diagnostics")
U.keep("d3_3", "diagnostics", [n_g, n_agree, ess_g, top10])
print(f"{n_g:,} seekers the rule would coach · the logs coached {n_agree} of them · effective sample size {ess_g:.0f} · "
      f"ten heaviest weights carry {top10:.0%} of the group's weight")
w_all = U.policy_weights(a, p, rule)
print(f"heaviest single weight {w_all.max():.0f} (probability {1 / w_all.max():.3f})")
U.check(n_agree < 0.1 * n_g, "the logs coached fewer than one in ten of the seekers the rule would coach")
U.check(top10 > 0.05, "a handful of decisions carry a visible share of the weight")

# %% [markdown]
# ## 3 · Where the rule leaves the logs (6')
#
# ✏️ **Exercise** · For each of the rule's three groups, return the number of seekers, the share offered in the logs and the minimum recorded probability.

# %%
def groups(L):
    # BEGIN SOLUTION
    parts = {"disabled, Rumh or Nijad": (L["disability"] == 1) & L["region"].isin(["Rumh", "Nijad"]),
             "over 36 months": L["months_unemployed"] > 36, "over 55": L["age"] > 55}
    return pd.DataFrame({k: {"n": int(m.sum()), "offered": float(L.loc[m, "offered"].mean()), "min_p": float(L.loc[m, "p_offer"].min())}
                         for k, m in parts.items()}).T
    # END SOLUTION


try:
    gt = groups(L)
except NotImplementedError:
    gt = U.reference("d3_3", "groups")
U.keep("d3_3", "groups", gt.round(5))
print(gt.round(3).to_string())
U.check((gt["offered"] < 0.06).all(), "all three groups were offered coaching in under 6% of decisions")

# %% [markdown]
# ## 4 · Clip, compare and write the diagnosis (7')
#
# Clipping weights at a ceiling lowers the variance and adds bias of unknown size. Compute the IPW value with weights clipped at 50 and at 20, and the true value (from the generator; real data never give you this).
#
# ✏️ **Exercise** · Return the two clipped values and a diagnosis of two or three sentences for Wijdan.

# %%
def clip_and_diagnose():
    # BEGIN SOLUTION
    w = U.policy_weights(a, p, rule)
    c50 = float(np.mean(np.minimum(w, 50) * y)); c20 = float(np.mean(np.minimum(w, 20) * y))
    text = (f"The rule would coach {n_g:,} seekers whom caseworkers almost never coached: the logs hold {n_agree} such cases, "
            f"worth about {ess_g:.0f} equally weighted decisions, and the estimate moves from {v:.1%} to {c20:.1%} when weights are clipped. "
            "The logs cannot value this rule; if the agency wants to coach these groups, it should run a small randomised pilot with them first.")
    return c50, c20, text
    # END SOLUTION


try:
    c50, c20, diagnosis = clip_and_diagnose()
except NotImplementedError:
    c50, c20, diagnosis = U.reference("d3_3", "clip")
U.keep("d3_3", "clip", [c50, c20, diagnosis])
truth = float(np.where(rule == 1, tr["p1"], tr["p0"]).mean())
print(f"IPW {v:.1%} · clipped at 50 {c50:.1%} · clipped at 20 {c20:.1%} · truth {truth:.1%}")
print(diagnosis)
U.check(abs(c20 - v) > 0.005, "clipping moves the estimate: the value depends on a few weights")

# %% [markdown]
# ## ✱ Stretch · The augmented value, and why it disagrees
#
# The augmented estimator fills in the rule-offered seekers with the outcome model `mu1`. What is it here, and what is it leaning on?

# %%
def stretch():
    # BEGIN SOLUTION
    d, ds = U.dr_value(y, a, p, rule, mu0, mu1)
    print(f"augmented {d:.1%} (se {ds:.2%}) · truth {truth:.1%}")
    print("Among the rule-offered seekers almost every value of mu1 is an extrapolation: the outcome model saw few coached seekers like them.")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. The IPW interval looks reasonable. Why is it not enough?
#    <details><summary>Answer</summary>The standard error itself is estimated from the same few heavy weights; when the heavy cases are rare, the interval is unreliable too.</details>
# 2. Is clipping a fix?
#    <details><summary>Answer</summary>It trades variance for bias of unknown size and direction. Report it as a sensitivity check, not as the estimate.</details>
# 3. What would make this rule evaluable?
#    <details><summary>Answer</summary>Data in which these seekers were coached: a small randomised pilot targeted at them, or logs from a period when the tool offered them places.</details>
#
# ## Carry forward
#
# Day 4 meets the same groups in the fairness audit: a rule may not use disability status to decide, and the data cannot value one that does.
