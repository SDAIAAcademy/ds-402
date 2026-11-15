# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5"]
# ///
# %% [markdown]
# # Pre-work · Potential outcomes and propensity scores
#
# About three hours, before Day 1 and its entry quiz.
#
# DS-402 assumes two ideas from DS-302 and DS-303 and uses them from the first hour: every job seeker has two possible outcomes, of which we see one; and when the decision was not random, the probability of being treated (the propensity score) is what lets us compare like with like. This notebook refreshes both on Fursa Employment Agency's data, the course's case, and ends with the entry quiz. Each exercise has a stored answer, so the notebook runs to the end even if you skip one.
#
# **AI policy.** Work without an assistant: the entry quiz asks the same things.
#
# **Reference state.** Nothing but this notebook and `fursa.py`; it reads `data/pilot.csv`, `data/logged.csv.gz` and the generator's truth files.
#
# **Tasks**
# - [ ] 1 · Two outcomes per seeker, one observed (25')
# - [ ] 2 · The randomised pilot: a difference in means (25')
# - [ ] 3 · The logs: a naive difference and its bias (25')
# - [ ] 4 · A propensity score and its overlap (40')
# - [ ] 5 · Weighting and balance (40')
# - [ ] 6 · The entry quiz (25')
# - [ ] ✱ Stretch · How well does the propensity model rank?

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

P = U.pilot()
TP = U.truth_pilot()
L = U.logged(drop_override=True)   # all decisions in both modes: the stored answers are per decision
TL = U.truth_for(L["decision_id"])
print(f"pilot {len(P):,} seekers · logs {len(L):,} decisions")

# %% [markdown]
# ## 1 · Two outcomes per seeker, one observed (25')
#
# The generator knows each pilot seeker's chance of employment within six months without coaching (`p0`) and with it (`p1`). Real data never give you these: here they let you see what an estimator is aiming at. A seeker's effect is `p1 - p0`; we observe only the outcome under the arm she was given.
#
# ✏️ **Exercise** · Return the average effect over all pilot seekers (the ATE) and over the coached seekers (the ATT), from the truth.

# %%
def true_effects():
    # BEGIN SOLUTION
    tau = (TP["p1"] - TP["p0"]).to_numpy()
    return float(tau.mean()), float(tau[P["coached"].to_numpy() == 1].mean())
    # END SOLUTION


try:
    ate, att = true_effects()
except NotImplementedError:
    ate, att = U.reference("p0", "true")
U.keep("p0", "true", [ate, att])
print(f"true ATE {U.pp(ate)} · true ATT {U.pp(att)}")
print(TP.head(6).assign(effect=lambda d: d.p1 - d.p0, coached=P["coached"].head(6), employed=P["employed_6m"].head(6)).round(3).to_string(index=False))
U.check(abs(ate - att) < 0.01, "in a lottery the coached seekers are like everyone else: ATT is close to ATE")

# %% [markdown]
# **Look at** the table: one row per seeker, two chances, one observed outcome. The missing outcome is why every causal estimate is a comparison of groups.
#
# ## 2 · The randomised pilot: a difference in means (25')
#
# ✏️ **Exercise** · Return the difference in employment rates between coached and uncoached pilot seekers, and its standard error.

# %%
def diff_in_means():
    # BEGIN SOLUTION
    y, t = P["employed_6m"].to_numpy(), P["coached"].to_numpy()
    a, b = y[t == 1], y[t == 0]
    return float(a.mean() - b.mean()), float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    # END SOLUTION


try:
    d, se = diff_in_means()
except NotImplementedError:
    d, se = U.reference("p0", "dim")
U.keep("p0", "dim", [d, se])
print(f"difference in means {U.pp(d)} (95% interval {U.pp(d - 1.96 * se)} to {U.pp(d + 1.96 * se)}) · true ATE {U.pp(ate)}")
U.check(d - 1.96 * se < ate < d + 1.96 * se, "the interval covers the true effect")

# %% [markdown]
# ## 3 · The logs: a naive difference and its bias (25')
#
# In the logs caseworkers chose whom to offer coaching. Compare offered and not-offered seekers the same way.
#
# ✏️ **Exercise** · Return the naive difference in the logs and the true ATT of the offered seekers.

# %%
def naive():
    # BEGIN SOLUTION
    y, a = L["employed_6m"].to_numpy(), L["offered"].to_numpy()
    tau = (TL["p1"] - TL["p0"]).to_numpy()
    return float(y[a == 1].mean() - y[a == 0].mean()), float(tau[a == 1].mean())
    # END SOLUTION


try:
    nd, att_l = naive()
except NotImplementedError:
    nd, att_l = U.reference("p0", "naive")
U.keep("p0", "naive", [nd, att_l])
print(f"naive difference {U.pp(nd)} · true ATT {U.pp(att_l)} · gap {U.pp(nd - att_l)}")
U.check(abs(nd - att_l) > 0.01, "the naive difference misses the effect: who was offered differs from who was not")

# %% [markdown]
# **Look at** the sign of the gap. The naive difference = the ATT + selection bias: the difference between offered and not-offered seekers had nobody been coached. Here it overstates the effect: caseworkers' choices, including their sense of a seeker's motivation, picked seekers who would more often have found work anyway.
#
# ## 4 · A propensity score and its overlap (40')
#
# The propensity score is the probability of being offered given the covariates. Fit a logistic regression on the 14 named covariates (`U.design(L)`), then look at the overlap: the distribution of the score among offered and not-offered seekers.
#
# ✏️ **Exercise** · Return the fitted propensity for every decision.

# %%
X = U.design(L)


def propensity():
    # BEGIN SOLUTION
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, L["offered"])
    return m.predict_proba(X)[:, 1]
    # END SOLUTION


try:
    e = propensity()
except NotImplementedError:
    e = np.array(U.reference("p0", "e"))
U.keep("p0", "e", np.round(e, 5))
a = L["offered"].to_numpy()
bins = pd.cut(e, [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0])
print(pd.crosstab(bins, a, rownames=["fitted propensity"], colnames=["offered"]).to_string())
grp = ((L["disability"] == 1) & L["region"].isin(["Rumh", "Nijad"])).to_numpy()
print(f"disabled seekers in Rumh or Nijad: offered {a[grp].mean():.1%} · mean fitted propensity {e[grp].mean():.1%}")
U.check(a[grp].mean() < 0.05, "one group was almost never offered coaching: overlap is poor there")
U.check(e[grp].mean() > 2 * a[grp].mean(), "the linear model does not see it: its propensities for that group are far too high")

# %% [markdown]
# **Look at** the last two lines. Disabled seekers in Rumh and Nijad were almost never offered coaching, yet the logistic model, which adds up the covariates' effects, gives them ordinary propensities: the gap comes from a combination of disability and region that an additive model cannot represent. Overlap problems hide in combinations; Day 1 maps them with a flexible model.
#
# ## 5 · Weighting and balance (40')
#
# Inverse propensity weights make the offered and not-offered groups look like the whole population: offered seekers get `1/e`, others `1/(1-e)`. Clip `e` to [0.01, 0.99] first.
#
# ✏️ **Exercise** · Return the weighted ATE estimate, and the largest absolute standardised mean difference over the 14 covariates before and after weighting.

# %%
def weighting(e):
    # BEGIN SOLUTION
    y = L["employed_6m"].to_numpy()
    ec = np.clip(e, 0.01, 0.99)
    w = np.where(a == 1, 1 / ec, 1 / (1 - ec))
    est = np.sum(w * a * y) / np.sum(w * a) - np.sum(w * (1 - a) * y) / np.sum(w * (1 - a))
    def smd(weights):
        out = []
        for c in X.columns:
            x = X[c].to_numpy()
            m1 = np.average(x[a == 1], weights=weights[a == 1]); m0 = np.average(x[a == 0], weights=weights[a == 0])
            s = np.sqrt((x[a == 1].var() + x[a == 0].var()) / 2) or 1.0
            out.append(abs(m1 - m0) / s)
        return max(out)
    return float(est), smd(np.ones(len(a))), smd(w)
    # END SOLUTION


try:
    ipw, smd0, smd1 = weighting(e)
except NotImplementedError:
    ipw, smd0, smd1 = U.reference("p0", "ipw")
U.keep("p0", "ipw", [ipw, smd0, smd1])
true_ate_l = float((TL["p1"] - TL["p0"]).mean())
print(f"weighted ATE {U.pp(ipw)} · true ATE in the logs {U.pp(true_ate_l)} · naive {U.pp(nd)}")
print(f"largest standardised mean difference: {smd0:.2f} before, {smd1:.2f} after weighting")
U.check(smd1 < smd0, "weighting improves balance on the named covariates")

# %% [markdown]
# **Look at** what is left. Balance on the 14 named covariates improves, but the estimate is still off: the caseworkers also used 46 administrative columns, which this model leaves out and which act in nonlinear ways. Day 1 brings them in with double machine learning.
#
# ## 6 · The entry quiz (25')
#
# Answer without an assistant, then open the answers.
#
# 1. A seeker was coached and found work. What is her effect of coaching?
#    <details><summary>Answer</summary>Unknown: we do not see whether she would have found work without coaching. Only group averages can be estimated.</details>
# 2. Why does the pilot's difference in means estimate the ATE without adjustment?
#    <details><summary>Answer</summary>The lottery makes coached and uncoached seekers alike on average in everything, measured or not, so the selection bias is zero in expectation.</details>
# 3. In another agency, caseworkers offered coaching only to seekers with the poorest prospects. Which way does that push the naive difference?
#    <details><summary>Answer</summary>Down: the offered would have done worse anyway, so the naive difference understates the effect (a negative selection bias). Fursa's logs show the opposite sign: the direction depends on how the choice was made.</details>
# 4. What does a propensity score of 0.01 for a group tell you?
#    <details><summary>Answer</summary>That the group was almost never treated: there is no overlap, and any effect estimate for it rests on extrapolation.</details>
# 5. After weighting, the standardised mean differences of all covariates are below 0.1. Is the estimate unbiased?
#    <details><summary>Answer</summary>Only if no confounder is missing from the covariates and the propensity model is right: balance on what was measured says nothing about what was not.</details>
# 6. What is the difference between the ATE and the ATT?
#    <details><summary>Answer</summary>The ATE averages the effect over everyone; the ATT over those who were treated. They differ when the treated are chosen for reasons related to the effect.</details>

# %% [markdown]
# ## ✱ Stretch · How well does the propensity model rank?
#
# Compute the area under the ROC curve of the propensity model, in-sample and with five-fold cross-validation. A high value means good prediction of who was offered, which is bad news for overlap.

# %%
def stretch():
    # BEGIN SOLUTION
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = cross_val_predict(m, X, a, cv=5, method="predict_proba")[:, 1]
    print(f"AUC in-sample {roc_auc_score(a, e):.3f} · cross-validated {roc_auc_score(a, cv):.3f}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Carry forward
#
# Day 1 starts from these ideas: the estimand of a decision about places, overlap first, then double machine learning with all 60 confounders.
