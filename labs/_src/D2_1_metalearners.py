# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 2.1 · S-, T- and X-learners on the pilot
#
# Day 2 · Session 1 (Meta-learners) · 25 min.
#
# The pilot was a lottery, so the propensity of coaching is 0.5 for everyone and the only hard part is the outcome model. You turn a gradient-boosted classifier into three estimators of each seeker's effect: the S-learner (one model with coaching as a feature), the T-learner (one model per arm) and the X-learner (imputed effects, then a model of them). You fit on the training split and score the held-out split against the true effects the generator wrote.
#
# **AI policy.** AI-off: this is the day's first estimator. Assistants closed; the slides, the documentation and the trainer are open.
#
# **Reference state.** `data/pilot.csv` (its `split` column fixes the training and held-out seekers), loaded fresh. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · S-learner (6')
# - [ ] 2 · T-learner (6')
# - [ ] 3 · X-learner (8')
# - [ ] 4 · Against the truth, one table (5')
# - [ ] ✱ Stretch · Refit the three with a linear base model

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
from sklearn.base import clone
import fursa as U
print("mode:", U.MODE)

P = U.pilot()
X = U.design(P).to_numpy()
y, t = P["employed_6m"].to_numpy(), P["coached"].to_numpy()
tr = (P["split"] == "train").to_numpy()
te = ~tr
tp = U.truth_pilot()
true_cate = (tp["p1"] - tp["p0"]).to_numpy()
Xtr, ytr, ttr = X[tr], y[tr], t[tr]
print(f"training {tr.sum():,} seekers · held out {te.sum():,} · true average effect {U.pp(true_cate.mean())}")
BASE = U.pilot_classifier(1)         # the classifier every learner uses (strongly regularised: 3,600 seekers, a binary outcome)
REG = U.pilot_regressor(2)           # the regressor of the X-learner's second stage

# %% [markdown]
# ## 1 · S-learner (6')
#
# One model `mu(x, t)` with coaching as one more column. The effect is `mu(x, 1) - mu(x, 0)`.
#
# ✏️ **Exercise** · Fit on the training split; return the predicted effects for **all** seekers.

# %%
def s_learner(Xtr, ttr, ytr, X):
    # BEGIN SOLUTION
    m = clone(BASE).fit(np.column_stack([Xtr, ttr]), ytr)
    one, zero = np.column_stack([X, np.ones(len(X))]), np.column_stack([X, np.zeros(len(X))])
    return m.predict_proba(one)[:, 1] - m.predict_proba(zero)[:, 1]
    # END SOLUTION


try:
    s = s_learner(Xtr, ttr, ytr, X)
except NotImplementedError:
    s = U.supplied("cate_pilot.csv")["s_learner"].to_numpy()
    U.check(not U.STRICT, "the solution run used its own S-learner")
print(f"S-learner: mean {U.pp(s[te].mean())}, spread (sd) {U.pp(s[te].std())}")
U.check(len(s) == len(P), "one effect per seeker")

# %% [markdown]
# ## 2 · T-learner (6')
#
# Two models, `mu1` on the coached and `mu0` on the uncoached. The effect is `mu1(x) - mu0(x)`.
#
# ✏️ **Exercise** · Return the predicted effects for all seekers, and the two fitted models (the X-learner reuses them).

# %%
def t_learner(Xtr, ttr, ytr, X):
    # BEGIN SOLUTION
    m1 = clone(BASE).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = clone(BASE).fit(Xtr[ttr == 0], ytr[ttr == 0])
    return m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1], m1, m0
    # END SOLUTION


try:
    tt, m1, m0 = t_learner(Xtr, ttr, ytr, X)
except NotImplementedError:
    tt = U.supplied("cate_pilot.csv")["t_learner"].to_numpy()
    m1 = clone(BASE).fit(Xtr[ttr == 1], ytr[ttr == 1]); m0 = clone(BASE).fit(Xtr[ttr == 0], ytr[ttr == 0])
    U.check(not U.STRICT, "the solution run used its own T-learner")
print(f"T-learner: mean {U.pp(tt[te].mean())}, spread (sd) {U.pp(tt[te].std())}")

# %% [markdown]
# ## 3 · X-learner (8')
#
# Impute each training seeker's effect with the other arm's model: for the coached, `d1 = y - mu0(x)`; for the uncoached, `d0 = mu1(x) - y`. Fit a regressor `tau1` on `d1` and `tau0` on `d0`. Combine with the propensity, which is 0.5 in a lottery: `tau(x) = 0.5 tau1(x) + 0.5 tau0(x)`.
#
# ✏️ **Exercise** · Return the predicted effects for all seekers. Use `clone(REG)` for the two regressors.

# %%
def x_learner(Xtr, ttr, ytr, X, m1, m0, e=0.5):
    # BEGIN SOLUTION
    d1 = ytr[ttr == 1] - m0.predict_proba(Xtr[ttr == 1])[:, 1]
    d0 = m1.predict_proba(Xtr[ttr == 0])[:, 1] - ytr[ttr == 0]
    tau1 = clone(REG).fit(Xtr[ttr == 1], d1)
    tau0 = clone(REG).fit(Xtr[ttr == 0], d0)
    return e * tau0.predict(X) + (1 - e) * tau1.predict(X)
    # END SOLUTION


try:
    xx = x_learner(Xtr, ttr, ytr, X, m1, m0)
except NotImplementedError:
    xx = U.supplied("cate_pilot.csv")["x_learner"].to_numpy()
    U.check(not U.STRICT, "the solution run used its own X-learner")
print(f"X-learner: mean {U.pp(xx[te].mean())}, spread (sd) {U.pp(xx[te].std())}")

# %% [markdown]
# **Look at** the weights. In general `tau(x) = e(x) tau0(x) + (1 - e(x)) tau1(x)`: the imputation from the larger arm gets more weight. In a 1:1 lottery the two count equally.

# %% [markdown]
# ## 4 · Against the truth, one table (5')
#
# On the held-out seekers, compare each learner with the true effects: the root mean squared error (PEHE), the correlation, the mean and the spread. On real data only the last two are available.
#
# ✏️ **Exercise** · Return the table, one row per learner.

# %%
def score_table(preds, truth):
    # BEGIN SOLUTION
    rows = []
    for name, p in preds.items():
        rows.append({"learner": name, "pehe": float(np.sqrt(np.mean((p - truth) ** 2))),
                     "corr": float(np.corrcoef(p, truth)[0, 1]), "mean": float(p.mean()), "sd": float(p.std())})
    return pd.DataFrame(rows).set_index("learner")
    # END SOLUTION


preds = {"S-learner": s[te], "T-learner": tt[te], "X-learner": xx[te]}
try:
    tab = score_table(preds, true_cate[te])
except NotImplementedError:
    tab = U.reference("d2_1", "table")
U.keep("d2_1", "table", tab)
print(tab.round(3).to_string())
print(f"truth: mean {true_cate[te].mean():.3f}, sd {true_cate[te].std():.3f}")
U.check(tab.loc["S-learner", "sd"] < tab.loc["T-learner", "sd"], "the S-learner's effects are the most compressed")
U.check(tab.loc["X-learner", "pehe"] < tab.loc["T-learner", "pehe"], "the X-learner beats the T-learner on error")
U.check(tab["corr"].max() > 0.3, "at least one learner ranks seekers usefully")

# %% [markdown]
# **Look at** the spread column. The truth varies by about the size of the T-learner's spread, but much of the T-learner's variation is noise: two separately fitted models subtract their errors. The S-learner's trees can ignore the coaching column, which shrinks effects towards zero.

# %% [markdown]
# ## ✱ Stretch · A linear base model
#
# Refit the three learners with logistic regression (and linear regression for the X-learner's second stage). Which learner suffers most, and why?

# %%
def stretch():
    # BEGIN SOLUTION
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    global BASE, REG
    keep_b, keep_r = BASE, REG
    BASE, REG = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), LinearRegression()
    s2 = s_learner(Xtr, ttr, ytr, X)
    t2, a, b = t_learner(Xtr, ttr, ytr, X)
    x2 = x_learner(Xtr, ttr, ytr, X, a, b)
    print(score_table({"S-linear": s2[te], "T-linear": t2[te], "X-linear": x2[te]}, true_cate[te]).round(3).to_string())
    BASE, REG = keep_b, keep_r
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why can the S-learner report almost no heterogeneity even when there is a lot?
#    <details><summary>Answer</summary>Its model treats coaching as one feature among fifteen; regularised trees may split on it rarely or never, so `mu(x, 1) - mu(x, 0)` shrinks towards zero.</details>
# 2. When does the X-learner help most?
#    <details><summary>Answer</summary>When the arms are unequal in size: each imputed effect leans on the model of the larger arm. In a 1:1 lottery its gain over the T-learner is smaller.</details>
# 3. On real data you have no truth column. What replaces it?
#    <details><summary>Answer</summary>Held-out randomised data: calibration by group, the best linear predictor and ranking checks, which Lab 2.3 runs.</details>
#
# ## Carry forward
#
# `data/reference/cate_pilot.csv` holds the course's version of these three predictions and a causal forest's, fitted on the same training split. Labs 2.3 and 2.4 start from it.
