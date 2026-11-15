# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4", "doubleml>=0.11"]
# ///
# %% [markdown]
# # Lab 1.3 · Double machine learning against regression
#
# Day 1 · Session 3 (Double machine learning) · 25 min.
#
# Q1 asks whether coaching helped. The logs hold 60 confounders, and caseworkers' choices depend on some of them in ways no straight line captures. You estimate the average effect naively and by regression adjustment, then by double machine learning: first by hand, with cross-fitting and the orthogonal score, then with DoubleML. The generator knows the true average effect, so every estimate gets a bias.
#
# **AI policy.** AI-off: this is the day's first estimator. Assistants closed; the slides, the documentation and the trainer are open.
#
# **Reference state.** `data/logged.csv.gz`, loaded fresh (10,000 decisions in quick mode). Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Naive and regression-adjusted estimates (5')
# - [ ] 2 · Cross-fitting by hand (9')
# - [ ] 3 · The same with DoubleML (6')
# - [ ] 4 · One table: estimate, interval, bias (5')
# - [ ] ✱ Stretch · The partially linear model against the interactive one

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

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import fursa as U
print("mode:", U.MODE)

L = U.logged_sample()
X = U.design(L, U.CONFOUNDERS)
y, t = L["employed_6m"].to_numpy(), L["offered"].to_numpy()
tr = U.truth_for(L["decision_id"])
ATE = float((tr["p1"] - tr["p0"]).mean())          # the truth for these decisions; real data never give you this
print(f"{len(L):,} decisions · true average effect {U.pp(ATE)}")

# %% [markdown]
# ## 1 · Naive and regression-adjusted estimates (5')
#
# ✏️ **Exercise** · Return the naive difference in employment rates (offered minus not offered) and the regression-adjusted estimate with its standard error (`U.ols_adjust(y, t, X)`: the coefficient of the offer in a linear regression on the offer and all 60 confounders).

# %%
def first_two(y, t, X):
    # BEGIN SOLUTION
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    ols, ols_se = U.ols_adjust(y, t, X)
    return naive, ols, ols_se
    # END SOLUTION


try:
    naive, ols, ols_se = first_two(y, t, X)
except NotImplementedError:
    naive, ols, ols_se = U.reference("d1_3", "first_two")
U.keep("d1_3", "first_two", [naive, ols, ols_se])
print(f"naive {U.pp(naive)} · regression {U.pp(ols)} (se {U.pp(ols_se)}) · truth {U.pp(ATE)}")
U.check(abs(naive - ATE) > 0.01, "the naive difference misses the truth by more than a point")

# %% [markdown]
# ## 2 · Cross-fitting by hand (9')
#
# The interactive regression model (IRM) score for the average effect, for each decision `i`:
#
# `psi_i = m1(x_i) - m0(x_i) + t_i (y_i - m1(x_i)) / e(x_i) - (1 - t_i)(y_i - m0(x_i)) / (1 - e(x_i))`
#
# with `e` the propensity model and `m1`, `m0` the outcome models among offered and not offered. **Cross-fitting**: split into five folds; for each fold, fit the three models on the other four and predict on this one. The estimate is the mean of `psi`; its standard error is `std(psi) / sqrt(n)`. Clip `e` to `[0.01, 0.99]`.
#
# ✏️ **Exercise** · Return `(estimate, se)`. Use `U.gbm_classifier()` for all three models and `U.folds(n, 5)` for the fold numbers.

# %%
from sklearn.base import clone


def dml_by_hand(y, t, X, k=5):
    X = np.asarray(X, float)
    # BEGIN SOLUTION
    n = len(y)
    fold = U.folds(n, k)
    e, m0, m1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for f in range(k):
        a, b = fold != f, fold == f
        e[b] = clone(U.gbm_classifier()).fit(X[a], t[a]).predict_proba(X[b])[:, 1]
        m1[b] = clone(U.gbm_classifier()).fit(X[a & (t == 1)], y[a & (t == 1)]).predict_proba(X[b])[:, 1]
        m0[b] = clone(U.gbm_classifier()).fit(X[a & (t == 0)], y[a & (t == 0)]).predict_proba(X[b])[:, 1]
    e = np.clip(e, 0.01, 0.99)
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(n))
    # END SOLUTION


try:
    hand, hand_se = dml_by_hand(y, t, X)
except NotImplementedError:
    r = U.crossfit_aipw(y, t, X, U.gbm_classifier(), U.gbm_classifier())
    hand, hand_se = r["est"], r["se"]
    print("↪ using the course's cross-fitting (U.crossfit_aipw): finish the exercise to use your own")
    U.check(not U.STRICT, "the solution run used its own cross-fitting")
print(f"by hand {U.pp(hand)} (se {U.pp(hand_se)}) · truth {U.pp(ATE)}")
U.check(abs(hand - ATE) < 2.5 * hand_se + 0.005, "cross-fitted DML lands within its interval of the truth")
U.check(abs(hand - ATE) < abs(ols - ATE), "DML is closer to the truth than regression adjustment")

# %% [markdown]
# ## 3 · The same with DoubleML (6')
#
# DoubleML does the same steps behind one object. `DoubleMLData` names the outcome, the treatment and the confounders; `DoubleMLIRM` takes a learner for the outcome (`ml_g`) and for the propensity (`ml_m`).
#
# ✏️ **Exercise** · Fit `DoubleMLIRM` with five folds, the course's gradient boosting for both learners and propensity scores clipped at 0.01. Return the fitted object.

# %%
import doubleml as dml

df = X.assign(employed_6m=y, offered=t)
data = dml.DoubleMLData(df, y_col="employed_6m", d_cols="offered", x_cols=list(X.columns))


def fit_irm(data):
    # BEGIN SOLUTION
    m = dml.DoubleMLIRM(data, ml_g=U.gbm_classifier(), ml_m=U.gbm_classifier(), n_folds=5,
                        ps_processor_config=dml.utils.PSProcessorConfig(clipping_threshold=0.01))
    np.random.seed(U.SEED)
    return m.fit()
    # END SOLUTION


try:
    irm = fit_irm(data)
except NotImplementedError:
    irm = dml.DoubleMLIRM(data, ml_g=U.gbm_classifier(), ml_m=U.gbm_classifier(), n_folds=5).fit()
    print("↪ using a default DoubleMLIRM: finish the exercise to set it up yourself")
    U.check(not U.STRICT, "the solution run used its own DoubleML model")
pkg, pkg_se = float(irm.coef[0]), float(irm.se[0])
print(irm.summary.round(4))
U.check(abs(pkg - hand) < 3 * hand_se, "DoubleML agrees with your cross-fitting within sampling noise")

# %% [markdown]
# ## 4 · One table: estimate, interval, bias (5')
#
# ✏️ **Exercise** · Return a table with one row per method (`naive`, `regression adjustment`, `DML by hand`, `DoubleML`): `estimate`, `lo`, `hi` (95% interval; none for naive) and `bias` (estimate minus truth).

# %%
def estimate_table():
    # BEGIN SOLUTION
    rows = [("naive", naive, np.nan), ("regression adjustment", ols, ols_se),
            ("DML by hand", hand, hand_se), ("DoubleML", pkg, pkg_se)]
    tab = pd.DataFrame(rows, columns=["method", "estimate", "se"])
    tab["lo"], tab["hi"] = tab.estimate - 1.96 * tab.se, tab.estimate + 1.96 * tab.se
    tab["bias"] = tab.estimate - ATE
    return tab[["method", "estimate", "lo", "hi", "bias"]]
    # END SOLUTION


try:
    tab = estimate_table()
except NotImplementedError:
    tab = U.reference("d1_3", "table")
U.keep("d1_3", "table", tab)
print((tab.set_index("method") * 100).round(1).to_string(), "\n(percentage points)")
tab.to_csv(U.artifacts("d1_3") / "estimates.csv", index=False)
reg = tab.set_index("method")
U.check(not (reg.loc["regression adjustment", "lo"] <= ATE <= reg.loc["regression adjustment", "hi"]) or U.QUICK,
        "the regression interval misses the truth (full data)")
U.check(reg.loc["DoubleML", "lo"] <= ATE <= reg.loc["DoubleML", "hi"], "the DoubleML interval covers the truth")

# %% [markdown]
# **Look at** the regression row. Sixty controls and a narrow interval, centred in the wrong place: the confounding runs through curves and interactions a linear term cannot follow. More controls did not fix the functional form; flexible nuisance models with an orthogonal score did.

# %% [markdown]
# ## ✱ Stretch · Partially linear against interactive
#
# `DoubleMLPLR` assumes one constant effect, `y = theta t + g(x) + noise`, and weights decisions by the variance of the offer given x. Fit it and compare with the IRM estimate. Which estimand does each target when effects differ by seeker?

# %%
def stretch():
    # BEGIN SOLUTION
    from sklearn.ensemble import HistGradientBoostingRegressor
    plr = dml.DoubleMLPLR(data, ml_l=HistGradientBoostingRegressor(max_iter=U.pick(200, 80), min_samples_leaf=40, random_state=U.SEED),
                          ml_m=U.gbm_classifier(), n_folds=5).fit()
    print(f"PLR {U.pp(plr.coef[0])} (se {U.pp(plr.se[0])}) · IRM {U.pp(pkg)} · truth ATE {U.pp(ATE)}")
    print("PLR averages effects with weights e(1-e): seekers whose offer was most uncertain count most.")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does the score subtract the residual terms rather than just averaging `m1 - m0`?
#    <details><summary>Answer</summary>The correction terms make the score insensitive to small errors in the nuisance models (Neyman orthogonality), so regularisation bias in `m` and `e` does not pass to first order into the estimate.</details>
# 2. What would go wrong without cross-fitting?
#    <details><summary>Answer</summary>Each decision's own outcome would help fit the models that predict it; flexible models overfit, and the residuals in the score shrink in a way that biases the estimate.</details>
# 3. The regression interval is narrow. Why is that no comfort?
#    <details><summary>Answer</summary>An interval measures sampling noise around the estimate, not bias from a wrong model. A precise wrong answer is still wrong.</details>
#
# ## Carry forward
#
# Lab 1.4 swaps the nuisance learners and removes a confounder; it starts from the logs again.
