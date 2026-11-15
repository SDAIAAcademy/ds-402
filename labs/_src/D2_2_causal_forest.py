# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "econml>=0.16", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 2.2 · A causal forest and its drivers
#
# Day 2 · Session 2 (Causal forests) · 25 min.
#
# A causal forest grows trees that split where effects differ, not where outcomes differ, after removing what predicts the outcome and the treatment. You fit EconML's `CausalForestDML` on the pilot's training split, rank the covariates that drive the variation in effects, and draw how the effect changes with months unemployed and digital skills.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/pilot.csv`, loaded fresh. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Fit the causal forest (8')
# - [ ] 2 · Feature importances (5')
# - [ ] 3 · Effect by months unemployed and digital skills (7')
# - [ ] 4 · Name the drivers for the director (5')
# - [ ] ✱ Stretch · Refit with another seed and compare the importances

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from econml.dml import CausalForestDML
import fursa as U
print("mode:", U.MODE)

P = U.pilot()
Xdf = U.design(P)
X = Xdf.to_numpy()
y, t = P["employed_6m"].to_numpy(), P["coached"].to_numpy()
tr = (P["split"] == "train").to_numpy()
te = ~tr
tp = U.truth_pilot()
true_cate = (tp["p1"] - tp["p0"]).to_numpy()
print(f"training {tr.sum():,} seekers, {Xdf.shape[1]} columns")

# %% [markdown]
# ## 1 · Fit the causal forest (8')
#
# `CausalForestDML` first residualises the outcome and the treatment with two models (local centring), then grows honest trees on the residuals: one half of each subsample chooses the splits, the other half estimates the leaf effects.
#
# ✏️ **Exercise** · Fit on the training split with gradient-boosted nuisance models (`HistGradientBoostingRegressor` for the outcome, `HistGradientBoostingClassifier` for the treatment, `min_samples_leaf=40`), `discrete_treatment=True`, `n_estimators` from `U.pick(400, 200)`, `min_samples_leaf=20`, `cv=5` and `random_state=U.SEED`. Return the fitted forest.

# %%
def fit_forest(X, t, y, seed=0):
    # BEGIN SOLUTION
    cf = CausalForestDML(model_y=HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=40, random_state=U.SEED),
                         model_t=HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=40, random_state=U.SEED),
                         discrete_treatment=True, n_estimators=U.pick(400, 200), min_samples_leaf=20,
                         cv=5, random_state=U.SEED + seed)
    return cf.fit(y, t, X=X)
    # END SOLUTION


try:
    cf = fit_forest(X[tr], t[tr], y[tr])
except NotImplementedError:
    cf = CausalForestDML(discrete_treatment=True, n_estimators=200, min_samples_leaf=20, random_state=U.SEED).fit(y[tr], t[tr], X=X[tr])
    U.check(not U.STRICT, "the solution run used its own forest")
tau = cf.effect(X)
print(f"held out: mean effect {U.pp(tau[te].mean())}, sd {U.pp(tau[te].std())}, "
      f"correlation with the truth {np.corrcoef(tau[te], true_cate[te])[0, 1]:.2f}")
U.check(abs(tau[te].mean() - true_cate[te].mean()) < 0.03, "the forest's average effect is within 3 points of the truth")

# %% [markdown]
# ## 2 · Feature importances (5')
#
# `cf.feature_importances_` measures how much each column's splits contributed to differences in effect (weighted by depth). It says where effects vary, not which way.
#
# ✏️ **Exercise** · Return a Series of importances indexed by column name, sorted from the largest.

# %%
def importances(cf, cols):
    # BEGIN SOLUTION
    return pd.Series(cf.feature_importances_, index=cols).sort_values(ascending=False)
    # END SOLUTION


try:
    imp = importances(cf, list(Xdf.columns))
except NotImplementedError:
    imp = U.reference("d2_2", "importances")
U.keep("d2_2", "importances", imp.round(4))
print(imp.round(3).head(8).to_string())
U.check("months_unemployed" in imp.index[:5], "months unemployed is among the five main drivers")

# %% [markdown]
# ## 3 · Effect by months unemployed and digital skills (7')
#
# Average the forest's predicted effect over the held-out seekers in bins of months unemployed, and over each level of digital skills.
#
# ✏️ **Exercise** · Return two Series: mean predicted effect by months bin (`0-2, 3-6, 7-10, 11-14, 15-18, 19-24, 25-36, over 36`) and by digital skills (0 to 10).

# %%
BINS = [-1, 2, 6, 10, 14, 18, 24, 36, 99]
LABELS = ["0-2", "3-6", "7-10", "11-14", "15-18", "19-24", "25-36", "over 36"]


def profiles(P, tau, rows):
    # BEGIN SOLUTION
    d = P[rows].assign(tau=tau[rows])
    by_m = d.groupby(pd.cut(d["months_unemployed"], BINS, labels=LABELS), observed=True)["tau"].mean()
    by_d = d.groupby("digital_skills")["tau"].mean()
    return by_m, by_d
    # END SOLUTION


try:
    by_m, by_d = profiles(P, tau, te)
except NotImplementedError:
    by_m, by_d = U.reference("d2_2", "profiles")
U.keep("d2_2", "profiles", [by_m.round(4), by_d.round(4)])
truth_m = P[te].assign(c=true_cate[te]).groupby(pd.cut(P[te]["months_unemployed"], BINS, labels=LABELS), observed=True)["c"].mean()
print(pd.DataFrame({"forest": by_m, "truth": truth_m}).mul(100).round(1).to_string(), "\n(points)")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(range(len(by_m)), 100 * by_m.values, marker="o", label="forest")
ax[0].plot(range(len(truth_m)), 100 * truth_m.values, marker="x", ls="--", label="truth")
ax[0].set_xticks(range(len(by_m)), by_m.index, rotation=30); ax[0].set_ylabel("effect (points)"); ax[0].set_xlabel("months unemployed"); ax[0].legend()
ax[1].bar(by_d.index, 100 * by_d.values); ax[1].set_xlabel("digital skills (0 to 10)")
fig.tight_layout(); fig.savefig(U.artifacts("d2_2") / "profiles.png", dpi=120)
U.check(by_m.idxmax() in ("7-10", "11-14", "15-18"), "the effect peaks for seekers unemployed about a year")
U.check(by_d.loc[by_d.index <= 3].mean() > by_d.loc[by_d.index >= 6].mean(), "seekers with low digital skills gain more")

# %% [markdown]
# ## 4 · Name the drivers for the director (5')
#
# ✏️ **Exercise** · Two or three sentences for Wijdan: which seekers gain most, which gain least, and what this does not yet prove.

# %%
def drivers_text():
    # BEGIN SOLUTION
    return (f"Coaching helps most for seekers unemployed for about a year ({100 * by_m.max():.0f} points in the best band) "
            f"and for those with low digital skills; it helps least, or not at all, for the newly unemployed "
            f"({100 * by_m.iloc[0]:.0f} points for 0 to 2 months) and the very long-term unemployed. "
            "These are the forest's estimates on the training seekers' model; Lab 2.3 checks on held-out seekers whether its ranking holds before any place follows it.")
    # END SOLUTION


try:
    text = drivers_text()
except NotImplementedError:
    text = U.reference("d2_2", "text")
U.keep("d2_2", "text", text)
print(text)
U.check(len(text) > 150, "a paragraph for the director")

# %% [markdown]
# ## ✱ Stretch · Another seed
#
# Refit with `seed=1` and compare the top five importances. Which are stable?

# %%
def stretch():
    # BEGIN SOLUTION
    cf2 = fit_forest(X[tr], t[tr], y[tr], seed=1)
    imp2 = pd.Series(cf2.feature_importances_, index=Xdf.columns).sort_values(ascending=False)
    print(pd.DataFrame({"seed 0": imp.head(5).index, "seed 1": imp2.head(5).index}).to_string(index=False))
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why residualise the outcome and the treatment before growing the trees?
#    <details><summary>Answer</summary>So the splits chase differences in the effect rather than differences in the outcome level or in who was treated. In a lottery the treatment part matters little; in observational data it is essential.</details>
# 2. What does honesty buy?
#    <details><summary>Answer</summary>The leaf effects are estimated on seekers that did not choose the splits, so the estimates are not inflated by the search, and confidence intervals are valid.</details>
# 3. A covariate has zero importance. Does it play no role in effects?
#    <details><summary>Answer</summary>Not necessarily: a correlated covariate may have taken its splits. Importance is a description of this forest, not of the world.</details>
#
# ## Carry forward
#
# The course's causal forest predictions are in `data/reference/cate_pilot.csv` (`causal_forest`), for Lab 2.3.
