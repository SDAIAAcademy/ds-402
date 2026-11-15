# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 1.2 · Groups almost never offered
#
# Day 1 · Session 2 (Overlap first) · 25 min.
#
# Before estimating any effect from the logs, find the seekers the logs can say nothing about. You fit a cross-fitted propensity model on the 60 confounders of 40,000 logged decisions, look at the scores of offered and not offered seekers, map the groups that were almost never offered coaching, and decide what an estimate may claim about them.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/logged.csv.gz`, loaded fresh (a seeded sample of 10,000 decisions in quick mode). Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · A cross-fitted propensity model (7')
# - [ ] 2 · Scores by arm (5')
# - [ ] 3 · Map the groups almost never offered (8')
# - [ ] 4 · Trim, or change the estimand (5')
# - [ ] ✱ Stretch · Set your fitted scores against the probabilities the tool recorded

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fursa as U
print("mode:", U.MODE)

L = U.logged_sample()
X = U.design(L, U.CONFOUNDERS)
t = L["offered"].to_numpy()
print(f"{len(L):,} decisions · {X.shape[1]} columns from 60 confounders (region as indicators) · offered {t.mean():.1%}")

# %% [markdown]
# ## 1 · A cross-fitted propensity model (7')
#
# Fit a gradient-boosted classifier of `offered` on the confounders, five folds: each decision's score comes from a model that never saw it. In-sample scores of a flexible model are too close to 0 and 1.
#
# ✏️ **Exercise** · Return the cross-fitted scores, one per decision. `sklearn.model_selection.cross_val_predict` with `method="predict_proba"` does the folds.

# %%
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score


def crossfit_propensity(X, t):
    # BEGIN SOLUTION
    cv = KFold(5, shuffle=True, random_state=U.SEED)
    return cross_val_predict(U.gbm_classifier(), X, t, cv=cv, method="predict_proba")[:, 1]
    # END SOLUTION


try:
    e = crossfit_propensity(X, t)
except NotImplementedError:
    e = U.reference("d1_2", "e")
    e = e if len(e) == len(L) else U.crossfit_aipw(L["employed_6m"], t, X, U.gbm_classifier(), U.gbm_classifier())["e"]
U.keep("d1_2", "e", np.round(e, 5))
print(f"AUC {roc_auc_score(t, e):.3f} · scores from {e.min():.3f} to {e.max():.3f}")
U.check(len(e) == len(L) and 0.6 < roc_auc_score(t, e) < 0.95, "one cross-fitted score per decision, informative but not perfect")

# %% [markdown]
# ## 2 · Scores by arm (5')
#
# Overlap means every kind of seeker had some chance of each decision. Plot the two distributions and count the decisions with a score under 0.05.
#
# ✏️ **Exercise** · Return the share of not-offered decisions with `e < cut`, and the share of offered ones.

# %%
def share_below(e, t, cut=0.05):
    # BEGIN SOLUTION
    return float(np.mean(e[t == 0] < cut)), float(np.mean(e[t == 1] < cut))
    # END SOLUTION


try:
    low0, low1 = share_below(e, t)
except NotImplementedError:
    low0, low1 = U.reference("d1_2", "low")
U.keep("d1_2", "low", [low0, low1])
fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.linspace(0, 1, 41)
ax.hist(e[t == 0], bins=bins, alpha=0.6, density=True, label="not offered")
ax.hist(e[t == 1], bins=bins, alpha=0.6, density=True, label="offered")
ax.axvline(0.05, color="k", lw=1, ls="--")
ax.set_xlabel("cross-fitted propensity score"); ax.legend()
fig.tight_layout(); fig.savefig(U.artifacts("d1_2") / "overlap.png", dpi=120)
print(f"scores under 0.05: {low0:.1%} of the not offered, {low1:.1%} of the offered")
U.check(low0 > low1, "low scores are mostly among the not offered: someone was almost never offered")

# %% [markdown]
# ## 3 · Map the groups almost never offered (8')
#
# Cut the decisions by region, disability status and months unemployed (`0-12`, `13-36`, `over 36`) and age (`55 or under`, `over 55`).
#
# ✏️ **Exercise** · Return one row per cell with at least 30 decisions: `n`, `offer_rate` (observed) and `mean_e`; keep the cells where `offer_rate < 0.05`, sorted by `offer_rate`.

# %%
def overlap_map(L, e, min_n=30):
    # BEGIN SOLUTION
    d = L.assign(e=e,
                 duration=pd.cut(L["months_unemployed"], [-1, 12, 36, 99], labels=["0-12", "13-36", "over 36"]),
                 age_band=np.where(L["age"] > 55, "over 55", "55 or under"))
    g = (d.groupby(["region", "disability", "duration", "age_band"], observed=True)
         .agg(n=("offered", "size"), offer_rate=("offered", "mean"), mean_e=("e", "mean")).reset_index())
    g = g[(g.n >= min_n) & (g.offer_rate < 0.05)]
    return g.sort_values("offer_rate").reset_index(drop=True)
    # END SOLUTION


try:
    cells = overlap_map(L, e)
except NotImplementedError:
    cells = U.reference("d1_2", "cells")
U.keep("d1_2", "cells", cells)
print(cells.round(3).to_string())
key = lambda r, d, du, a: r.astype(str) + "/" + d.astype(str) + "/" + du.astype(str) + "/" + a.astype(str)
mapped = set(key(cells.region, cells.disability, cells.duration, cells.age_band))
dur = pd.cut(L.months_unemployed, [-1, 12, 36, 99], labels=["0-12", "13-36", "over 36"])
inside = key(L.region, L.disability, dur, pd.Series(np.where(L.age > 55, "over 55", "55 or under"))).isin(mapped)
print(f"{inside.mean():.1%} of decisions sit in a cell almost never offered")
U.check(((cells.disability == 1) & cells.region.isin(["Rumh", "Nijad"])).any(), "disabled seekers in Rumh or Nijad are in the map")
U.check((cells.duration.astype(str) == "over 36").any(), "seekers unemployed over 36 months are in the map")

# %% [markdown]
# ## 4 · Trim, or change the estimand (5')
#
# Two honest options. **Trim**: drop decisions with `e` outside `[0.05, 0.95]` and estimate the average effect for the rest; the population changes, so the estimand changes. **Keep** the population, and say that for the mapped groups the estimate rests on the outcome model alone.
#
# ✏️ **Exercise** · Return the share of decisions trimming would drop, and a short decision for the Q1 estimand with its reason.

# %%
def decide(e):
    # BEGIN SOLUTION
    dropped = float(np.mean((e < 0.05) | (e > 0.95)))
    note = (f"Trim at 0.05: it drops {dropped:.0%} of decisions, almost all from the mapped groups. Report the effect for the "
            "seekers the agency actually considers for coaching, name the groups left out, and do not extend the estimate to them.")
    return dropped, note
    # END SOLUTION


try:
    dropped, note = decide(e)
except NotImplementedError:
    dropped, note = U.reference("d1_2", "decision")
U.keep("d1_2", "decision", [dropped, note])
print(note)
U.check(0 < dropped < 0.3, "trimming drops some decisions, not most")

# %% [markdown]
# ## ✱ Stretch · Your scores against the tool's records
#
# The logs carry `p_offer`, the probability the allocation tool recorded. Plot it against your cross-fitted score. Where do they disagree, and which would you trust? (Lab 3.1 returns to this.)

# %%
def stretch():
    # BEGIN SOLUTION
    r = np.corrcoef(e, L["p_offer"])[0, 1]
    by = L.assign(e=e).groupby("office")[["e", "p_offer", "offered"]].mean().round(3)
    print(f"correlation {r:.2f}\n{by.to_string()}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why fit the propensity model with cross-fitting rather than on all the data at once?
#    <details><summary>Answer</summary>A flexible model scores the rows it was trained on too close to 0 and 1, which inflates weights and hides or invents overlap problems. Out-of-fold scores behave like scores for new seekers.</details>
# 2. Trimming at 0.05 removes the disabled seekers of Rumh and Nijad. What does the trimmed estimate say about them?
#    <details><summary>Answer</summary>Nothing. The estimand is now the effect for seekers with overlap; their effect is not estimated, and any rule for them needs other evidence.</details>
# 3. Is a high AUC of the propensity model good news?
#    <details><summary>Answer</summary>No. It means caseworkers' choices are predictable from the covariates, which is what produces poor overlap.</details>
#
# ## Carry forward
#
# Labs 1.3 and 1.4 fit their own propensity models inside double machine learning. Lab 3.3 meets the same groups again when a rule wants to coach them.
