# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 3.1 · Current practice from the logs
#
# Day 3 · Session 1 (Logged decisions) · 25 min.
#
# Fursa's allocation tool recorded, for every decision, the probability with which it would offer coaching (`p_offer`). Caseworkers then decided. If the recorded probabilities are right, the logs can value any rule, including ones never used. Before trusting them, you value current practice, check that the recorded probabilities reproduce the decisions they describe, and find the office where they do not.
#
# **AI policy.** AI-off: this is the day's first estimator. Assistants closed; the slides, the documentation and the trainer are open.
#
# **Reference state.** `data/logged.csv.gz`, loaded fresh, all 40,000 decisions (the checks here are fast). Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · The value of current practice (4')
# - [ ] 2 · Weight the logging rule by its own records (6')
# - [ ] 3 · Recorded against actual offers, per office (9')
# - [ ] 4 · Flag the office and decide (6')
# - [ ] ✱ Stretch · A calibration plot of the recorded probabilities

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

L = U.logged()
y, a, p = L["employed_6m"].to_numpy(), L["offered"].to_numpy(), L["p_offer"].to_numpy()
print(f"{len(L):,} decisions in {L.office.nunique()} offices · offered {a.mean():.1%} · recorded probability, mean {p.mean():.3f}")

# %% [markdown]
# ## 1 · The value of current practice (4')
#
# Current practice is the rule that produced the logs, so its value needs no weighting: it is the employment rate in the logs.
#
# ✏️ **Exercise** · Return the value of current practice and its standard error.

# %%
def current_value(y):
    # BEGIN SOLUTION
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y)))
    # END SOLUTION


try:
    v_cur, se_cur = current_value(y)
except NotImplementedError:
    v_cur, se_cur = U.reference("d3_1", "current")
U.keep("d3_1", "current", [v_cur, se_cur])
print(f"current practice: {v_cur:.1%} employed within six months (se {se_cur:.1%})")
U.check(0.4 < v_cur < 0.8, "a plausible employment rate")

# %% [markdown]
# ## 2 · Weight the logging rule by its own records (6')
#
# If `p` is the true probability of an offer, then on average `a / p = 1` and `(1 - a) / (1 - p) = 1`. These two means are the simplest test of logged propensities: each should be within a few standard errors of 1. They are also the weights every off-policy estimator uses.
#
# ✏️ **Exercise** · Return the two means and their standard errors.

# %%
def weight_means(a, p):
    # BEGIN SOLUTION
    w1, w0 = a / p, (1 - a) / (1 - p)
    n = len(a)
    return float(w1.mean()), float(w1.std(ddof=1) / np.sqrt(n)), float(w0.mean()), float(w0.std(ddof=1) / np.sqrt(n))
    # END SOLUTION


try:
    m1, s1, m0, s0 = weight_means(a, p)
except NotImplementedError:
    m1, s1, m0, s0 = U.reference("d3_1", "weights")
U.keep("d3_1", "weights", [m1, s1, m0, s0])
print(f"mean of a/p = {m1:.3f} (se {s1:.3f}) · mean of (1-a)/(1-p) = {m0:.3f} (se {s0:.3f})")
U.check(abs(m1 - 1) > 3 * s1, "the offer weights do not average to 1: some recorded probabilities are wrong")

# %% [markdown]
# ## 3 · Recorded against actual offers, per office (9')
#
# A wrong probability can hide in one place. For each office compare the mean recorded probability with the share actually offered, as a z-score: `(offered - recorded) / sqrt(sum p(1-p)) * n`, which is the standardised difference between observed and expected offers.
#
# ✏️ **Exercise** · Return one row per office: `n`, `recorded` (mean `p_offer`), `actual` (share offered), `weight_mean` (mean of `a/p`) and `z`, sorted by `z` from the largest.

# %%
def by_office(L):
    # BEGIN SOLUTION
    rows = []
    for office, g in L.groupby("office"):
        p_, a_ = g["p_offer"].to_numpy(), g["offered"].to_numpy()
        z = (a_.sum() - p_.sum()) / np.sqrt(np.sum(p_ * (1 - p_)))
        rows.append({"office": office, "n": len(g), "recorded": p_.mean(), "actual": a_.mean(),
                     "weight_mean": np.mean(a_ / p_), "z": z})
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)
    # END SOLUTION


try:
    off = by_office(L)
except NotImplementedError:
    off = U.reference("d3_1", "offices")
U.keep("d3_1", "offices", off.round(4))
print(off.round(3).to_string())
U.check(off.loc[0, "office"] == "RUM-2" and off.loc[0, "z"] > 10, "one office stands far above the rest")
U.check(off.loc[1:, "z"].abs().max() < 4, "every other office is within sampling noise")

# %% [markdown]
# **Look at** RUM-2's weight mean. Its offered seekers carry weights computed from probabilities about a third of the real ones, so every rule that agrees with RUM-2's offers is overvalued by the same factor there.

# %% [markdown]
# ## 4 · Flag the office and decide (6')
#
# Duhaim Al-Thumairi, head of caseworkers, confirms that RUM-2 overrode the tool for much of 2025 and 2026 after a staffing change; the tool still recorded its own number.
#
# ✏️ **Exercise** · Return the flagged office, the decision (`"exclude"`, `"re-estimate"` or `"keep"`) and one sentence of reasons. Then check your decision: the inverse-propensity value of "offer to no one" with and without the office, against the truth.

# %%
def decide(off):
    # BEGIN SOLUTION
    return off.loc[0, "office"], "exclude", (
        "RUM-2's recorded probabilities describe the tool, not the decisions; excluding its 4,000 decisions keeps every other "
        "office's records, which pass the check, and costs 10% of the data. Re-estimating its propensities from its covariates "
        "is the alternative when the office matters to a rule.")
    # END SOLUTION


try:
    flagged, decision, reason = decide(off)
except NotImplementedError:
    flagged, decision, reason = U.reference("d3_1", "decision")
U.keep("d3_1", "decision", [flagged, decision, reason])
print(flagged, "·", decision, "·", reason)
none = np.zeros(len(L), int)
keep = (L["office"] != flagged).to_numpy()
v_all = U.ipw_value(y, a, p, none)[0]
v_keep = U.ipw_value(y[keep], a[keep], p[keep], none[keep])[0]
tr = U.truth_logged()
print(f"offer to no one, IPW: all offices {v_all:.1%}, without {flagged} {v_keep:.1%} · truth {tr.loc[keep, 'p0'].mean():.1%} (offices kept)")
U.check(flagged == "RUM-2" and decision in ("exclude", "re-estimate"), "the override office is flagged and not used as recorded")

# %% [markdown]
# ## ✱ Stretch · A calibration plot of the recorded probabilities
#
# Bin decisions by `p_offer` into ten groups and plot the actual offer rate against the mean recorded probability, once for RUM-2 and once for every other office.

# %%
def stretch():
    # BEGIN SOLUTION
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for name, g in [("other offices", L[L.office != "RUM-2"]), ("RUM-2", L[L.office == "RUM-2"])]:
        b = pd.qcut(g["p_offer"], 10, duplicates="drop")
        c = g.groupby(b, observed=True).agg(rec=("p_offer", "mean"), act=("offered", "mean"))
        ax.plot(c.rec, c.act, marker="o", label=name)
    ax.plot([0, 0.9], [0, 0.9], "k--", lw=1); ax.set_xlabel("recorded"); ax.set_ylabel("actual"); ax.legend()
    fig.tight_layout(); fig.savefig(U.artifacts("d3_1") / "calibration.png", dpi=120)
    print("saved", U.artifacts("d3_1") / "calibration.png")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does the value of current practice need no weights?
#    <details><summary>Answer</summary>The logs were produced by current practice; their mean outcome is its value for this population.</details>
# 2. The weight means are 1.00 overall in another agency's logs. Are its probabilities right?
#    <details><summary>Answer</summary>Not necessarily: errors in opposite directions can cancel overall. Check by office, by period and by group too.</details>
# 3. Why not simply refit a propensity model for every office?
#    <details><summary>Answer</summary>Recorded probabilities, when right, are exact and include information no model of the covariates has (a random component of the tool). Refit only where the records are shown to be wrong, and say so.</details>
#
# ## Carry forward
#
# Labs 3.2 and 3.3 use the logs without RUM-2: `U.logged(drop_override=True)`.
