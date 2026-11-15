# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 2.4 · Uplift, Qini and a threshold
#
# Day 2 · Session 4 (Uplift for targeting) · 25 min.
#
# A ranking becomes places only through a curve: how many extra jobs each extra place buys. On the held-out pilot you draw uplift and Qini curves for targeting by the causal forest's estimated benefit and by the agency's risk score, set the threshold for one place per five seekers, and count the places that would change hands if the agency switched.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/reference/cate_pilot.csv` (the `causal_forest` column) and `data/pilot.csv`. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Uplift in the top shares (6')
# - [ ] 2 · Qini curves against random and risk (7')
# - [ ] 3 · The threshold at one place in five (6')
# - [ ] 4 · Places that change hands (6')
# - [ ] ✱ Stretch · Extra jobs per hundred places at three budgets

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

P = U.pilot()
C = U.supplied("cate_pilot.csv")
te = (P["split"] == "test").to_numpy()
H = P[te].reset_index(drop=True)
benefit = C.loc[te, "causal_forest"].to_numpy()
risk = H["risk_score"].to_numpy().astype(float)
y, t = H["employed_6m"].to_numpy(), H["coached"].to_numpy()
print(f"held-out seekers: {len(H):,}")

# %% [markdown]
# ## 1 · Uplift in the top shares (6')
#
# For a ranking and a share `q`, the uplift is the observed effect among the top `q` of seekers (`U.uplift_at`).
#
# ✏️ **Exercise** · Return a table with one row per share in `[0.1, 0.2, 0.3, 0.5, 1.0]` and two columns, `benefit` and `risk`.

# %%
SHARES = [0.1, 0.2, 0.3, 0.5, 1.0]


def uplift_table(benefit, risk, y, t):
    # BEGIN SOLUTION
    return pd.DataFrame({"benefit": [U.uplift_at(benefit, y, t, q) for q in SHARES],
                         "risk": [U.uplift_at(risk, y, t, q) for q in SHARES]}, index=SHARES)
    # END SOLUTION


try:
    up = uplift_table(benefit, risk, y, t)
except NotImplementedError:
    up = U.reference("d2_4", "uplift")
U.keep("d2_4", "uplift", up.round(4))
print((up * 100).round(1).to_string(), "\n(points)")
U.check(up.loc[0.2, "benefit"] > up.loc[0.2, "risk"], "in the top fifth, ranking by benefit beats ranking by risk")
U.check(abs(up.loc[1.0, "benefit"] - up.loc[1.0, "risk"]) < 1e-9, "at 100% the two rankings coach the same people")

# %% [markdown]
# ## 2 · Qini curves against random and risk (7')
#
# The Qini curve counts the extra employed seekers, per seeker in the population, when the top share is coached. A random ranking gives the straight line from zero to the full effect. `U.qini_curve` and `U.qini_area` (the area between a curve and that line) do the work.
#
# ✏️ **Exercise** · Return the two curves and their areas.

# %%
def qini_both(benefit, risk, y, t):
    # BEGIN SOLUTION
    qb, qr = U.qini_curve(benefit, y, t), U.qini_curve(risk, y, t)
    return qb, qr, U.qini_area(qb), U.qini_area(qr)
    # END SOLUTION


try:
    qb, qr, ab, ar = qini_both(benefit, risk, y, t)
except NotImplementedError:
    qb, qr = U.qini_curve(benefit, y, t), U.qini_curve(risk, y, t)
    ab, ar = U.qini_area(qb), U.qini_area(qr)
    U.check(not U.STRICT, "the solution run used its own curves")
print(f"Qini area: benefit {1000 * ab:.1f}, risk {1000 * ar:.1f} (per thousand seekers)")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(100 * qb.share, 100 * qb.qini, label="by estimated benefit")
ax.plot(100 * qr.share, 100 * qr.qini, label="by risk score")
ax.plot([0, 100], [0, 100 * qb.qini.iloc[-1]], "k--", lw=1, label="random")
ax.axvline(20, color="grey", lw=1, ls=":")
ax.set_xlabel("share of seekers coached (%)"); ax.set_ylabel("extra employed per 100 seekers"); ax.legend()
fig.tight_layout(); fig.savefig(U.artifacts("d2_4") / "qini.png", dpi=120)
U.check(ab > ar, "the benefit ranking has the larger Qini area")

# %% [markdown]
# ## 3 · The threshold at one place in five (6')
#
# A rule for next year needs a number, not a rank: offer coaching when the estimated benefit is at least the threshold that selects one seeker in five.
#
# ✏️ **Exercise** · Return the threshold (the 80th percentile of `benefit`) and the observed effect among held-out seekers above it.

# %%
def threshold(benefit, y, t, share=U.BUDGET):
    # BEGIN SOLUTION
    c = float(np.quantile(benefit, 1 - share))
    sel = benefit >= c
    return c, float(y[sel & (t == 1)].mean() - y[sel & (t == 0)].mean())
    # END SOLUTION


try:
    cut, gain = threshold(benefit, y, t)
except NotImplementedError:
    cut, gain = U.reference("d2_4", "threshold")
U.keep("d2_4", "threshold", [cut, gain])
print(f"offer coaching when the estimated benefit is at least {U.pp(cut)}; observed effect above it {U.pp(gain)}")
U.check(cut > 0, "the threshold is a positive effect")

# %% [markdown]
# ## 4 · Places that change hands (6')
#
# If the agency switched from the risk score to estimated benefit, how many of the top fifth would change, and what would the true gain be? (The truth is the generator's; real data never give you this.)
#
# ✏️ **Exercise** · Return the share of the benefit fifth that is not in the risk fifth, and the true mean gain of each fifth.

# %%
tp = U.truth_pilot()[te].reset_index(drop=True)
true_cate = (tp["p1"] - tp["p0"]).to_numpy()


def change_hands(benefit, risk, true_cate):
    # BEGIN SOLUTION
    b, r = U.top_share(benefit).astype(bool), U.top_share(risk).astype(bool)
    return float((b & ~r).sum() / b.sum()), float(true_cate[b].mean()), float(true_cate[r].mean())
    # END SOLUTION


try:
    moved, g_b, g_r = change_hands(benefit, risk, true_cate)
except NotImplementedError:
    moved, g_b, g_r = U.reference("d2_4", "change")
U.keep("d2_4", "change", [moved, g_b, g_r])
print(f"{moved:.0%} of places change hands · true gain: benefit fifth {U.pp(g_b)}, risk fifth {U.pp(g_r)}")
print(f"per 1,000 places: about {1000 * (g_b - g_r):.0f} more seekers in work within six months")
U.check(g_b > g_r, "the switch raises the true gain")

# %% [markdown]
# ## ✱ Stretch · Three budgets
#
# Extra employed per hundred places if the top 10%, 20% or 30% by estimated benefit are coached (observed on the held-out pilot). In expectation the number falls as the budget grows, because each extra place goes to a seeker with a smaller estimated benefit. Does it fall here? What does that say about the noise in 480 seekers?

# %%
def stretch():
    # BEGIN SOLUTION
    for q in (0.1, 0.2, 0.3):
        print(f"budget {q:.0%}: {100 * U.uplift_at(benefit, y, t, q):.1f} extra employed per 100 places")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does every Qini curve end at the same point?
#    <details><summary>Answer</summary>At 100% every ranking coaches everyone, so the extra employed equal the whole effect.</details>
# 2. The benefit curve is above the random line everywhere. Does that prove the estimator is calibrated?
#    <details><summary>Answer</summary>No. A Qini curve checks ranking, not scale; Lab 2.3's calibration checks scale.</details>
# 3. Why is a threshold better than "the top fifth" for next year's rule?
#    <details><summary>Answer</summary>Next year's seekers arrive one at a time; a fixed threshold decides each one, and the agency can watch whether it fills one place in five.</details>
#
# ## Carry forward
#
# Day 3 values rules like this one on the logs, before using them. Its labs start from `data/reference/cate_logged.csv.gz`.
