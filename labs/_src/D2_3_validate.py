# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4", "matplotlib>=3.9"]
# ///
# %% [markdown]
# # Lab 2.3 · Calibration and ranking per estimator
#
# Day 2 · Session 3 (Validating effects) · 25 min.
#
# Four estimators of each seeker's effect were fitted on the pilot's training split. None has been checked. The held-out seekers were also randomised, so within any group of them the coached-minus-uncoached difference is an honest estimate of that group's effect. You use them to check each estimator three ways: calibration by quintile, the best linear predictor slope and the gain in the top fifth. Then you choose one, with reasons. The truth is used only at the end, to see how the checks did.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/reference/cate_pilot.csv` (S-, T-, X-learner and causal forest predictions, fitted on the training split) and `data/pilot.csv`. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Calibration by quintile, per estimator (8')
# - [ ] 2 · The best linear predictor slope (5')
# - [ ] 3 · Ranking: the gain in the top fifth (6')
# - [ ] 4 · Choose and justify (6')
# - [ ] ✱ Stretch · Repeat the checks on bootstrap draws of the held-out split

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
Ch = C[te].reset_index(drop=True)
y, t = H["employed_6m"].to_numpy(), H["coached"].to_numpy()
EST = ["s_learner", "t_learner", "x_learner", "causal_forest"]
# Outcome predictions for the held-out seekers, cross-fitted among themselves: they make the effect score of step 2 less noisy
mu0, mu1 = U.outcome_models_crossfit(y, t, U.design(H), model=U.pilot_classifier(5))
print(f"held-out seekers: {len(H):,} · observed average effect {U.pp(y[t == 1].mean() - y[t == 0].mean())}")

# %% [markdown]
# ## 1 · Calibration by quintile, per estimator (8')
#
# Sort the held-out seekers into five groups by an estimator's prediction. In each group compare the mean predicted effect with the observed one (coached minus uncoached). `U.calibration_table(pred, y, t)` does one estimator.
#
# ✏️ **Exercise** · Return a dict from estimator name to its calibration table.

# %%
def calibrate_all(Ch, y, t):
    # BEGIN SOLUTION
    return {name: U.calibration_table(Ch[name].to_numpy(), y, t, groups=5) for name in EST}
    # END SOLUTION


try:
    cal = calibrate_all(Ch, y, t)
except NotImplementedError:
    cal = {name: U.calibration_table(Ch[name].to_numpy(), y, t, groups=5) for name in EST}
    print("↪ using the course's loop: finish the exercise to use your own")
    U.check(not U.STRICT, "the solution run used its own loop")
for name, tab in cal.items():
    print(f"\n{name}\n" + (tab.set_index("group")[["predicted", "observed", "se"]] * 100).round(1).to_string())
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for a, (name, tab) in zip(ax, cal.items()):
    a.errorbar(100 * tab.predicted, 100 * tab.observed, yerr=196 * tab.se, fmt="o")
    lim = [-5, 25]; a.plot(lim, lim, "k--", lw=1); a.set_title(name); a.set_xlabel("predicted (points)")
ax[0].set_ylabel("observed (points)")
fig.tight_layout(); fig.savefig(U.artifacts("d2_3") / "calibration.png", dpi=120)
U.check(all(len(v) == 5 for v in cal.values()), "five groups for each estimator")

# %% [markdown]
# **Look at** two shapes. An estimator whose predicted range is narrow while the observed effects climb steeply is too timid: it ranks well but understates differences. One whose top group predicts more than it delivers is overconfident. Each group holds about 480 seekers, so an observed effect has a standard error near 4 points.

# %% [markdown]
# ## 2 · The best linear predictor slope (5')
#
# Regress a per-seeker doubly robust effect score on the centred prediction (`U.blp_slope(pred, y, t, 0.5, mu0, mu1)`). A slope near 1: predictions carry the right amount of signal. Near 0: no signal. Above 1: the predictions are too timid.
#
# ✏️ **Exercise** · Return a table with `slope` and `se` per estimator, using `mu0` and `mu1` from the setup cell.

# %%
def blp_all(Ch, y, t):
    # BEGIN SOLUTION
    rows = {name: U.blp_slope(Ch[name].to_numpy(), y, t, 0.5, mu0, mu1) for name in EST}
    return pd.DataFrame(rows, index=["slope", "se"]).T
    # END SOLUTION


try:
    blp = blp_all(Ch, y, t)
except NotImplementedError:
    blp = U.reference("d2_3", "blp")
U.keep("d2_3", "blp", blp.round(4))
print(blp.round(2).to_string())
U.check((blp["slope"] - 1.96 * blp["se"] > 0).all(), "every estimator carries signal (slope interval above 0)")
U.check(blp.loc["causal_forest", "slope"] > 1.5, "the causal forest is too timid: its slope is well above 1")

# %% [markdown]
# ## 3 · Ranking: the gain in the top fifth (6')
#
# The budget is one place per five seekers, so what matters most is whether the top fifth by an estimator really gains more. `U.uplift_at(pred, y, t, 0.2)` gives the observed effect in that fifth.
#
# ✏️ **Exercise** · Return a Series: the observed effect in the top fifth, per estimator, and add the same for the agency's `risk_score`.

# %%
def top_fifth_all(Ch, H, y, t):
    # BEGIN SOLUTION
    out = {name: U.uplift_at(Ch[name].to_numpy(), y, t, 0.2) for name in EST}
    out["risk_score"] = U.uplift_at(H["risk_score"].to_numpy(), y, t, 0.2)
    return pd.Series(out)
    # END SOLUTION


try:
    top = top_fifth_all(Ch, H, y, t)
except NotImplementedError:
    top = U.reference("d2_3", "top")
U.keep("d2_3", "top", top.round(4))
print((top * 100).round(1).to_string(), "\n(points, observed in the top fifth)")
U.check(top.drop("risk_score").max() > top["risk_score"], "an effect estimator beats the risk score in the top fifth")

# %% [markdown]
# ## 4 · Choose and justify (6')
#
# ✏️ **Exercise** · Return the estimator you would use for targeting and two sentences of reasons that cite the three checks.

# %%
def choose():
    # BEGIN SOLUTION
    best = "x_learner"
    return best, (f"The X-learner's top fifth gains {100 * top[best]:.1f} points on held-out seekers against "
                  f"{100 * top['risk_score']:.1f} for the risk score, its best-linear-predictor slope is {blp.loc[best, 'slope']:.2f} "
                  "(near 1: right in scale as well as rank), and its quintiles rise in order. The causal forest ranks almost as well "
                  "but its slope near 3 says it understates differences, which matters when places are priced against a cost.")
    # END SOLUTION


try:
    choice, why = choose()
except NotImplementedError:
    choice, why = U.reference("d2_3", "choice")
U.keep("d2_3", "choice", [choice, why])
print(choice, "·", why)
U.check(choice in EST and abs(blp.loc[choice, "slope"] - 1) < 2 * blp.loc[choice, "se"] and top[choice] >= top.drop("risk_score").max() - 0.03,
        "the chosen estimator is calibrated in scale and among the best in its top fifth")

# %% [markdown]
# **Now look at the truth**, once. How did the checks rank the estimators compared with their real errors?

# %%
tp = U.truth_pilot()[te].reset_index(drop=True)
true_cate = (tp["p1"] - tp["p0"]).to_numpy()
truth_tab = pd.DataFrame({name: {"pehe": np.sqrt(np.mean((Ch[name] - true_cate) ** 2)),
                                 "corr": np.corrcoef(Ch[name], true_cate)[0, 1],
                                 "true gain, top fifth": true_cate[U.top_share(Ch[name], 0.2) == 1].mean()} for name in EST}).T
print(truth_tab.round(3).to_string())
truth_tab.to_csv(U.artifacts("d2_3") / "truth_check.csv")

# %% [markdown]
# ## ✱ Stretch · Bootstrap draws
#
# Resample the held-out seekers 200 times (50 in quick mode) and recompute the top-fifth gain per estimator. How often does your choice stay first?

# %%
def stretch():
    # BEGIN SOLUTION
    r = U.rng(23)
    wins = pd.Series(0, index=EST)
    for _ in range(U.pick(200, 50)):
        i = r.integers(0, len(H), len(H))
        g = {name: U.uplift_at(Ch[name].to_numpy()[i], y[i], t[i], 0.2) for name in EST}
        wins[max(g, key=g.get)] += 1
    print((wins / wins.sum()).round(2).to_string())
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why must validation use seekers the estimator did not see?
#    <details><summary>Answer</summary>On its own training data a flexible estimator's groups look better than they are: it found the patterns there, noise included.</details>
# 2. The causal forest's predictions span only a few points, yet its top fifth gains much more than average. Is it useless?
#    <details><summary>Answer</summary>Not for ranking: its groups rise in order. It is miscalibrated in scale (too timid). For a fixed budget, ranking matters most; for a cost-benefit case, scale matters too.</details>
# 3. Why can these checks use simple differences in means?
#    <details><summary>Answer</summary>The held-out seekers were randomised 1:1, so within any group defined by their covariates the coached and uncoached are comparable.</details>
#
# ## Carry forward
#
# Lab 2.4 draws uplift and Qini curves from the causal forest's predictions in the same file.
