# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4", "doubleml>=0.11"]
# ///
# %% [markdown]
# # Lab 1.4 · Swap learners, bound a hidden confounder
#
# Day 1 · Session 4 (Nuisance models and hidden confounding) · 25 min.
#
# Double machine learning is only as good as two things: the nuisance models, and the confounders in the data. You first re-estimate the average effect with four nuisance learners and set each estimate beside its prediction losses. Then you take the 2027 extract, which lost the caseworkers' motivation note in a system change, see the estimate move, and bound it with DoubleML's sensitivity analysis.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/logged.csv.gz`, loaded fresh (10,000 decisions in quick mode); the extract is the same file without the `motivation` column. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Four nuisance learners, one table (8')
# - [ ] 2 · Nuisance losses beside each estimate (4')
# - [ ] 3 · The extract without motivation (5')
# - [ ] 4 · Bound it with a sensitivity analysis (8')
# - [ ] ✱ Stretch · Benchmark the bound with an observed covariate

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import doubleml as dml
import fursa as U
print("mode:", U.MODE)

L = U.logged_sample()
X = U.design(L, U.CONFOUNDERS)
y, t = L["employed_6m"].to_numpy(), L["offered"].to_numpy()
tr = U.truth_for(L["decision_id"])
ATE = float((tr["p1"] - tr["p0"]).mean())
print(f"{len(L):,} decisions · true average effect {U.pp(ATE)}")

LEARNERS = {
    "lasso logistic": make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", C=0.05, solver="liblinear")),
    "random forest": RandomForestClassifier(n_estimators=U.pick(200, 60), min_samples_leaf=20, n_jobs=-1, random_state=U.SEED),
    "gradient boosting": U.gbm_classifier(),
    "one split": DecisionTreeClassifier(max_depth=1),
}

# %% [markdown]
# ## 1 · Four nuisance learners, one table (8')
#
# The same cross-fitted score as Lab 1.3, four times. `U.crossfit_aipw(y, t, X, model_y, model_t)` returns the estimate, its standard error and the cross-fitted predictions `e`, `mu0`, `mu1`.
#
# ✏️ **Exercise** · Return a dict from learner name to the result of `U.crossfit_aipw`, using the same learner for the outcome and the propensity.

# %%
def run_learners(y, t, X):
    # BEGIN SOLUTION
    return {name: U.crossfit_aipw(y, t, X, m, m) for name, m in LEARNERS.items()}
    # END SOLUTION


try:
    runs = run_learners(y, t, X)
except NotImplementedError:
    runs = {name: U.crossfit_aipw(y, t, X, m, m) for name, m in LEARNERS.items()}
    print("↪ using the course's loop: finish the exercise to use your own")
    U.check(not U.STRICT, "the solution run used its own loop")
est = pd.DataFrame({"estimate": {k: r["est"] for k, r in runs.items()}, "se": {k: r["se"] for k, r in runs.items()}})
est["bias"] = est["estimate"] - ATE
print((est * 100).round(2).to_string(), "\n(percentage points)")
U.check(est["bias"].abs().idxmax() == "one split", "the one-split learner is furthest from the truth")

# %% [markdown]
# ## 2 · Nuisance losses beside each estimate (4')
#
# A nuisance model is judged by how well it predicts, out of fold: log loss of `e` for the offer, and of `mu1` or `mu0` (whichever arm the decision is in) for employment.
#
# ✏️ **Exercise** · Add the two losses to the table.

# %%
def add_losses(est, runs):
    # BEGIN SOLUTION
    out = est.copy()
    for k, r in runs.items():
        out.loc[k, "loss_offer"] = log_loss(t, r["e"])
        out.loc[k, "loss_outcome"] = log_loss(y, np.clip(np.where(t == 1, r["mu1"], r["mu0"]), 1e-4, 1 - 1e-4))
    return out
    # END SOLUTION


try:
    est = add_losses(est, runs)
except NotImplementedError:
    est = U.reference("d1_4", "losses")
U.keep("d1_4", "losses", est)
print(est.assign(estimate=est.estimate * 100, se=est.se * 100, bias=est.bias * 100).round(3).sort_values("loss_offer").to_string())
U.check(est["loss_offer"].idxmin() in ("gradient boosting", "random forest"), "a flexible learner predicts the offer best")

# %% [markdown]
# **Look at** the order. Better nuisance predictions go with smaller bias here, but the link is loose: the estimate needs both nuisances good enough, and the sampling noise of one dataset is a point or more. Choose learners by their out-of-fold losses, decided before you look at the estimate.

# %% [markdown]
# ## 3 · The extract without motivation (5')
#
# The 2027 extract lost `motivation`, the caseworker's note of the seeker's motivation (1 to 5). Motivated seekers were offered coaching more often and found work more often.
#
# ✏️ **Exercise** · Re-estimate with gradient boosting on the extract and return the result of `U.crossfit_aipw`.

# %%
X27 = X.drop(columns=["motivation"])


def without_motivation(y, t, X27):
    # BEGIN SOLUTION
    return U.crossfit_aipw(y, t, X27, U.gbm_classifier(), U.gbm_classifier())
    # END SOLUTION


try:
    r27 = without_motivation(y, t, X27)
except NotImplementedError:
    r27 = U.crossfit_aipw(y, t, X27, U.gbm_classifier(), U.gbm_classifier())
    U.check(not U.STRICT, "the solution run used its own estimate")
full = runs["gradient boosting"]["est"]
print(f"with motivation {U.pp(full)} · without {U.pp(r27['est'])} (se {U.pp(r27['se'])}) · truth {U.pp(ATE)}")
U.check(r27["est"] - full > r27["se"] and r27["est"] > ATE, "without motivation the estimate moves up, away from the truth")

# %% [markdown]
# ## 4 · Bound it with a sensitivity analysis (8')
#
# DoubleML's sensitivity analysis asks: if a hidden confounder explained a share `cf_y` of the outcome's remaining variation and changed the treatment's in proportion `cf_d`, how far could the estimate move? It returns a lower and an upper bound and a robustness value: the strength, on both, at which the bound reaches zero.
#
# ✏️ **Exercise** · Fit `DoubleMLIRM` on the extract (gradient boosting for both learners, five folds), run `sensitivity_analysis(cf_y=0.03, cf_d=0.03)` and return the fitted object.

# %%
df27 = X27.assign(employed_6m=y, offered=t)
data27 = dml.DoubleMLData(df27, y_col="employed_6m", d_cols="offered", x_cols=list(X27.columns))


def bound(data27):
    # BEGIN SOLUTION
    np.random.seed(U.SEED)
    m = dml.DoubleMLIRM(data27, ml_g=U.gbm_classifier(), ml_m=U.gbm_classifier(), n_folds=5).fit()
    m.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
    return m
    # END SOLUTION


try:
    irm27 = bound(data27)
except NotImplementedError:
    np.random.seed(U.SEED)
    irm27 = dml.DoubleMLIRM(data27, ml_g=U.gbm_classifier(), ml_m=U.gbm_classifier(), n_folds=5).fit()
    irm27.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
    U.check(not U.STRICT, "the solution run used its own sensitivity analysis")
sp = irm27.sensitivity_params
lo, hi = float(sp["theta"]["lower"][0]), float(sp["theta"]["upper"][0])
rv = float(sp["rv"][0])
print(irm27.sensitivity_summary)
print(f"estimate {U.pp(irm27.coef[0])} · bound {U.pp(lo)} to {U.pp(hi)} · truth {U.pp(ATE)} · robustness value {rv:.1%}")
U.keep("d1_4", "bound", [lo, hi, rv])
U.check(lo < irm27.coef[0] < hi, "the bound brackets the estimate")
U.check(lo <= ATE, "a confounder of strength 0.03 on both is enough to reach the truth")

# %% [markdown]
# **Look at** what the bound says and does not say. It does not find the confounder; it tells Wijdan how strong something missing would have to be to change the answer. "Coaching helped" survives anything below the robustness value; "coaching raised employment by this many points" does not survive even a weak confounder.

# %% [markdown]
# ## ✱ Stretch · Benchmark with an observed covariate
#
# Is 0.03 a lot? `sensitivity_benchmark(["months_unemployed"])` refits without that covariate and reports the `cf_y` and `cf_d` it would have had as a hidden confounder. Rerun the analysis at those values.

# %%
def stretch():
    # BEGIN SOLUTION
    b = irm27.sensitivity_benchmark(benchmarking_set=["months_unemployed"])
    print(b.round(4))
    cf_y, cf_d = float(b["cf_y"].iloc[0]), float(b["cf_d"].iloc[0])
    irm27.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d)
    sp2 = irm27.sensitivity_params
    print(f"as strong as months unemployed: bound {U.pp(sp2['theta']['lower'][0])} to {U.pp(sp2['theta']['upper'][0])}")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. The random forest and gradient boosting estimates differ by a point. Which do you report?
#    <details><summary>Answer</summary>The one chosen before looking, by out-of-fold nuisance losses, and the other as a robustness check. Picking the estimate you like is a forking path.</details>
# 2. Could a better learner on the 2027 extract remove the bias?
#    <details><summary>Answer</summary>No. The information is not in the data; no model of the remaining columns can adjust for it.</details>
# 3. What does a robustness value of a few percent tell the director?
#    <details><summary>Answer</summary>That a hidden confounder explaining a few percent of the remaining variation in both the offer and employment would reduce the estimate to zero. Whether one could exist is a question for the caseworkers, not the model.</details>
#
# ## Carry forward
#
# Day 2 leaves the logs for the randomised pilot, where no confounder, hidden or not, can bias the comparison.
