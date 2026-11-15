# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 3.4 · Four seeded faults
#
# Day 3 · Session 4 (Reviewing an effects claim) · 25 min.
#
# Anwar Al-Mohaimeed, an analyst in programme analytics, has drafted a subgroup analysis of the pilot for the director. It recommends whom to coach first. It contains four faults, each of a kind the course has named. You find each one, re-test the claim it supports on data or with a method that avoids the fault, and write a review memo that names each fault and its effect on the claim.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the review.
#
# **Reference state.** `data/review/anwar_analysis.md` and `data/pilot.csv`, loaded fresh. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Read the claim and list what it rests on (4')
# - [ ] 2 · The post-treatment subgroup (5')
# - [ ] 3 · The dredged cell on held-out data (6')
# - [ ] 4 · Multiplicity and the cells with no effect (5')
# - [ ] 5 · In-sample against held-out agreement (5')
# - [ ] ✱ Stretch · Tag each fault with its name in the national bias guide

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

memo = (U.DATA / "review" / "anwar_analysis.md").read_text(encoding="utf-8")
print(memo)
P = U.pilot()
y, t = P["employed_6m"].to_numpy(), P["coached"].to_numpy()
tp = U.truth_pilot()
true_cate = (tp["p1"] - tp["p0"]).to_numpy()
memo_facts = U.truth()["review"]          # the numbers the analyst reported

# %% [markdown]
# ## 1 · Read the claim and list what it rests on (4')
#
# ✏️ **Exercise** · Return a list of four short strings, one per finding, each naming the step of the analysis the finding depends on (for example "a subgroup defined by sessions attended").

# %%
def rests_on():
    # BEGIN SOLUTION
    return ["finding 1: a subgroup defined by sessions attended, which only coached seekers have",
            "finding 2: one cell chosen as the largest of 60 searched, estimated on the same seekers",
            "finding 3: keeping cells by p < 0.05 out of 60 tests, and reading the rest as no effect",
            "finding 4: predicted and observed effects compared on the seekers the model was fitted to"]
    # END SOLUTION


try:
    basis = rests_on()
except NotImplementedError:
    basis = U.reference("d3_4", "basis")
U.keep("d3_4", "basis", basis)
print("\n".join(basis))
U.check(len(basis) == 4, "one line per finding")

# %% [markdown]
# ## 2 · The post-treatment subgroup (5')
#
# Sessions attended exist only for coached seekers and are decided after the lottery, by the seeker. Compare the coached seekers who attended four or more sessions with those who attended fewer: do they differ in `motivation`, measured before the lottery?
#
# ✏️ **Exercise** · Return the mean motivation of coached seekers with four or more sessions and with fewer, and the intention-to-treat effect (all coached minus all uncoached).

# %%
def post_treatment(P):
    # BEGIN SOLUTION
    c = P[P.coached == 1]
    hi, lo = c.loc[c.coaching_sessions >= 4, "motivation"].mean(), c.loc[c.coaching_sessions < 4, "motivation"].mean()
    itt = P.loc[P.coached == 1, "employed_6m"].mean() - P.loc[P.coached == 0, "employed_6m"].mean()
    return float(hi), float(lo), float(itt)
    # END SOLUTION


try:
    mot_hi, mot_lo, itt = post_treatment(P)
except NotImplementedError:
    mot_hi, mot_lo, itt = U.reference("d3_4", "post")
U.keep("d3_4", "post", [mot_hi, mot_lo, itt])
print(f"motivation: {mot_hi:.2f} (4+ sessions) against {mot_lo:.2f} (fewer) · claimed effect {U.pp(memo_facts['f1_effect'])} · "
      f"intention-to-treat effect {U.pp(itt)}")
U.check(mot_hi > mot_lo + 0.2, "attenders were more motivated before the lottery: the subgroup is selected, not randomised")

# %% [markdown]
# ## 3 · The dredged cell on held-out data (6')
#
# The analyst searched 60 cells on all 6,000 seekers and reported the largest. Repeat the search honestly: search only the training split, take its five largest cells, and estimate the same five cells on the held-out split. (The analyst's own cell cannot be re-tested on held-out seekers: his search used them. The generator's truth judges it below.)
#
# ✏️ **Exercise** · Return the mean effect of the training search's top five cells on training seekers, and their mean effect on held-out seekers. `U.review_cells(df)` does the search; `U.in_cell(df, cell)` finds a cell's rows.

# %%
TR, TE = P[P.split == "train"].reset_index(drop=True), P[P.split == "test"].reset_index(drop=True)


def dredge_check():
    # BEGIN SOLUTION
    top = U.review_cells(TR).sort_values("effect", ascending=False).head(5)
    held = []
    for _, c in top.iterrows():
        m = U.in_cell(TE, c)
        held.append(TE.loc[m & (TE.coached == 1), "employed_6m"].mean() - TE.loc[m & (TE.coached == 0), "employed_6m"].mean())
    return float(top["effect"].mean()), float(np.nanmean(held))
    # END SOLUTION


try:
    win_train, win_held = dredge_check()
except NotImplementedError:
    win_train, win_held = U.reference("d3_4", "dredge")
U.keep("d3_4", "dredge", [win_train, win_held])
cell = memo_facts["f2"]
true_cell = true_cate[U.in_cell(P, cell)].mean()
print(f"top five cells of the training search: {U.pp(win_train)} on training, {U.pp(win_held)} on held-out seekers")
print(f"the analyst's cell ({cell['n']} seekers): claimed {U.pp(cell['effect'])}, true {U.pp(true_cell)}")
U.check(win_held < win_train - 0.10, "the winners of a search shrink by more than 10 points on fresh seekers")
U.check(true_cell < cell["effect"] / 2, "the analyst's cell's true effect is less than half the claim")

# %% [markdown]
# ## 4 · Multiplicity and the cells with no effect (5')
#
# Sixty tests at 0.05 would give about three "significant" cells even if coaching did nothing anywhere. And a cell without significance is not a cell without effect: most cells hold 30 to 60 seekers.
#
# ✏️ **Exercise** · Return the number of cells significant after a Bonferroni correction, and the mean **true** effect in the cells the analyst said showed no effect (p of 0.05 or more).

# %%
def multiplicity():
    # BEGIN SOLUTION
    cells = U.review_cells(P)
    bonf = int((cells.p < 0.05 / len(cells)).sum())
    quiet = cells[cells.p >= 0.05]
    effects = [true_cate[U.in_cell(P, r)].mean() for _, r in quiet.iterrows()]
    return bonf, float(np.mean(effects))
    # END SOLUTION


try:
    bonf, quiet_true = multiplicity()
except NotImplementedError:
    bonf, quiet_true = U.reference("d3_4", "multiplicity")
U.keep("d3_4", "multiplicity", [bonf, quiet_true])
print(f"{memo_facts['cells_significant']} of {memo_facts['cells']} cells at p < 0.05 · after Bonferroni: {bonf} · "
      f"true effect in the 'no effect' cells: {U.pp(quiet_true)}")
U.check(bonf <= 2, "almost nothing survives a correction for 60 tests")
U.check(quiet_true > 0.05, "the 'no effect' cells have a real effect of more than 5 points")

# %% [markdown]
# ## 5 · In-sample against held-out agreement (5')
#
# The analyst's model saw every seeker it was then checked on. Fit the same kind of model (gradient boosting, coaching as a feature, the course's lightly regularised `U.gbm_classifier`) twice: once on all seekers and checked on all; once on the training split and checked on the held-out split. Compare predicted and observed effects across deciles of the prediction.
#
# ✏️ **Exercise** · Return the two decile correlations.

# %%
X = U.design(P).to_numpy()
tr_m = (P.split == "train").to_numpy()


def s_effects(fit_rows, X):
    m = clone(U.gbm_classifier(7)).fit(np.column_stack([X[fit_rows], t[fit_rows]]), y[fit_rows])
    return m.predict_proba(np.column_stack([X, np.ones(len(X))]))[:, 1] - m.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1]


def decile_corr(pred, rows):
    tab = U.calibration_table(pred[rows], y[rows], t[rows], groups=10)
    return float(np.corrcoef(tab.predicted, tab.observed)[0, 1])


def in_vs_out():
    # BEGIN SOLUTION
    all_rows = np.ones(len(P), bool)
    r_in = decile_corr(s_effects(all_rows, X), all_rows)
    r_out = decile_corr(s_effects(tr_m, X), ~tr_m)
    return r_in, r_out
    # END SOLUTION


try:
    r_in, r_out = in_vs_out()
except NotImplementedError:
    r_in, r_out = U.reference("d3_4", "corr")
U.keep("d3_4", "corr", [r_in, r_out])
print(f"decile correlation: in sample {r_in:.2f} · held out {r_out:.2f}")
U.check(r_in > r_out, "agreement on the fitted seekers flatters the model")

# %% [markdown]
# **The review memo.** One entry per fault: what it is, the evidence from this notebook, and its effect on the claim.

# %%
def review_memo():
    # BEGIN SOLUTION
    return [
        {"fault": "post-treatment subgroup", "evidence": f"attenders' motivation {mot_hi:.2f} against {mot_lo:.2f} before the lottery",
         "effect": f"the {100 * memo_facts['f1_effect']:.0f}-point claim mixes coaching with motivation; the randomised effect is {100 * itt:.0f} points"},
        {"fault": "discovery and evaluation on the same data", "evidence": f"the top five cells of an honest search fall from {100 * win_train:.0f} to {100 * win_held:.0f} points on fresh seekers; the analyst's cell truly gains {100 * true_cell:.0f}",
         "effect": f"the {100 * cell['effect']:.0f}-point cell is mostly the noise that made it win the search"},
        {"fault": "multiplicity, and no significance read as no effect", "evidence": f"{bonf} cells survive Bonferroni; 'no effect' cells truly gain {100 * quiet_true:.0f} points",
         "effect": "the target list is largely chance, and the excluded cells would benefit"},
        {"fault": "in-sample validation", "evidence": f"decile correlation {r_in:.2f} in sample, {r_out:.2f} held out",
         "effect": "the agreement shown is the model's fit to its own data, not evidence that it ranks new seekers"},
    ]
    # END SOLUTION


try:
    review = review_memo()
except NotImplementedError:
    review = U.reference("d3_4", "memo")
U.keep("d3_4", "memo", review)
for r in review:
    print(f"- {r['fault']}: {r['evidence']}. Effect: {r['effect']}.")
U.check(len(review) == 4 and all({"fault", "evidence", "effect"} <= set(r) for r in review), "four faults, each with evidence and effect")

# %% [markdown]
# ## ✱ Stretch · The national bias guide
#
# SDAIA's AI Bias Reference Guide (first edition, May 2026) names more than a hundred biases. Which of its names fit faults 2 and 3? Look up "Data-dredging Bias" and "Evaluation Bias" in the guide.

# %%
def stretch():
    # BEGIN SOLUTION
    print("fault 2 and fault 3: data-dredging bias (searching many combinations and reporting the significant ones)")
    print("fault 4: evaluation bias (a model judged on data that do not represent its use)")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why can't sessions attended define a subgroup for an effect, even in a randomised pilot?
#    <details><summary>Answer</summary>It is measured after the lottery and only for the coached; the attenders are a self-selected group with no comparable uncoached group.</details>
# 2. The analyst's cell has p = 0.005. Why is that weak evidence?
#    <details><summary>Answer</summary>It is the smallest of 60 p-values from a search; the chance that the best of 60 looks this good by luck is high, and the estimate is inflated by the selection.</details>
# 3. What should the analysis have done instead?
#    <details><summary>Answer</summary>Pre-register a few subgroups, or discover on one split and test on another; report all cells with corrected intervals; validate any model on held-out randomised data.</details>
#
# ## Carry forward
#
# Day 4 learns a rule on the training split and values it on data it did not see.
