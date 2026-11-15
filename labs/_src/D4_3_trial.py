# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.13", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 4.3 · Size the trial
#
# Day 4 · Session 3 (The confirmatory trial) · 25 min.
#
# The offline value of the depth-two rule is a hypothesis: it rests on recorded probabilities, an outcome model and a population that will change. Before Fursa adopts it, Wijdan wants a trial that would confirm it against current practice, and Majd Al-Mazroa in finance wants to know how many seekers and how long. You state the estimand and the arms, find how much the two policies disagree, size the trial, choose a randomisation unit and a stopping rule, and write the protocol table.
#
# **AI policy.** AI-tutor: an assistant may explain a function or an error; it does not write the analysis.
#
# **Reference state.** `data/reference/rule_tree.json` (the rule), the logs without RUM-2 with `data/reference/cate_logged.csv.gz`. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · The estimand and the arms (4')
# - [ ] 2 · Where the two policies disagree (6')
# - [ ] 3 · Seekers per arm, then the design effect (8')
# - [ ] 4 · A stopping rule and the protocol table (7')
# - [ ] ✱ Stretch · Simulate the trial to check its power

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

import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import fursa as U
print("mode:", U.MODE)

L = U.logged(drop_override=True)
R = U.supplied("cate_logged.csv.gz").set_index("decision_id").loc[L["decision_id"]].reset_index()
y, a, p = L["employed_6m"].to_numpy(), L["offered"].to_numpy(), L["p_offer"].to_numpy()
mu0, mu1 = R["mu0"].to_numpy(), R["mu1"].to_numpy()
rule = U.predict_tree(json.loads((U.REF / "rule_tree.json").read_text(encoding="utf-8")), L)
PER_YEAR = int((L["year"] == 2026).sum() / 0.9)   # registrations a year, all offices (RUM-2 is about 10%)
print(f"{len(L):,} decisions · the rule offers {rule.mean():.1%} · about {PER_YEAR:,} registrations a year")

# %% [markdown]
# ## 1 · The estimand and the arms (4')
#
# The question is not "does coaching work" (the pilot answered that) but "do seekers decided by the rule find work more often than seekers decided by current practice?". Each arm is a policy: a seeker is randomised to an arm, then the arm's policy decides whether she is offered a place.
#
# ✏️ **Exercise** · Fill the protocol's first rows: `unit`, `arms`, `primary_outcome`, `estimand`.

# %%
def first_rows():
    # BEGIN SOLUTION
    return {"unit": "a newly registered job seeker, randomised at registration, stratified by office",
            "arms": "A: the depth-two rule decides · B: current practice (the caseworker, with the allocation tool) decides",
            "primary_outcome": "employed within six months of registration, from the social insurance register",
            "estimand": "difference in the employment rate between arm A and arm B, among all randomised seekers (intention to treat)"}
    # END SOLUTION


try:
    PROTO = first_rows()
except NotImplementedError:
    PROTO = U.reference("d4_3", "first")
U.keep("d4_3", "first", PROTO)
for k, v in PROTO.items():
    print(f"{k:16s} {v}")
U.check(all(PROTO.get(k) for k in ["unit", "arms", "primary_outcome", "estimand"]), "the four rows are filled")
U.check("current" in PROTO["arms"].lower(), "current practice is the comparison arm")

# %% [markdown]
# ## 2 · Where the two policies disagree (6')
#
# Where the rule and the caseworker make the same decision, the two arms cannot differ. The whole difference comes from the seekers on whom they disagree, so the effect the trial must detect is diluted.
#
# ✏️ **Exercise** · On the logs, return: the share of seekers on whom the rule and current practice disagree; current practice's employment rate (`p_B`); the rule's augmented value (`p_A`); and the difference among the disagreeing seekers, `(p_A - p_B) / share`.

# %%
def disagreement():
    # BEGIN SOLUTION
    share = float((rule != a).mean())
    p_B = float(y.mean())
    p_A = U.dr_value(y, a, p, rule, mu0, mu1)[0]
    return share, p_B, p_A, (p_A - p_B) / share
    # END SOLUTION


try:
    share, p_B, p_A, diff_dis = disagreement()
except NotImplementedError:
    share, p_B, p_A, diff_dis = U.reference("d4_3", "disagree")
U.keep("d4_3", "disagree", [share, p_B, p_A, diff_dis])
print(f"disagree on {share:.1%} of seekers · current practice {p_B:.1%} · rule {p_A:.1%} · difference {U.pp(p_A - p_B)} overall, "
      f"{U.pp(diff_dis)} among the seekers they disagree on")
both = pd.crosstab(pd.Series(rule, name="rule offers"), pd.Series(a, name="caseworker offered"), normalize=True).round(3)
print(both.to_string())
U.check(0.15 < share < 0.4, "the policies disagree on a minority of seekers")
U.check(p_A - p_B < 0.03, "the overall difference to detect is under 3 points")

# %% [markdown]
# **Look at** the crosstab. Each policy offers about one seeker in six, but mostly to different seekers: the rule to the 11-to-15-month band, the caseworker to the hardest cases. A difference of 5 points among a quarter of seekers is a difference of about 1.4 points among everyone.
#
# ## 3 · Seekers per arm, then the design effect (8')
#
# `U.n_per_arm(p_B, p_A)` gives the seekers per arm for a two-sided test at 5% with 80% power.
#
# If whole offices were randomised instead of seekers (to stop caseworkers mixing the two policies), seekers in an office would share its labour market: the variance grows by the design effect `1 + (m - 1) ICC`, with `m` seekers per office per arm and `ICC` the intra-office correlation of employment.
#
# ✏️ **Exercise** · Return seekers per arm, the years of registrations the trial would need, the ICC estimated from the logs (the variance of office employment rates over `p(1-p)`), and the design effect for office randomisation with 11 offices.

# %%
def sizes():
    # BEGIN SOLUTION
    n = U.n_per_arm(p_B, p_A)
    years = 2 * n / PER_YEAR
    rates = L.groupby("office")["employed_6m"].mean()
    icc = float(rates.var() / (y.mean() * (1 - y.mean())))
    m = n / (11 / 2)                     # seekers per office if offices, not seekers, are randomised
    deff = 1 + (m - 1) * icc
    return n, years, icc, deff
    # END SOLUTION


try:
    n, years, icc, deff = sizes()
except NotImplementedError:
    n, years, icc, deff = U.reference("d4_3", "sizes")
U.keep("d4_3", "sizes", [n, years, icc, deff])
print(f"{n:,} seekers per arm · {years:.1f} years of registrations · ICC {icc:.3f} · design effect for office randomisation {deff:.0f}")
print(f"if the trial could enrol only the disagreeing seekers: {U.n_per_arm(p_B, p_B + diff_dis):,} per arm")
U.check(n > 5000, "the diluted difference needs thousands of seekers per arm")
U.check(deff > 10, "randomising offices would multiply the sample by more than ten")

# %% [markdown]
# **Look at** the two answers. Randomising seekers within offices needs about two years of Fursa's registrations; randomising its 11 offices could never be powered. So seekers are randomised, stratified by office, and caseworkers are told which policy decides for each seeker, with a check that they follow it. Two threats remain. **Contamination**: a caseworker who sees the rule may change how she treats arm B. **Displacement**: coached seekers may take jobs other seekers would have found. Crépon and colleagues found exactly this in a large French job-search trial, with gains for the coached partly offset by losses for others in the same labour markets; a seeker-level trial cannot see it, so the protocol names it as a limit.
#
# ## 4 · A stopping rule and the protocol table (7')
#
# One interim look at half the seekers lets the trial stop early if the rule is clearly better or clearly harmful. Looking twice at 5% each would raise the false-positive rate; O'Brien–Fleming boundaries spend almost nothing at the first look. With looks at information fractions 0.5 and 1, the boundaries are `c / sqrt(t)`: find `c` so that the chance of crossing either boundary under no difference is 5%.
#
# ✏️ **Exercise** · Find `c` by simulation (200,000 pairs of correlated z-statistics: `z1 = W(0.5)/sqrt(0.5)`, `z2 = W(1)`), return the two boundaries and complete the protocol table.

# %%
def obf(sims=U.pick(200_000, 50_000)):
    # BEGIN SOLUTION
    r = U.rng(43)
    w1 = r.normal(0, np.sqrt(0.5), sims)
    w2 = w1 + r.normal(0, np.sqrt(0.5), sims)
    z1, z2 = np.abs(w1 / np.sqrt(0.5)), np.abs(w2)
    lo, hi = 1.9, 3.0
    for _ in range(40):
        c = (lo + hi) / 2
        alpha = np.mean((z1 > c / np.sqrt(0.5)) | (z2 > c))
        lo, hi = (c, hi) if alpha > 0.05 else (lo, c)
    return c / np.sqrt(0.5), c
    # END SOLUTION


try:
    b1, b2 = obf()
except NotImplementedError:
    b1, b2 = U.reference("d4_3", "obf")
U.keep("d4_3", "obf", [b1, b2])
print(f"O'Brien-Fleming boundaries: |z| > {b1:.2f} at half the seekers, |z| > {b2:.2f} at the end")
U.check(2.6 < b1 < 3.0 and 1.95 < b2 < 2.05, "the boundaries match the published values (about 2.80 and 1.98)")

PROTO.update({
    "minimum difference": f"{U.pp(p_A - p_B)} overall (the offline estimate; {U.pp(diff_dis)} among disagreeing seekers)",
    "seekers per arm": f"{int(np.ceil(n * 1.01)):,} (1% added for the interim look)",
    "duration": f"about {years * 1.01:.1f} years of registrations, plus six months of follow-up",
    "interim look": f"at half the seekers: stop if |z| > {b1:.2f}",
    "final analysis": f"|z| > {b2:.2f}; difference in employment rates with a 95% interval, intention to treat",
    "audit": "the same analysis by region, gender and disability status (Lab 4.2), reported whatever the result",
    "limits": "displacement between coached and uncoached seekers is not measured; contamination checked by caseworker compliance",
})
for k, v in PROTO.items():
    print(f"{k:19s} {v}")
U.check(len(PROTO) >= 11, "the protocol table has every row")

# %% [markdown]
# ## ✱ Stretch · Simulate the trial to check its power
#
# Draw the trial 2,000 times with employment rates `p_B` and `p_A` and `n` seekers per arm, apply the stopping rule, and count how often it ends with |z| above the boundary at either look.

# %%
def stretch():
    # BEGIN SOLUTION
    r = U.rng(44)
    sims, half = U.pick(2000, 500), n // 2
    hits = 0
    for _ in range(sims):
        A, B = r.random(n) < p_A, r.random(n) < p_B
        def z(k):
            pa, pb = A[:k].mean(), B[:k].mean()
            pbar = (pa + pb) / 2
            return (pa - pb) / np.sqrt(2 * pbar * (1 - pbar) / k)
        hits += abs(z(half)) > b1 or abs(z(n)) > b2
    print(f"simulated power: {hits / sims:.1%} (target 80%; the interim look costs a little power without the 1% added)")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is the difference to detect smaller than the rule's gain per place?
#    <details><summary>Answer</summary>Both arms coach about one seeker in six, and on most seekers the two policies agree; only the disagreeing quarter can differ.</details>
# 2. Why not randomise whole offices?
#    <details><summary>Answer</summary>With 11 offices, the design effect is in the hundreds: the trial could never be powered. Seekers are randomised within offices instead, with a compliance check.</details>
# 3. What can a seeker-level trial not tell you?
#    <details><summary>Answer</summary>Whether coached seekers displace others in the same labour market: the comparison is between individuals who share that market.</details>
#
# ## Carry forward
#
# Lab 4.4 puts the rule, its value, its audit and this protocol on one page.
