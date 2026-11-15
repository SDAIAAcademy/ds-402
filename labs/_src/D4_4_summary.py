# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 4.4 · One-page summary and the capstone brief
#
# Day 4 · Session 4 (Explaining a rule) · 25 min.
#
# Wijdan will read one page before she decides, and she will repeat parts of it to the agency's board. You gather the numbers of the rule, write the page, list the cases a caseworker should review by hand, and choose your capstone brief.
#
# **AI policy.** AI-tutor: an assistant may comment on your draft's clarity; it does not write the summary, and every number in it comes from this notebook.
#
# **Reference state.** `data/reference/rule_tree.json` (the rule), the logs without RUM-2 with `data/reference/cate_logged.csv.gz`, and the briefs in `capstone/`. Nothing from another lab is needed; the numbers are recomputed here.
#
# **Tasks**
# - [ ] 1 · Gather the numbers (5')
# - [ ] 2 · Write the rule summary (10')
# - [ ] 3 · Edge cases for human review (5')
# - [ ] 4 · Choose the capstone brief (5')
# - [ ] ✱ Stretch · Cut the summary to what the director would repeat

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

import json, re
import numpy as np
import pandas as pd
import fursa as U
print("mode:", U.MODE)

TREE = json.loads((U.REF / "rule_tree.json").read_text(encoding="utf-8"))
L = U.logged(drop_override=True)
R = U.supplied("cate_logged.csv.gz").set_index("decision_id").loc[L["decision_id"]].reset_index()
y, a, p = L["employed_6m"].to_numpy(), L["offered"].to_numpy(), L["p_offer"].to_numpy()
mu0, mu1 = R["mu0"].to_numpy(), R["mu1"].to_numpy()
rule = U.predict_tree(TREE, L)
print(U.tree_text(TREE))

# %% [markdown]
# ## 1 · Gather the numbers (5')
#
# ✏️ **Exercise** · Return a dict with: `share` offered; `value` and its 95% interval (`lo`, `hi`) on the logs; `current` (current practice); `gain` in points; `jobs_per_1000` (extra seekers employed per 1,000 seekers compared with current practice); the lowest region's selection rate over the highest (`region_ratio`); and `left_out`, the share of seekers unemployed more than 15 months.

# %%
def gather():
    # BEGIN SOLUTION
    v, s = U.dr_value(y, a, p, rule, mu0, mu1)
    rates = pd.Series(rule).groupby(L["region"].to_numpy()).mean()
    return {"share": float(rule.mean()), "value": v, "lo": v - 1.96 * s, "hi": v + 1.96 * s, "current": float(y.mean()),
            "gain": v - float(y.mean()), "jobs_per_1000": 1000 * (v - float(y.mean())),
            "region_ratio": float(rates.min() / rates.max()), "left_out": float((L["months_unemployed"] > 15).mean())}
    # END SOLUTION


try:
    F = gather()
except NotImplementedError:
    F = U.reference("d4_4", "facts")
U.keep("d4_4", "facts", F)
for k, v in F.items():
    print(f"{k:14s} {v:.4f}")
U.check(F["lo"] > F["current"], "the rule's interval lies above current practice")
U.check(F["region_ratio"] > 0.8, "every region is offered within 20% of the highest rate")

# %% [markdown]
# ## 2 · Write the rule summary (10')
#
# One page, five headings, in this order: **What the rule does** (the rule in a sentence, the share offered), **What it is worth** (value with interval against current practice, in jobs), **Who it leaves out** (the audit, and who is outside the band), **Who reviews** (the edge cases of Task 3), **What would confirm it** (the trial of Lab 4.3, one line). No term a director would need explained: say "places", "seekers", "employed within six months", not "policy value" or "CATE".
#
# ✏️ **Exercise** · Return the page as Markdown, built from `F` so every number is computed.

# %%
def summary(F):
    # BEGIN SOLUTION
    return f"""# Coaching places in 2027: a proposed rule

## What the rule does
Offer a coaching place to job seekers who have been unemployed for more than 10 and at most 15 months. This would offer {F['share']:.0%} of seekers a place, within the budget of one place per five seekers; the remaining places can go to the seekers with the highest estimated gain in each region.

## What it is worth
On the 2025 and 2026 records, seekers decided this way would be employed within six months at a rate of {F['value']:.1%} (95% interval {F['lo']:.1%} to {F['hi']:.1%}), against {F['current']:.1%} under current practice: about {F['jobs_per_1000']:.0f} more seekers in work for every 1,000 registered. Giving places to the highest risk scores would do no better than current practice.

## Who it leaves out
Each region, women and men, and seekers with and without a disability are offered places at similar rates (the lowest region's rate is {F['region_ratio']:.0%} of the highest). Seekers unemployed for more than 15 months ({F['left_out']:.0%} of all seekers, and half of Nijad's) are not offered a place by the rule: the pilot found coaching helps them less, and other services should serve them.

## Who reviews
Caseworkers review by hand the seekers at the edges of the band, seekers whose registration dates are uncertain, and seekers with a disability or living far from an office whom the rule leaves out (see the list attached).

## What would confirm it
A trial in which each new seeker is randomised to the rule or to current practice: about 18,000 seekers in each, roughly two years of registrations, with one look at the halfway point.
"""
    # END SOLUTION


try:
    page = summary(F)
except NotImplementedError:
    page = U.reference("d4_4", "page")
U.keep("d4_4", "page", page)
out = U.artifacts("d4_4") / "rule_summary.md"
out.write_text(page, encoding="utf-8")
print(page)
words = len(re.findall(r"\w+", page))
heads = re.findall(r"^## (.+)$", page, flags=re.M)
print(f"{words} words · saved {out}")
U.check(heads == ["What the rule does", "What it is worth", "Who it leaves out", "Who reviews", "What would confirm it"], "the five headings, in order")
U.check(words <= 450, "it fits on one page")
U.check(not re.search(r"\b(CATE|policy value|propensit|doubly robust|IPW)\b", page, flags=re.I), "no jargon a director would need explained")

# %% [markdown]
# ## 3 · Edge cases for human review (5')
#
# A rule with hard thresholds is wrong at its edges and blind to what it does not use. List at least four kinds of seeker a caseworker should review, each with how many of the logged seekers it covers and why.
#
# ✏️ **Exercise** · Return a table with columns `case`, `seekers` (count in the logs) and `why`.

# %%
def edge_cases():
    # BEGIN SOLUTION
    m = L["months_unemployed"]
    rows = [
        ("unemployed exactly 10 or 16 months", int(m.isin([10, 16]).sum()), "one month either side of the band: the effect barely differs"),
        ("disability, unemployed more than 15 months", int(((L["disability"] == 1) & (m > 15)).sum()),
         "disabled seekers gain more on average; the rule does not see disability"),
        ("in the band but more than 60 km from an office", int((m.between(11, 15) & (L["distance_km"] > 60)).sum()),
         "offered a place they may not be able to attend; consider remote coaching"),
        ("unemployed more than 36 months", int((m > 36).sum()), "the logs hold almost no evidence for them (Lab 3.3): a pilot, not the rule, should decide"),
        ("months unemployed recorded as 0", int((m == 0).sum()), "often a re-registration with the earlier spell lost: check the history"),
    ]
    return pd.DataFrame(rows, columns=["case", "seekers", "why"])
    # END SOLUTION


try:
    edges = edge_cases()
except NotImplementedError:
    edges = U.reference("d4_4", "edges")
U.keep("d4_4", "edges", edges)
print(edges.to_string(index=False))
U.check(len(edges) >= 4 and (edges["seekers"] > 0).all(), "at least four kinds of case, each with its count")

# %% [markdown]
# ## 4 · Choose the capstone brief (5')
#
# Three briefs are in `capstone/`, each with a `BRIEF.md`, a randomised pilot, logs with recorded probabilities and a truth folder for checking afterwards. The capstone asks for the full chain of the week on one of them.
#
# ✏️ **Exercise** · Read the three briefs below, then set `CHOICE` to one folder name and `PLAN` to your first three steps.

# %%
BRIEFS = sorted(d.name for d in (U.LABS / "capstone").iterdir() if d.is_dir())
for b in BRIEFS:
    first = (U.LABS / "capstone" / b / "BRIEF.md").read_text(encoding="utf-8").splitlines()[0]
    pil = pd.read_csv(U.LABS / "capstone" / b / "pilot.csv", nrows=0).columns
    print(f"{b:22s} {first[2:]} · {len(pil)} pilot columns")


def choose():
    # BEGIN SOLUTION
    return "siraj-energy-visits", ["state the estimand and check overlap in the logs, district by district",
                                   "fit and validate effect estimates on the pilot's held-out split",
                                   "learn a rule under the 15% budget and value it on the logs with diagnostics"]
    # END SOLUTION


try:
    CHOICE, PLAN = choose()
except NotImplementedError:
    CHOICE, PLAN = U.reference("d4_4", "choice")
U.keep("d4_4", "choice", [CHOICE, PLAN])
print("chosen:", CHOICE, *[f"\n  {i + 1}. {s}" for i, s in enumerate(PLAN)])
U.check(CHOICE in BRIEFS, "the choice is one of the three briefs")
U.check(len(PLAN) >= 3, "three first steps")

# %% [markdown]
# ## ✱ Stretch · Cut the summary to what the director would repeat
#
# Wijdan will say two or three sentences to the board. Write them, with one number each at most.

# %%
def stretch():
    # BEGIN SOLUTION
    print(f"We will give coaching places to seekers unemployed between ten and fifteen months, where the pilot shows coaching helps most. "
          f"On our records that would put about {F['jobs_per_1000']:.0f} more people in work for every thousand who register. "
          f"We will confirm it with a two-year trial before it becomes the rule everywhere.")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Why does the page give the gain in jobs per 1,000 seekers rather than in points?
#    <details><summary>Answer</summary>A director repeats counts of people more reliably than differences of percentages, and the count makes the scale clear.</details>
# 2. Why are edge cases part of a rule, not an afterthought?
#    <details><summary>Answer</summary>A rule with thresholds treats seekers one month apart differently and cannot see what it does not use; human review where the rule is weakest is part of its design and of the agency's accountability.</details>
# 3. What must the page not claim?
#    <details><summary>Answer</summary>That the rule is proven: its value is an offline estimate from records, to be confirmed by the trial.</details>
#
# ## Carry forward
#
# The capstone: the same chain on your chosen brief, in your own public repository. The capstone deck lists what the README must report.
