# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "statsmodels>=0.14.4"]
# ///
# %% [markdown]
# # Lab 1.1 · Three estimands and a risk trap
#
# Day 1 · Session 1 (Estimands for allocation) · 25 min.
#
# Wijdan Al-Hazzaa, director of employment services at Fursa, has sent three questions. Before any method, you write each one as an estimand: whose outcome, under what, against what, summarised how. Then you use the randomised coaching pilot to test the belief behind her third question: that the seekers with the highest risk score gain most from coaching.
#
# **AI policy.** AI-tutor: an assistant may explain a term or an error; it does not write your estimands or code.
#
# **Reference state.** `data/requests.json` and `data/pilot.csv`, loaded fresh. Nothing from another lab is needed.
#
# **Tasks**
# - [ ] 1 · Write the three estimands (8')
# - [ ] 2 · Effects by risk decile in the pilot (7')
# - [ ] 3 · Top fifth by risk against top fifth by benefit (6')
# - [ ] 4 · Answer the director's third question (4')
# - [ ] ✱ Stretch · Write the estimand for the share of seekers whom coaching harms

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
import fursa as U
print("mode:", U.MODE)

for q in U.requests()["questions"]:
    print(q["id"], "·", q["text"])
P = U.pilot()
print(f"pilot: {len(P):,} seekers, {P.coached.mean():.0%} coached by lottery, employment within six months {P.employed_6m.mean():.1%}")

# %% [markdown]
# ## 1 · Write the three estimands (8')
#
# An estimand has five parts: the **population** (whose outcomes), the **treatment** (what is done, or the rule that decides it), the **comparison** (against what), the **outcome** (measured when) and the **summary** (an average, a difference, a value). Q1 is about seekers who were offered coaching. Q2 is about a rule under a budget. Q3 compares two ways to rank seekers.
#
# ✏️ **Exercise** · Return a dict `{"Q1": {...}, "Q2": {...}, "Q3": {...}}`, each with the five keys.

# %%
PARTS = ["population", "treatment", "comparison", "outcome", "summary"]


def estimands():
    # BEGIN SOLUTION
    return {
        "Q1": {"population": "job seekers offered coaching in 2025 and 2026",
               "treatment": "coaching as offered",
               "comparison": "no coaching for the same seekers",
               "outcome": "employed within six months of registration",
               "summary": "average effect on the offered (ATT), a difference in employment rates"},
        "Q2": {"population": "next year's job seekers",
               "treatment": "an allocation rule that offers one place per five seekers",
               "comparison": "current caseworker practice, and other rules within the same budget",
               "outcome": "employed within six months",
               "summary": "policy value: the employment rate if the rule decided, and its gain over current practice"},
        "Q3": {"population": "next year's job seekers",
               "treatment": "places to the top fifth by conditional effect of coaching",
               "comparison": "places to the top fifth by risk score",
               "outcome": "employed within six months",
               "summary": "difference in policy value between the two rankings, same budget"},
    }
    # END SOLUTION


try:
    est = estimands()
except NotImplementedError:
    est = U.reference("d1_1", "estimands")
U.keep("d1_1", "estimands", est)
for k, v in est.items():
    print(k, "·", v["summary"])
U.check(all(set(PARTS) <= set(v) for v in est.values()), "five parts for each question")
U.check("offer" in est["Q1"]["population"].lower(), "Q1 is about the seekers who were offered coaching (the effect on the offered)")
U.check("rule" in est["Q2"]["treatment"].lower() and "value" in est["Q2"]["summary"].lower(), "Q2 is the value of a rule under the budget")
U.check("risk" in est["Q3"]["comparison"].lower() + est["Q3"]["treatment"].lower(), "Q3 sets ranking by risk against another ranking")

# %% [markdown]
# ## 2 · Effects by risk decile in the pilot (7')
#
# The pilot was randomised, so within any group fixed before the lottery the coached-minus-uncoached difference in employment estimates that group's effect. Cut the seekers into ten deciles of the agency's `risk_score` (decile 10 = highest risk).
#
# ✏️ **Exercise** · Return a table with one row per decile: `decile`, `n`, `employed_uncoached` (the employment rate without coaching), `effect` and its `se`.

# %%
def effect_by_decile(P):
    # BEGIN SOLUTION
    d = P.assign(decile=pd.qcut(P["risk_score"].rank(method="first"), 10, labels=False) + 1)
    rows = []
    for k, g in d.groupby("decile"):
        a, b = g.loc[g.coached == 1, "employed_6m"], g.loc[g.coached == 0, "employed_6m"]
        rows.append({"decile": int(k), "n": len(g), "employed_uncoached": b.mean(), "effect": a.mean() - b.mean(),
                     "se": np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))})
    return pd.DataFrame(rows)
    # END SOLUTION


try:
    dec = effect_by_decile(P)
except NotImplementedError:
    dec = U.reference("d1_1", "deciles")
U.keep("d1_1", "deciles", dec)
print(dec.round(3).to_string(index=False))
U.check(len(dec) == 10, "ten deciles")
U.check(dec["employed_uncoached"].iloc[-1] < dec["employed_uncoached"].iloc[0], "the score does find low employment: decile 10 is lowest without coaching")
U.check(dec["effect"].idxmax() != 9, "the largest effect is not in the highest-risk decile")

# %% [markdown]
# **Look at** the two columns. Risk rises as employment without coaching falls, as the score promises. The effect of coaching does not rise with it: each decile's interval is wide, so read the shape, not a single row.

# %% [markdown]
# ## 3 · Top fifth by risk against top fifth by benefit (6')
#
# The generator knows each pilot seeker's true chances with and without coaching (`U.truth_pilot()`); real data never give you this. Use it once, to see the size of the trap.
#
# ✏️ **Exercise** · Return a dict: `gain_risk` (mean true effect in the top fifth by risk score), `gain_benefit` (mean true effect in the top fifth by true effect) and `overlap` (the share of the risk fifth that is also in the benefit fifth).

# %%
tp = U.truth_pilot()
true_effect = (tp["p1"] - tp["p0"]).to_numpy()


def top_fifths(risk, effect):
    # BEGIN SOLUTION
    r, b = U.top_share(risk, 0.2).astype(bool), U.top_share(effect, 0.2).astype(bool)
    return {"gain_risk": float(effect[r].mean()), "gain_benefit": float(effect[b].mean()),
            "overlap": float((r & b).sum() / r.sum())}
    # END SOLUTION


try:
    tf = top_fifths(P["risk_score"].to_numpy(), true_effect)
except NotImplementedError:
    tf = U.reference("d1_1", "fifths")
U.keep("d1_1", "fifths", tf)
print(f"top fifth by risk: true gain {U.pp(tf['gain_risk'])} · by benefit: {U.pp(tf['gain_benefit'])} · "
      f"overlap {tf['overlap']:.0%} · everyone: {U.pp(true_effect.mean())}")
U.check(tf["gain_risk"] < 0.6 * tf["gain_benefit"] and tf["gain_risk"] < true_effect.mean() + 0.01, "the risk fifth gains no more than the average seeker, far less than the benefit fifth")

# %% [markdown]
# ## 4 · Answer the director's third question (4')
#
# ✏️ **Exercise** · Two or three sentences Wijdan can repeat: what the score measures, what coaching needs, and what Day 2 will estimate. Use your numbers.

# %%
def answer_q3():
    # BEGIN SOLUTION
    return (f"The risk score finds the seekers least likely to find work without help, but in the pilot those seekers gain "
            f"about {100 * tf['gain_risk']:.0f} points from coaching, against {100 * tf['gain_benefit']:.0f} points for the fifth who gain most; "
            f"only {tf['overlap']:.0%} of the two groups are the same people. Risk is the outcome without coaching, benefit is the "
            "difference coaching makes, so places should follow estimated benefit, which we estimate and check on Day 2.")
    # END SOLUTION


try:
    text = answer_q3()
except NotImplementedError:
    text = U.reference("d1_1", "answer")
U.keep("d1_1", "answer", text)
print(text)
U.check(len(text) > 150 and any(ch.isdigit() for ch in text), "an answer with numbers")

# %% [markdown]
# ## ✱ Stretch · The share whom coaching harms
#
# Write the estimand, then compute it from the truth. Why can no randomised trial estimate it directly?

# %%
def stretch():
    # BEGIN SOLUTION
    print("estimand: among next year's seekers, the share whose chance of employment within six months is lower with coaching than without")
    print(f"truth: {np.mean(true_effect < 0):.1%} · a trial sees each seeker in one arm only, so individual effects, and their sign, are never observed")
    # END SOLUTION


try:
    stretch()
except NotImplementedError:
    print("✱ stretch not done")

# %% [markdown]
# ## Checkpoint
#
# 1. Q1 and Q2 both mention coaching. Why are they different estimands?
#    <details><summary>Answer</summary>Q1 averages over the seekers who were offered coaching (the effect on the offered). Q2 is the value of a rule that has not been used yet, over next year's seekers, under a budget.</details>
# 2. Why was decile 10 the lowest without coaching, and still not where coaching helps most?
#    <details><summary>Answer</summary>The score predicts the outcome without coaching. The effect is the difference between two outcomes; it can be small for seekers who will not find work either way.</details>
# 3. Why could you read effects by decile from the pilot but not from the logs?
#    <details><summary>Answer</summary>In the pilot a lottery decided coaching, so the coached and uncoached in a decile are comparable. In the logs caseworkers decided, using information that also predicts employment.</details>
#
# ## Carry forward
#
# The estimands of Q1 and Q2 are the targets of Labs 1.3 and 4.1. Every lab starts from its own supplied files.
