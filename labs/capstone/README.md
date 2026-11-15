# Capstone: a policy evaluation, defended

An individual policy evaluation of one of the three synthetic briefs in this folder: effects, a constrained rule, its offline value, a subgroup audit and a trial plan. Each brief has its own folder with a `BRIEF.md`, a randomised pilot (`pilot.csv`), logged decisions with the probabilities the deciding tool recorded (`logged.csv.gz`) and a `truth/` folder. Lab 4.4 is where you choose.

| Folder | Organisation (fictional) | Decision | Budget | Outcome |
|---|---|---|---|---|
| `siraj-energy-visits/` | Siraj Power Distribution | energy-saving home visits | 15% of households | % of summer electricity saved |
| `nakheel-inspections/` | Al-Nakheel Municipality Food Safety Department | unannounced restaurant inspections | 25% of restaurants | compliant at follow-up |
| `fursa-coaching-2027/` | Fursa Employment Agency | coaching places, new cohort | 20% of seekers | employed within six months |

## What you submit

A link to a **public GitHub repository** with:

1. `README.md` that reports measured results, in this order:
   - **the estimand** (population, contrast, outcome, time, summary) and **the average effect with its overlap**: the propensity distribution in the logs, the groups with almost no treated units, and what you did about them;
   - **effect estimates** by at least two methods, **validated on the held-out randomised split** of the pilot (calibration by group and the best-linear-predictor slope);
   - **a rule under the brief's budget and fairness constraint**, with the attributes it may use and why (a drive, audit or exclude register);
   - **its offline value** on the logs with intervals and diagnostics (the recorded probabilities checked, effective sample sizes), against treat-all, risk or need targeting and current practice;
   - **an audit by the brief's groups**, and the constraint's price with a paired interval;
   - **a confirmatory trial plan**: arms, estimand, the difference to detect after dilution, units per arm, the randomisation unit, a stopping rule and the limits.
2. Notebooks that rerun everything from the raw data with a fixed seed.
3. The commit history, a run log (the printed output of the final run) and a **verification note**: every step an assistant helped with, and how you checked its result.

Rubric: identification 15, effects and validation 25, offline evaluation 25, rule and fairness audit 20, trial plan and reproducibility 15.

## The defence

Fifteen minutes. You present the evaluation; at about minute ten the assessor changes the budget or the fairness constraint (halves or doubles the budget, or removes the constraint). You first **predict** the change in the rule's value and in the audit, then **rerun** and show it. Keep the budget and the constraint as parameters at the top of your notebook.

AI policy: AI-full in the capstone (an assistant may write code with you), declared in the verification note and defended live. You must be able to explain and change every line.

## The truth folders

`truth/` holds each unit's expected outcome without and with treatment, written by the generator (`../prepare_data.py`). Real data never give you this. Use it only after your estimates are made: the README reports estimates made without it, and may then add one line comparing them with the truth. The capstone is assessed on the chain, the diagnostics and the defence, not on recovering the true values.
