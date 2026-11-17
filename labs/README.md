# Causal Machine Learning and Policy Evaluation: labs

## Quick start

- **Google Colab:** open a notebook from the table below or from the course page; its first cell clones this repository and installs `requirements-colab.txt`.
- **Locally:** clone the repository, then from `labs/` run `uv sync` and `uv run jupyter lab` (Python 3.12, laptop processor; nothing calls or downloads a model).
- **Pre-work (about three hours, before Day 1).** Potential outcomes and propensity scores on the course's data, ending with the entry quiz: [`P0_prework_refresher.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/P0_prework_refresher.ipynb).
- **Follow-up (tracked).** Within three weeks of the course, review one eligibility rule from your workplace or a public service: [`followup/README.md`](followup/README.md).

## The notebooks

| Unit | Session | Notebook |
|---|---|---|
| Pre-work | Potential outcomes and propensity scores | [`P0_prework_refresher.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/P0_prework_refresher.ipynb) |
| Day 1 | Session 1 · Estimands for allocation | [`D1_1_estimands.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_1_estimands.ipynb) |
| Day 1 | Session 2 · Overlap first | [`D1_2_overlap.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_2_overlap.ipynb) |
| Day 1 | Session 3 · Double machine learning | [`D1_3_dml.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_3_dml.ipynb) |
| Day 1 | Session 4 · Nuisance models and hidden confounding | [`D1_4_sensitivity.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_4_sensitivity.ipynb) |
| Day 2 | Session 1 · Meta-learners | [`D2_1_metalearners.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_1_metalearners.ipynb) |
| Day 2 | Session 2 · Causal forests | [`D2_2_causal_forest.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_2_causal_forest.ipynb) |
| Day 2 | Session 3 · Validating effects | [`D2_3_validate.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_3_validate.ipynb) |
| Day 2 | Session 4 · Uplift for targeting | [`D2_4_qini.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_4_qini.ipynb) |
| Day 3 | Session 1 · Logged decisions | [`D3_1_logged.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_1_logged.ipynb) |
| Day 3 | Session 2 · Weighted and augmented estimators | [`D3_2_offline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_2_offline.ipynb) |
| Day 3 | Session 3 · Untrustworthy estimates | [`D3_3_weights.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_3_weights.ipynb) |
| Day 3 | Session 4 · Reviewing an effects claim | [`D3_4_review.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_4_review.ipynb) |
| Day 4 | Session 1 · Policy learning under a budget | [`D4_1_policy_tree.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_1_policy_tree.ipynb) |
| Day 4 | Session 2 · Fair allocation | [`D4_2_fairness.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_2_fairness.ipynb) |
| Day 4 | Session 3 · The confirmatory trial | [`D4_3_trial.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_3_trial.ipynb) |
| Day 4 | Session 4 · Explaining a rule | [`D4_4_summary.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_4_summary.ipynb) |

Exercises are marked **✏️ Exercise** and raise `NotImplementedError` until you fill them in. Complete reference versions are in `solutions/`. Cells marked **✱ Stretch** are optional. Self-checks print ✅ or ❌.

**Every lab starts from a supplied reference state.** No lab needs the output of another. The effect estimates, outcome predictions, doubly robust scores and the reference rule that later labs build on are supplied in `data/reference/`; when an exercise is not done yet, the next cells use its stored result and say so (↪), so the notebook still runs to the end. Outputs go to `artifacts/<lab>/`, which is not committed.

**AI policy.** Each lab states its own. In brief: no assistant in the pre-work, the supervised check and each day's first estimator (Labs 1.3, 2.1, 3.1, 4.1); a hint-only tutor elsewhere ([`tutor_instruction.md`](tutor_instruction.md) is its instruction, to paste into an approved assistant); any assistant in the capstone, declared in a verification note and defended live.

## The case study: Fursa Employment Agency

Fursa Employment Agency is fictional, and so are its six regions (Sahil, Rumh, Jaww, Qaraa, Mazari, Nijad), its twelve offices and every job seeker. All data are synthetic, written by `prepare_data.py` with a fixed seed.

| Data | File | Contents | Used in |
|---|---|---|---|
| The pilot | `data/pilot.csv` | 6,000 job seekers coached by lottery (one in two within each office), 14 covariates, coaching sessions attended, employed within six months, a training (60%) and held-out (40%) split | Days 1, 2 and 4 |
| The logs | `data/logged.csv.gz` | 40,000 caseworker decisions of 2025 and 2026: 60 confounders (13 named, the risk score, 46 administrative columns), the offer probability the allocation tool recorded, the offer, employed within six months | Days 1, 3 and 4 |
| Requests | `data/requests.json` | the director's three questions | Lab 1.1 |
| The analyst's draft | `data/review/anwar_analysis.md` | a subgroup analysis with four seeded faults | Lab 3.4 |
| Truths | `data/truth.json`, `data/truth_pilot.csv`, `data/truth_logged.csv.gz` | the true averages, the planted problems and each seeker's chance of employment without and with coaching | self-checks |
| Reference states | `data/reference/` | supplied estimates and the stored result of every exercise | safety nets |
| Capstone briefs | `capstone/` | three briefs, each with a pilot, logs and a `BRIEF.md` | capstone |

**The planted problems** the labs find: the risk score ranks need, not benefit; groups almost never offered coaching (disabled seekers in Rumh and Nijad, seekers unemployed more than 36 months, seekers over 55); nonlinear confounding through administrative history, so linear adjustment is biased; a later extract without the motivation note (a hidden confounder); office RUM-2 overrode the tool while it kept recording its own probabilities; a rule that reaches the never-offered groups rests on a few heavy weights; four faults in the analyst's draft; seekers in Nijad, unemployed longer and farther from an office, gain less, so a ranking by benefit leaves the region behind.

**The people.** Wijdan Al-Hazzaa, director of employment services, owns the decision. Duhaim Al-Thumairi, head of caseworkers, owns current practice. Shuruq Al-Obaikan is the data protection officer. Anwar Al-Mohaimeed is the analyst whose draft is reviewed. Majd Al-Mazroa, finance lead, owns the coaching budget. Ohood Al-Saqabi, senior data scientist, is the participant's role.

## The helper module

`fursa.py` holds the loaders (`pilot()`, `logged()`, `truth()`, `truth_pilot()`, `truth_logged()`, `requests()`, `supplied()`), the learners the labs share (`gbm_classifier`, `pilot_classifier`, `folds`, `crossfit_aipw`, `outcome_models_crossfit`), the validation tools (`dr_scores`, `calibration_table`, `blp_slope`, `qini_curve`), the offline estimators (`ipw_value`, `snipw_value`, `dr_value`, `ess`), the policy tree (`policy_tree`, `budget_tree`, `predict_tree`, `tree_text`) and the trial helper `n_per_arm`. A lab asks you to write each one before it uses the course's version.

## Modes

- **Full** (the default) runs every lab on all the data, as the slides quote it. The slowest labs (the causal forests of Day 2 and the policy trees of Day 4) take one to three minutes on a laptop.
- **Quick** (`DS402_QUICK=1`) samples the logs and coarsens searches so every solution runs in a few minutes; its numbers differ a little from the slides.

`prepare_data.py` rebuilds every file in `data/` and `capstone/` (they are committed): `uv run python prepare_data.py`. `_tools/make_reference.py` rebuilds the supplied estimates in `data/reference/`.
