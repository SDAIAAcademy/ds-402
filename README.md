<!-- academy:title -->
# Causal Machine Learning and Policy Evaluation

تعلم الآلة السببي وتقييم السياسات

Data Science Track · SDAIA Academy · DS-402 · Level 400

REPLACE: one line under the title: what participants can do after the course

Participants use machine learning to estimate who benefits from a programme and to choose the rule that allocates it. On a fictional employment agency's synthetic coaching pilot and logged caseworker decisions, they estimate average effects by double machine learning, estimate and validate heterogeneous effects, evaluate allocation rules offline and learn a rule under a budget and a fairness constraint. Every rule ends with the trial that would confirm it.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ds-402/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Implement double machine learning with cross-fitting for an average effect and compare it with regression adjustment, given logged data with 60 confounders and a known effect | Applied | an estimate table with intervals and each method's bias |
| LO2 | Build conditional average treatment effect (CATE) estimators by three methods and validate them on held-out randomised data, given the randomised coaching pilot | Performance | calibration and ranking checks per estimator and a justified choice |
| LO3 | Evaluate allocation rules offline with inverse propensity and augmented estimators and diagnose untrustworthy estimates, given logged decisions with recorded propensities and poor overlap in places | Performance | a value table with intervals, overlap diagnostics and effective sample sizes |
| LO4 | Design an allocation rule under a budget and a fairness constraint and compare it with treat-all, risk targeting and current practice, given unit-level effect estimates and one place per five job seekers | Performance | gains and outcomes by region, gender and disability status, with the attributes the rule may use |
| LO5 | Review a heterogeneous-effect claim for overfitting, unstable subgroups and reuse of data for discovery and evaluation, given an analysis with seeded faults | Performance | a review memo naming each fault and its effect |
| LO6 | Design a confirmatory trial for a learned rule, powered for the comparison with current practice, given the capstone rule and its offline value | Performance | a protocol with randomisation unit, primary metric, sample size and stopping rule |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Average effects with many confounders | Estimands for allocation; Overlap first; Double machine learning; Nuisance models and hidden confounding | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Who benefits | Meta-learners; Causal forests; Validating effects; Uplift for targeting | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Evaluating rules before using them | Logged decisions; Weighted and augmented estimators; Untrustworthy estimates; Reviewing an effects claim | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Choosing and testing a policy | Policy learning under a budget; Fair allocation; The confirmatory trial; Explaining a rule | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Write estimands for three agency questions; flag risk confused with benefit | [`D1_1_write_estimands_three.ipynb`](labs/D1_1_write_estimands_three.ipynb) |
| 1.2 | Day 1 · Session 2 | Map groups almost never offered coaching, from a propensity model | [`D1_2_map_groups_almost.ipynb`](labs/D1_2_map_groups_almost.ipynb) |
| 1.3 | Day 1 · Session 3 | Estimate the average effect and compare plain regression with the truth | [`D1_3_estimate_average_effect.ipynb`](labs/D1_3_estimate_average_effect.ipynb) |
| 1.4 | Day 1 · Session 4 | Swap nuisance learners, then bound the estimate for a hidden confounder | [`D1_4_swap_nuisance_learners.ipynb`](labs/D1_4_swap_nuisance_learners.ipynb) |
| 2.1 | Day 2 · Session 1 | Fit S-, T- and X-learners on the randomised pilot | [`D2_1_fit_s_t.ipynb`](labs/D2_1_fit_s_t.ipynb) |
| 2.2 | Day 2 · Session 2 | Fit a causal forest and report the drivers of effect variation | [`D2_2_fit_causal_forest.ipynb`](labs/D2_2_fit_causal_forest.ipynb) |
| 2.3 | Day 2 · Session 3 | Check calibration and ranking on a held-out randomised split | [`D2_3_check_calibration_ranking.ipynb`](labs/D2_3_check_calibration_ranking.ipynb) |
| 2.4 | Day 2 · Session 4 | Draw uplift and Qini curves and set a budget threshold | [`D2_4_draw_uplift_qini.ipynb`](labs/D2_4_draw_uplift_qini.ipynb) |
| 3.1 | Day 3 · Session 1 | Recover the value of current practice from the logs | [`D3_1_recover_value_current.ipynb`](labs/D3_1_recover_value_current.ipynb) |
| 3.2 | Day 3 · Session 2 | Evaluate three candidate rules offline with intervals | [`D3_2_evaluate_three_candidate.ipynb`](labs/D3_2_evaluate_three_candidate.ipynb) |
| 3.3 | Day 3 · Session 3 | Diagnose a rule whose value rests on a few heavy weights | [`D3_3_diagnose_rule_whose.ipynb`](labs/D3_3_diagnose_rule_whose.ipynb) |
| 3.4 | Day 3 · Session 4 | Find the seeded faults in a heterogeneous-effects analysis | [`D3_4_find_seeded_faults.ipynb`](labs/D3_4_find_seeded_faults.ipynb) |
| 4.1 | Day 4 · Session 1 | Learn a depth-two policy tree; compare with treat-all and risk targeting | [`D4_1_learn_depth_two.ipynb`](labs/D4_1_learn_depth_two.ipynb) |
| 4.2 | Day 4 · Session 2 | Decide which attributes may only audit the rule, audit subgroups, add a constraint | [`D4_2_decide_which_attributes.ipynb`](labs/D4_2_decide_which_attributes.ipynb) |
| 4.3 | Day 4 · Session 3 | Size a trial of the learned rule against current practice | [`D4_3_size_trial_learned.ipynb`](labs/D4_3_size_trial_learned.ipynb) |
| 4.4 | Day 4 · Session 4 | Draft a one-page rule summary and choose the capstone brief | [`D4_4_draft_one_page.ipynb`](labs/D4_4_draft_one_page.ipynb) |
<!-- /academy:labs -->

## Capstone

A policy evaluation for one of three synthetic briefs (coaching places, energy-saving visits, food-safety inspections): effects, a constrained rule, its offline value, a subgroup audit and a trial plan, in a public repository whose README reports measured results; 15-minute defence

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- DS-302 Online Experiments and A/B Testing
- DS-303 Causal Inference from Observational Data
- Estimates propensity scores and reads balance diagnostics
- Trains and cross-validates machine learning models in Python

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Causal machine learning library:** DoubleML, EconML, CausalML, grf for R
- **Off-policy evaluation and policy learning:** course-provided NumPy estimators, policytree for R
- **Subgroup fairness metrics:** Fairlearn, course-provided pandas checks
- **Analysis environment:** Python through uv with Jupyter or marimo, R with Quarto
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- SDAIA AI Bias Reference Guide: REPLACE: the address of the official text
- Personal Data Protection Law and its implementing regulations: REPLACE: the address of the official text
- National AI Risk Management Framework: REPLACE: the address of the official text
