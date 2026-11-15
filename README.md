<!-- academy:title -->
# Causal Machine Learning and Policy Evaluation

تعلم الآلة السببي وتقييم السياسات

Data Science Track · SDAIA Academy · DS-402 · Level 400

Estimate who benefits from a programme with machine learning, evaluate allocation rules offline before using them, and choose a rule under a budget and a fairness constraint that ends in the trial that would confirm it

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
| Capstone | Capstone: a policy evaluation and its defence |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**Fursa Employment Agency.** Fursa Employment Agency (وكالة فرصة للتوظيف) is a fictional public employment agency with six fictional regions and twelve offices, and all of its data are synthetic, generated with a fixed seed and carrying known true effects: a randomised coaching pilot of 6,000 job seekers and 40,000 logged caseworker decisions with 60 confounders and the offer probability the allocation tool recorded. Next year it has one coaching place for every five job seekers. Planted problems (a risk score that ranks the hopeless and the fine first, groups almost never offered coaching, nonlinear confounding through administrative history, a later extract without the motivation note, an office whose recorded probabilities are wrong, a rule resting on a few heavy weights, an analyst's subgroup claim with four seeded faults, remote seekers left behind by an unconstrained rule) are each found by a named lab. Participants work as Ohood Al-Saqabi, a senior data scientist answering Wijdan Al-Hazzaa, the director of employment services, with Duhaim Al-Thumairi, head of caseworkers, and Shuruq Al-Obaikan, the data protection officer.

The data are written by `labs/prepare_data.py` with seed 402; because the generator wrote every true effect, each lab sets its estimate against the truth, which real data never allow. Nothing in the course calls or downloads a model.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Three estimands and a risk trap | [`D1_1_estimands.ipynb`](labs/D1_1_estimands.ipynb) |
| 1.2 | Day 1 · Session 2 | Groups almost never offered | [`D1_2_overlap.ipynb`](labs/D1_2_overlap.ipynb) |
| 1.3 | Day 1 · Session 3 | Double machine learning against regression | [`D1_3_dml.ipynb`](labs/D1_3_dml.ipynb) |
| 1.4 | Day 1 · Session 4 | Swap learners, bound a hidden confounder | [`D1_4_sensitivity.ipynb`](labs/D1_4_sensitivity.ipynb) |
| 2.1 | Day 2 · Session 1 | S-, T- and X-learners on the pilot | [`D2_1_metalearners.ipynb`](labs/D2_1_metalearners.ipynb) |
| 2.2 | Day 2 · Session 2 | A causal forest and its drivers | [`D2_2_causal_forest.ipynb`](labs/D2_2_causal_forest.ipynb) |
| 2.3 | Day 2 · Session 3 | Calibration and ranking per estimator | [`D2_3_validate.ipynb`](labs/D2_3_validate.ipynb) |
| 2.4 | Day 2 · Session 4 | Uplift, Qini and a threshold | [`D2_4_qini.ipynb`](labs/D2_4_qini.ipynb) |
| 3.1 | Day 3 · Session 1 | Current practice from the logs | [`D3_1_logged.ipynb`](labs/D3_1_logged.ipynb) |
| 3.2 | Day 3 · Session 2 | Three rules offline | [`D3_2_offline.ipynb`](labs/D3_2_offline.ipynb) |
| 3.3 | Day 3 · Session 3 | A rule resting on heavy weights | [`D3_3_weights.ipynb`](labs/D3_3_weights.ipynb) |
| 3.4 | Day 3 · Session 4 | Four seeded faults | [`D3_4_review.ipynb`](labs/D3_4_review.ipynb) |
| 4.1 | Day 4 · Session 1 | A depth-two rule against the alternatives | [`D4_1_policy_tree.ipynb`](labs/D4_1_policy_tree.ipynb) |
| 4.2 | Day 4 · Session 2 | Attributes, audit and a constraint | [`D4_2_fairness.ipynb`](labs/D4_2_fairness.ipynb) |
| 4.3 | Day 4 · Session 3 | Size the trial | [`D4_3_trial.ipynb`](labs/D4_3_trial.ipynb) |
| 4.4 | Day 4 · Session 4 | One-page summary and the capstone brief | [`D4_4_summary.ipynb`](labs/D4_4_summary.ipynb) |
<!-- /academy:labs -->

## Capstone

A policy evaluation for one of three synthetic briefs (coaching places, energy-saving visits, food-safety inspections): effects, a constrained rule, its offline value, a subgroup audit and a trial plan, in a public repository whose README reports measured results; 15-minute defence

**Briefs** (in `labs/capstone/`): Fursa's 2027 coaching cohort; Siraj Power Distribution's energy-saving home visits; Al-Nakheel Municipality's food-safety inspections. Each has a randomised pilot, logged decisions with recorded probabilities, a budget and the groups to audit.

**Submission:** A link to a public GitHub repository whose README reports measured results for one of the three briefs in labs/capstone/: the estimand, the average effect with its overlap, CATE estimates validated on the held-out randomised split, a rule under the brief's budget and a fairness constraint with the attributes it may use, its offline value with intervals and diagnostics against treat-all, risk or need targeting and current practice, an audit by the brief's groups and a confirmatory trial plan, with notebooks that rerun from the raw data with a fixed seed, the commit history, the run log and a verification note on assisted steps. Then a 15-minute live defence in which the assessor changes the budget or the fairness constraint and the participant predicts the change in policy value before re-running.

**Rubric:** identification 15, effects and validation 25, offline evaluation 25, rule and fairness audit 20, trial plan and reproducibility 15. **Assessment:** an AI-off supervised check (an estimand, an overlap plot, an inverse-propensity value by hand) 20%, the capstone 60%, the live defence of one design choice 20%; passing is 70% overall with at least 60% on the capstone and on the supervised check.

**AI policy:** AI-off for the supervised check and each day's first estimator (Labs 1.3, 2.1, 3.1, 4.1); AI-tutor in the other labs (hints, no finished code); AI-full in the capstone, with assisted steps in a verification note, defended live.

## Responsible use

Allocation rules decide who receives public support. Every rule in the course is audited by region, gender and disability status, and is recommended only with a confirmatory trial and human review of edge cases. Lab 4.2 applies the purpose limitation of the Personal Data Protection Law and the national bias guide to decide which attributes may drive a rule and which may only audit it.

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

### Double machine learning and sensitivity

- [Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal](https://doi.org/10.1111/ectj.12097)
- [Bach, Chernozhukov, Kurz and Spindler (2022). DoubleML: an object-oriented implementation of double machine learning in Python. JMLR 23](https://jmlr.org/papers/v23/21-0862.html)
- [DoubleML documentation](https://docs.doubleml.org/stable/index.html)
- [DoubleML user guide: sensitivity analysis](https://docs.doubleml.org/stable/guide/sensitivity.html)
- [Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis. Long story short: omitted variable bias in causal machine learning](https://arxiv.org/abs/2112.13398)

### Heterogeneous effects and their validation

- [Künzel, Sekhon, Bickel and Yu (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. PNAS](https://doi.org/10.1073/pnas.1804597116)
- [Wager and Athey (2018). Estimation and inference of heterogeneous treatment effects using random forests. JASA](https://doi.org/10.1080/01621459.2017.1319839)
- [Athey, Tibshirani and Wager (2019). Generalized random forests. The Annals of Statistics](https://doi.org/10.1214/18-AOS1709)
- [EconML documentation: forest-based estimators](https://www.pywhy.org/EconML/spec/estimation/forest.html)
- [grf: generalized random forests for R](https://grf-labs.github.io/grf/)
- [Chernozhukov, Demirer, Duflo and Fernández-Val. Generic machine learning inference on heterogeneous treatment effects in randomized experiments](https://arxiv.org/abs/1712.04802)
- [Yadlowsky, Fleming, Shah, Brunskill and Wager. Evaluating treatment prioritization rules via rank-weighted average treatment effects](https://arxiv.org/abs/2111.07966)
- [CausalML documentation (uplift modelling)](https://causalml.readthedocs.io/en/latest/)
- [Knaus, Lechner and Strittmatter (2022). Heterogeneous employment effects of job search programs: a machine learning approach. Journal of Human Resources](https://doi.org/10.3368/jhr.57.2.0718-9615R1)

### Off-policy evaluation, policy learning and trials

- [Dudík, Langford and Li. Doubly robust policy evaluation and learning](https://arxiv.org/abs/1103.4601)
- [Athey and Wager (2021). Policy learning with observational data. Econometrica](https://doi.org/10.3982/ECTA15732)
- [Kitagawa and Tetenov (2018). Who should be treated? Empirical welfare maximization methods for treatment choice. Econometrica](https://doi.org/10.3982/ECTA13288)
- [Sverdrup, Kanodia, Zhou, Athey and Wager (2020). policytree: policy learning via doubly robust empirical welfare maximization over trees. JOSS](https://doi.org/10.21105/joss.02232)
- [policytree on CRAN](https://cran.r-project.org/package=policytree)
- [Crépon, Duflo, Gurgand, Rathelot and Zamora (2013). Do labor market policies have displacement effects? Evidence from a clustered randomized experiment. QJE](https://doi.org/10.1093/qje/qjt001)

### Tools

- [Fairlearn](https://fairlearn.org/)
- [uv](https://docs.astral.sh/uv/)

### National references

- [SDAIA AI Ethics Principles](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [AI Bias Reference Guide, first edition (SDAIA, May 2026)](https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/EN-AI-BIAS.pdf)
- [Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf)
- [Implementing Regulation of the Personal Data Protection Law](https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf)
- [National AI Risk Management Framework, English executive summary (SDAIA, April 2026)](https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/ExecutiveSummaryEn.pdf)
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: <https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf>
- SDAIA AI Bias Reference Guide (first edition, May 2026): <https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/EN-AI-BIAS.pdf>
- Personal Data Protection Law: <https://sdaia.gov.sa/en/SDAIA/about/Documents/PersonalDataProtectionLaw.pdf>; its Implementing Regulation: <https://sdaia.gov.sa/en/SDAIA/about/Documents/ImplementingRegulationPersonalDataProtectionLaw.pdf>
- National AI Risk Management Framework (April 2026), English executive summary: <https://sdaia.gov.sa/en/MediaCenter/KnowledgeCenter/ResearchLibrary/ExecutiveSummaryEn.pdf>
