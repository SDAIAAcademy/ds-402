---
course: "Causal Machine Learning and Policy Evaluation"
code: "DS-402"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-402/index.html"
---

# Causal Machine Learning and Policy Evaluation

تعلّم الآلة السببي وتقييم السياسات

Data Science Track · 4 teaching days + capstone day · 26 hours · Level 400

Estimate who benefits from a programme with machine learning, evaluate allocation rules offline before using them, and choose a rule under a budget and a fairness constraint that ends in the trial that would confirm it

Participants use machine learning to estimate who benefits from a programme and to choose the rule that allocates it. On a fictional employment agency's synthetic coaching pilot and logged caseworker decisions, they estimate average effects by double machine learning, estimate and validate heterogeneous effects, evaluate allocation rules offline and learn a rule under a budget and a fairness constraint. Every rule ends with the trial that would confirm it.

**Who it is for:** Specialists. Data scientists who have completed the level 300 core or can show equivalent work: a defended analysis with a validated model or a causal estimate that they can explain.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Fursa Employment Agency.** Fursa Employment Agency (وكالة فرصة للتوظيف) is a fictional public employment agency with six fictional regions and twelve offices, and all of its data are synthetic, generated with a fixed seed and carrying known true effects: a randomised coaching pilot of 6,000 job seekers and 40,000 logged caseworker decisions with 60 confounders and the offer probability the allocation tool recorded. Next year it has one coaching place for every five job seekers. Planted problems (a risk score that ranks the hopeless and the fine first, groups almost never offered coaching, nonlinear confounding through administrative history, a later extract without the motivation note, an office whose recorded probabilities are wrong, a rule resting on a few heavy weights, an analyst's subgroup claim with four seeded faults, remote seekers left behind by an unconstrained rule) are each found by a named lab. Participants work as Ohood Al-Saqabi, a senior data scientist answering Wijdan Al-Hazzaa, the director of employment services, with Duhaim Al-Thumairi, head of caseworkers, and Shuruq Al-Obaikan, the data protection officer.

## The course

### Day 1: Average effects with many confounders

Slides: https://sdaiaacademy.github.io/ds-402/slides/day1.md

- **Session 1: Estimands for allocation**
  - Write an allocation question as an estimand with population, contrast, outcome and summary
  - Distinguish the average effect, the effect on the offered and the value of a rule
  - Show from randomised data that the highest risk is not the highest benefit
  - [Lab 1.1: Three estimands and a risk trap (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D1_1_estimands.ipynb)
- **Session 2: Overlap first**
  - Fit a cross-fitted propensity model on 60 confounders
  - Map the groups almost never offered coaching
  - Decide between trimming and changing the estimand, and say what each costs
  - [Lab 1.2: Groups almost never offered (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D1_2_overlap.ipynb)
- **Session 3: Double machine learning**
  - Explain why plugging machine learning predictions into a regression biases the effect
  - Estimate an average effect with cross-fitting by hand and with a library
  - Report an estimate table with intervals against regression adjustment
  - [Lab 1.3: Double machine learning against regression (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D1_3_dml.ipynb)
- **Session 4: Nuisance models and hidden confounding**
  - Compare estimates across nuisance learners with their prediction losses
  - Bound an estimate for a hidden confounder with sensitivity parameters
  - Benchmark a bound against an observed covariate
  - [Lab 1.4: Swap learners, bound a hidden confounder (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D1_4_sensitivity.ipynb)

### Day 2: Who benefits

Slides: https://sdaiaacademy.github.io/ds-402/slides/day2.md

- **Session 1: Meta-learners**
  - Fit S-, T- and X-learners on a randomised pilot
  - Explain why an S-learner shrinks effects towards zero
  - Measure each learner against known effects and say why that is impossible on real data
  - [Lab 2.1: S-, T- and X-learners on the pilot (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D2_1_metalearners.ipynb)
- **Session 2: Causal forests**
  - Fit a causal forest with local centring
  - Read which covariates drive effect variation
  - Describe how the effect changes with months unemployed and digital skills
  - [Lab 2.2: A causal forest and its drivers (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D2_2_causal_forest.ipynb)
- **Session 3: Validating effects**
  - Check calibration of effect estimates by quantile group on held-out randomised data
  - Test whether predictions carry signal with the best linear predictor
  - Choose an estimator from calibration and ranking, with reasons
  - [Lab 2.3: Calibration and ranking per estimator (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D2_3_validate.ipynb)
- **Session 4: Uplift for targeting**
  - Draw uplift and Qini curves from randomised data
  - Compare targeting by estimated benefit with targeting by risk
  - Set a threshold for one place per five seekers
  - [Lab 2.4: Uplift, Qini and a threshold (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D2_4_qini.ipynb)

### Day 3: Evaluating rules before using them

Slides: https://sdaiaacademy.github.io/ds-402/slides/day3.md

- **Session 1: Logged decisions**
  - Compute the value of current practice from the logs
  - Check recorded probabilities against the decisions they describe
  - Decide what to do with an office whose records are wrong
  - [Lab 3.1: Current practice from the logs (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D3_1_logged.ipynb)
- **Session 2: Weighted and augmented estimators**
  - Write allocation rules as decisions per seeker
  - Estimate a rule's value by inverse propensity and self-normalised weighting
  - Estimate it by the augmented estimator and compare the intervals
  - [Lab 3.2: Three rules offline (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D3_2_offline.ipynb)
- **Session 3: Untrustworthy estimates**
  - Compute the effective sample size of a rule's weights
  - Find the decisions that carry a rule's value
  - Show what clipping weights does to the estimate and its bias
  - [Lab 3.3: A rule resting on heavy weights (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D3_3_weights.ipynb)
- **Session 4: Reviewing an effects claim**
  - Recognise a subgroup defined after treatment
  - Re-test a discovered subgroup on data not used to find it
  - Name the effect of each fault on the claim
  - [Lab 3.4: Four seeded faults (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D3_4_review.ipynb)

### Day 4: Choosing and testing a policy

Slides: https://sdaiaacademy.github.io/ds-402/slides/day4.md

- **Session 1: Policy learning under a budget**
  - Compute doubly robust scores from randomised data
  - Learn a depth-two policy tree under a budget
  - Compare the rule with treat-all, risk targeting and current practice
  - [Lab 4.1: A depth-two rule against the alternatives (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D4_1_policy_tree.ipynb)
- **Session 2: Fair allocation**
  - Decide which attributes may drive a rule and which may only audit it
  - Audit a rule by region, gender and disability status
  - Add a fairness constraint and measure its cost in value
  - [Lab 4.2: Attributes, audit and a constraint (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D4_2_fairness.ipynb)
- **Session 3: The confirmatory trial**
  - State the estimand of a trial of a rule against current practice
  - Size the trial for the difference diluted by agreeing decisions
  - Choose a randomisation unit and a stopping rule
  - [Lab 4.3: Size the trial (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D4_3_trial.ipynb)
- **Session 4: Explaining a rule**
  - Summarise a rule, its value and its audit on one page
  - Name the edge cases that go to human review
  - Choose a capstone brief and plan its analysis
  - [Lab 4.4: One-page summary and the capstone brief (25 min)](https://sdaiaacademy.github.io/ds-402/labs/D4_4_summary.ipynb)

### Capstone: Capstone: a policy evaluation and its defence

Slides: https://sdaiaacademy.github.io/ds-402/slides/capstone.md

## Capstone

A policy evaluation for one of three synthetic briefs (coaching places, energy-saving visits, food-safety inspections): effects, a constrained rule, its offline value, a subgroup audit and a trial plan, in a public repository whose README reports measured results; 15-minute defence

**What you submit:** A link to a public GitHub repository whose README reports measured results for one of the three briefs in labs/capstone/: the estimand, the average effect with its overlap, CATE estimates validated on the held-out randomised split, a rule under the brief's budget and a fairness constraint with the attributes it may use, its offline value with intervals and diagnostics against treat-all, risk or need targeting and current practice, an audit by the brief's groups and a confirmatory trial plan, with notebooks that rerun from the raw data with a fixed seed, the commit history, the run log and a verification note on assisted steps. Then a 15-minute live defence in which the assessor changes the budget or the fairness constraint and the participant predicts the change in policy value before re-running.

**Rubric:**

- Identification: 15%
- Effects and validation: 25%
- Offline evaluation: 25%
- Rule and fairness audit: 20%
- Trial plan and reproducibility: 15%

## Outcomes

- **LO1** Implement double machine learning with cross-fitting for an average effect and compare it with regression adjustment, given logged data with 60 confounders and a known effect (Applied) Evidence: an estimate table with intervals and each method's bias
- **LO2** Build conditional average treatment effect (CATE) estimators by three methods and validate them on held-out randomised data, given the randomised coaching pilot (Performance) Evidence: calibration and ranking checks per estimator and a justified choice
- **LO3** Evaluate allocation rules offline with inverse propensity and augmented estimators and diagnose untrustworthy estimates, given logged decisions with recorded propensities and poor overlap in places (Performance) Evidence: a value table with intervals, overlap diagnostics and effective sample sizes
- **LO4** Design an allocation rule under a budget and a fairness constraint and compare it with treat-all, risk targeting and current practice, given unit-level effect estimates and one place per five job seekers (Performance) Evidence: gains and outcomes by region, gender and disability status, with the attributes the rule may use
- **LO5** Review a heterogeneous-effect claim for overfitting, unstable subgroups and reuse of data for discovery and evaluation, given an analysis with seeded faults (Performance) Evidence: a review memo naming each fault and its effect
- **LO6** Design a confirmatory trial for a learned rule, powered for the comparison with current practice, given the capstone rule and its offline value (Performance) Evidence: a protocol with randomisation unit, primary metric, sample size and stopping rule

## Tools

- **Causal machine learning library:** DoubleML, EconML, CausalML, grf for R
- **Off-policy evaluation and policy learning:** course-provided NumPy estimators, policytree for R
- **Subgroup fairness metrics:** Fairlearn, course-provided pandas checks
- **Analysis environment:** Python through uv with Jupyter or marimo, R with Quarto

## Prerequisites

- DS-302 Online Experiments and A/B Testing
- DS-303 Causal Inference from Observational Data
- Estimates propensity scores and reads balance diagnostics
- Trains and cross-validates machine learning models in Python
- Setup: a laptop CPU with uv, or Google Colab. Nothing calls or downloads a model. Three hours of pre-work on potential outcomes and propensity scores (labs/P0_prework_refresher.ipynb) with its entry quiz before Day 1; see labs/README.md.

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

---

SDAIA Academy · Data Science Track · Causal Machine Learning and Policy Evaluation

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit 0067d77
