---
course: "Causal Machine Learning and Policy Evaluation"
code: "DS-402"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-402/"
---

# Causal Machine Learning and Policy Evaluation

Estimate who benefits from a programme with machine learning, evaluate allocation rules offline before using them, and choose a rule under a budget and a fairness constraint that ends in the trial that would confirm it

Participants use machine learning to estimate who benefits from a programme and to choose the rule that allocates it. On a fictional employment agency's synthetic coaching pilot and logged caseworker decisions, they estimate average effects by double machine learning, estimate and validate heterogeneous effects, evaluate allocation rules offline and learn a rule under a budget and a fairness constraint. Every rule ends with the trial that would confirm it.

- **Audience:** Specialists. Data scientists who have completed the level 300 core or can show equivalent work: a defended analysis with a validated model or a causal estimate that they can explain.
- **Prerequisites:** DS-302 Online Experiments and A/B Testing; DS-303 Causal Inference from Observational Data; Estimates propensity scores and reads balance diagnostics; Trains and cross-validates machine learning models in Python; Setup: a laptop CPU with uv, or Google Colab. Nothing calls or downloads a model. Three hours of pre-work on potential outcomes and propensity scores (labs/P0_prework_refresher.ipynb) with its entry quiz before Day 1; see labs/README.md.
- **Case study:** Fursa Employment Agency: Fursa Employment Agency (وكالة فرصة للتوظيف) is a fictional public employment agency with six fictional regions and twelve offices, and all of its data are synthetic, generated with a fixed seed and carrying known true effects: a randomised coaching pilot of 6,000 job seekers and 40,000 logged caseworker decisions with 60 confounders and the offer probability the allocation tool recorded. Next year it has one coaching place for every five job seekers. Planted problems (a risk score that ranks the hopeless and the fine first, groups almost never offered coaching, nonlinear confounding through administrative history, a later extract without the motivation note, an office whose recorded probabilities are wrong, a rule resting on a few heavy weights, an analyst's subgroup claim with four seeded faults, remote seekers left behind by an unconstrained rule) are each found by a named lab. Participants work as Ohood Al-Saqabi, a senior data scientist answering Wijdan Al-Hazzaa, the director of employment services, with Duhaim Al-Thumairi, head of caseworkers, and Shuruq Al-Obaikan, the data protection officer.

## Outcomes

- LO1 (Applied): Implement double machine learning with cross-fitting for an average effect and compare it with regression adjustment, given logged data with 60 confounders and a known effect
- LO2 (Performance): Build conditional average treatment effect (CATE) estimators by three methods and validate them on held-out randomised data, given the randomised coaching pilot
- LO3 (Performance): Evaluate allocation rules offline with inverse propensity and augmented estimators and diagnose untrustworthy estimates, given logged decisions with recorded propensities and poor overlap in places
- LO4 (Performance): Design an allocation rule under a budget and a fairness constraint and compare it with treat-all, risk targeting and current practice, given unit-level effect estimates and one place per five job seekers
- LO5 (Performance): Review a heterogeneous-effect claim for overfitting, unstable subgroups and reuse of data for discovery and evaluation, given an analysis with seeded faults
- LO6 (Performance): Design a confirmatory trial for a learned rule, powered for the comparison with current practice, given the capstone rule and its offline value

## Day 1: Average effects with many confounders

From three agency questions to an average effect estimated by double machine learning, with its overlap and its sensitivity to what the data lack

#### [day1#title] Average effects with many confounders

*SDAIA Academy · Causal Machine Learning and Policy Evaluation*

Day 1: From three agency questions to an average effect estimated by double machine learning, with its overlap and its sensitivity to what the data lack

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-402

#### [day1#four-days-one-decision] Four days that end in one decision: who gets a coaching place

*The course at a glance*

- **Day 1 · Average effects.** Estimands, overlap, double machine learning, hidden confounding.
- **Day 2 · Who benefits.** Meta-learners, causal forests, validation on randomised data, uplift.
- **Day 3 · Rules offline.** Logged decisions, weighted and augmented values, heavy weights, a review.
- **Day 4 · A policy.** A rule under a budget, a fairness audit, the trial that would confirm it.

> **Day 5 · Capstone.** A policy evaluation for one of three synthetic briefs, in a public repository, defended for 15 minutes while the assessor changes the budget.

#### [day1#fursa-employment-agency] Fursa Employment Agency has one coaching place for every five job seekers

*One case study all week*

A fictional public agency with six regions and twelve offices. You are Ohood Al-Saqabi, senior data scientist.

- **The pilot.** 6,000 seekers; coaching decided by lottery within each office; employment within six months.
- **The logs.** 40,000 caseworker decisions, 60 confounders and the offer probability the allocation tool recorded.
- **The truth.** The generator wrote each seeker's chance with and without coaching. Real data never give you this.

#### [day1#how-the-labs-work] Every lab starts from supplied files and checks itself against the truth

*How this course works*

- Notebooks in `labs/`, run with **uv** on a laptop or in **Colab**. No model is called or downloaded.
- Each lab loads its own **reference state**: no lab needs another.
- Checks print ✅ or ❌; a safety net lets later cells run if an exercise is unfinished.
- Set `DS402_QUICK=1` on a slow laptop: 10,000 decisions instead of 40,000.

> **AI policy.** AI off for the supervised check and for each day's first estimator (Labs 1.3, 2.1, 3.1, 4.1). In the other labs an assistant may give hints, not finished code. In the capstone, anything goes, with assisted steps in a verification note.

#### [day1#agenda] Average effects with many confounders

*Today*

- **Session 1 · Estimands for allocation**: An allocation question is an estimand first: whose outcome, under which rule, against which alternative, and why risk is not benefit
- **Session 2 · Overlap first**: Before any effect, find the seekers the logs can say nothing about: positivity is checked with a cross-fitted propensity model
- **Session 3 · Double machine learning**: Orthogonal scores and cross-fitting let flexible models adjust for many confounders without biasing the effect
- **Session 4 · Nuisance models and hidden confounding**: The estimate depends on the nuisance learners, and no learner repairs a confounder the data lack: bound it instead

### Session 1: Estimands for allocation

#### [day1#s1] Estimands for allocation

*Session 1*

- Write an allocation question as an estimand with population, contrast, outcome and summary
- Distinguish the average effect, the effect on the offered and the value of a rule
- Show from randomised data that the highest risk is not the highest benefit

#### [day1#three-questions] Three questions that sound alike and ask for three different numbers

*The director's inbox*

- **Q1.** Did coaching help the job seekers we offered it to in 2025 and 2026?
- **Q2.** Next year we have one coaching place for every five job seekers. Who should get them?
- **Q3.** Our risk score finds the people least likely to find work. Should places go to the highest scores?

#### [day1#five-parts-of-an-estimand] An estimand has five parts, and the method comes after all five

*Before any method*

| Part | Q1, written out |
| --- | --- |
| **Population** | Seekers offered coaching in 2025 and 2026 |
| **Treatment** | Coaching as offered |
| **Comparison** | No coaching, for the same seekers |
| **Outcome** | Employed within six months of registration |
| **Summary** | Average difference: the effect on the offered (ATT) |

#### [day1#ate-att-cate-value] Average effects describe people; a policy value describes a rule

*Four summaries you will use all week*

- **ATE.** Mean of `Y(1) - Y(0)` over everyone.
- **ATT.** The same mean over those who were treated.
- **CATE.** `τ(x)`: the mean effect for seekers who look like `x`.
- **Policy value.** `V(π)`: mean outcome if rule `π` decided.

> **Q1 is an ATT. Q2 is a policy value under a budget. Q3 compares two policy values.**

#### [day1#budget-makes-a-ranking] With one place per five seekers, the question becomes a ranking

*Why the budget matters*

- Treat-all is not on the table: **20%** of seekers get a place.
- The best rule gives places to the **largest effects**, not the largest needs.
- A rule's gain is the sum of the effects of the seekers it picks.
- So the estimand of Q2 needs the CATE, then the value of a rule built from it.

**1 in 5** places per job seeker next year

#### [day1#risk-is-not-benefit] Risk is the outcome without coaching; benefit is the difference coaching makes

*The idea the week turns on*

- **Risk score.** Predicts `Y(0)`: who will not find work without help. Built to find need.
- **Benefit.** `Y(1) - Y(0)`: who finds work because of coaching. Small for those who find work anyway, and for those who do not either way.

> **What goes wrong.** A model with a high AUC for the outcome says nothing about who benefits. Ranking by it fills places with seekers coaching does not move.

#### [day1#effects-by-risk-decile] Employment falls steadily with risk; the effect of coaching does not follow

*The pilot, by risk decile (Lab 1.1)*

| Risk decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Employed without coaching | 77% | 69% | 67% | 64% | 58% | 53% | 60% | 46% | 39% | 24% |
| Effect of coaching (points) | 4.4 | 10.8 | 13.0 | 9.3 | 15.7 | 15.5 | 5.4 | 13.7 | 14.3 | 4.4 |

600 seekers per decile; each effect has a standard error of 3 to 4 points. Randomised pilot, coached minus uncoached.

#### [day1#top-fifth-risk-against-benefit] Places by risk would buy less than half the jobs that places by benefit buy

*The size of the trap (Lab 1.1, with the generator's truth)*

| Item | Value |
| --- | --- |
| Top fifth by benefit | 23.7 points |
| Top fifth by risk | 9.5 points |
| Every seeker | 10.2 points |

Mean true effect of coaching in each group. Only 24% of the risk fifth is also in the benefit fifth.

#### [day1#check-q3-estimand] Which estimand answers the director's third question?

*Checkpoint*

**Question:** Wijdan asks whether next year's places should go to the highest risk scores. Which quantity should you estimate first?

- A. The AUC of the risk score for unemployment at six months
- B. The value of top-fifth-by-risk against top-fifth-by-effect, same budget
- C. The effect of coaching on the seekers offered it in 2025
- D. The average effect of coaching over all seekers

#### [day1#lab-1-1] Three estimands and a risk trap

Lab 1.1 · 25 min

Write estimands for three questions from the director, then show on the randomised pilot that the agency's risk score and the benefit of coaching rank seekers differently.

[D1_1_estimands.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D1_1_estimands.ipynb)

- [ ] Write the three estimands (8 min)
- [ ] Effects by risk decile in the pilot (7 min)
- [ ] Top fifth by risk against top fifth by benefit (6 min)
- [ ] Answer the director's third question (4 min)
- [ ] ✱ Stretch: Write the estimand for the share of seekers whom coaching harms

### Session 2: Overlap first

#### [day1#s2] Overlap first

*Session 2*

- Fit a cross-fitted propensity model on 60 confounders
- Map the groups almost never offered coaching
- Decide between trimming and changing the estimand, and say what each costs

#### [day1#recap-propensity] The propensity score summarises why a seeker was offered coaching

*Recap from DS-303*

- `e(x) = P(offered | x)`, estimated from the confounders.
- Given no hidden confounders, comparing seekers with equal `e(x)` compares like with like.
- Balance and overlap are checked, never assumed.

- **DS-303, Day 2.** Logistic propensity, matching, weighting, trimming and augmented weighting are taught there. Here they run on 60 confounders with flexible models.

#### [day1#positivity] Positivity: every kind of seeker had some chance of each decision

*The assumption every method needs*

- **Where it holds.** Offered and not offered seekers look alike somewhere: the logs show what coaching did for people like them.
- **Where it fails.** A group never offered coaching has no offered twin. Any number for them comes from a model's extrapolation.

> **What goes wrong.** Weighting divides by `e(x)`. A score of 0.01 makes one decision count like a hundred.

#### [day1#crossfit-the-propensity] Score each decision with a model that never saw it

*Flexible models need honest predictions*

1. **1 · Split** five folds
2. **2 · Fit** on four folds
3. **3 · Predict** the fifth
4. **4 · Repeat** until each fold is scored

```python
# file: D1_2_overlap.py
cv = KFold(5, shuffle=True, random_state=U.SEED)
e = cross_val_predict(U.gbm_classifier(), X, t, cv=cv,
                      method="predict_proba")[:, 1]
```

#### [day1#overlap-plot] Most decisions overlap; a thin tail of seekers was almost never offered

*Lab 1.2, 40,000 decisions*

Figure: Cross-fitted propensity scores by arm. Dashed line: 0.05.

#### [day1#groups-almost-never-offered] Disabled seekers in Nijad and Rumh, the long-term unemployed and the over-55s were almost never offered

*The overlap map (Lab 1.2)*

| Region | Disability | Months unemployed | Age | Decisions | Offered |
| --- | --- | --- | --- | --- | --- |
| Nijad | yes | over 36 | 55 or under | 38 | 0.0% |
| Nijad | yes | 13 to 36 | 55 or under | 94 | 1.1% |
| Nijad | yes | 0 to 12 | 55 or under | 65 | 1.5% |
| Sahil | no | 0 to 12 | over 55 | 202 | 2.0% |
| Nijad | no | over 36 | 55 or under | 444 | 2.3% |
| Rumh | yes | 0 to 12 | 55 or under | 362 | 3.6% |

Six of the twelve cells with at least 30 decisions and an offer rate under 5%. Together the twelve hold 4.6% of decisions.

#### [day1#trim-or-change-the-estimand] Trimming changes the estimand; keeping the seekers changes what you can claim

*Two honest options*

- **Trim at 0.05.** Drops 5% of decisions. The estimate is now the effect for seekers caseworkers actually consider. Say who left.
- **Keep everyone.** The estimate covers the mapped groups only through the outcome model. Say so, and give no group number there.

> **What goes wrong.** Trimming quietly, then reporting the result as the effect for "all job seekers".

#### [day1#check-overlap] What does a propensity AUC of 0.95 tell you?

*Checkpoint*

**Question:** A colleague's propensity model for the 2026 offers has a cross-fitted AUC of 0.95 and calls it excellent. What do you conclude?

- A. The adjustment will be reliable because the model fits well
- B. Employment within six months is easy to predict from the logs
- C. Decisions are nearly determined by the covariates, so overlap is probably poor
- D. Cross-fitting has already solved any extreme-weight problem

#### [day1#lab-1-2] Groups almost never offered

Lab 1.2 · 25 min

Fit a cross-fitted propensity model on the 60 confounders of the logs, plot the scores by arm and map the groups almost never offered coaching by region, disability and months unemployed.

[D1_2_overlap.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D1_2_overlap.ipynb)

- [ ] A cross-fitted propensity model (7 min)
- [ ] Scores by arm (5 min)
- [ ] Map the groups almost never offered (8 min)
- [ ] Trim, or change the estimand (5 min)
- [ ] ✱ Stretch: Set your fitted scores against the probabilities the tool recorded

### Session 3: Double machine learning

#### [day1#s3] Double machine learning

*Session 3*

- Explain why plugging machine learning predictions into a regression biases the effect
- Estimate an average effect with cross-fitting by hand and with a library
- Report an estimate table with intervals against regression adjustment

#### [day1#sixty-confounders] Sixty confounders, and caseworkers who used them in curves and combinations

*Why the old tools struggle*

- 13 named attributes, the risk score and 46 fields of administrative history.
- Offers depend on some fields through thresholds and interactions.
- A linear regression with 60 controls assumes straight lines for all of them.
- A flexible model fits the curves, but its predictions carry bias of their own.

**60** confounders in the logs

#### [day1#plug-in-bias] Plugging machine learning predictions into an effect formula passes their bias straight through

*The tempting shortcut*

- **Plug-in.** Fit `m1`, `m0`, average `m1(x) - m0(x)`. Regularisation shrinks the models, and the effect shrinks or shifts with them.
- **Same-data fitting.** Fit and predict on the same rows: residuals look too small, intervals too narrow.

> **Two fixes, together.** An orthogonal score so small model errors cancel to first order, and cross-fitting so no decision helps predict itself.

#### [day1#the-irm-score] The DML score is the augmented weighting score of DS-303, with cross-fitted parts

*The interactive regression model*

```python
# file: D1_3_dml.py
psi = (m1 - m0
       + t * (y - m1) / e
       - (1 - t) * (y - m0) / (1 - e))
estimate = psi.mean()
se = psi.std(ddof=1) / np.sqrt(n)
```

- **Outcome models** give the first guess.
- **Weighted residuals** correct it.
- **Errors in one part** are cancelled by the other.

#### [day1#crossfitting-loop] Each fold's nuisance predictions come from models fitted on the other folds

*Cross-fitting, written out*

```python
# file: D1_3_dml.py
fold = U.folds(n, 5)
for f in range(5):
    a, b = fold != f, fold == f
    e[b]  = clf().fit(X[a], t[a]).predict_proba(X[b])[:, 1]
    m1[b] = clf().fit(X[a & (t == 1)], y[a & (t == 1)]).predict_proba(X[b])[:, 1]
    m0[b] = clf().fit(X[a & (t == 0)], y[a & (t == 0)]).predict_proba(X[b])[:, 1]
e = np.clip(e, 0.01, 0.99)
```

#### [day1#doubleml-in-four-lines] DoubleML wraps the folds, the score and the interval in one object

*The same thing with a library*

```python
# file: D1_3_dml.py
data = dml.DoubleMLData(df, y_col="employed_6m", d_cols="offered",
                        x_cols=list(X.columns))
irm = dml.DoubleMLIRM(data, ml_g=U.gbm_classifier(),
                      ml_m=U.gbm_classifier(), n_folds=5).fit()
print(irm.summary)
```

> **Examples, not endorsements.** EconML's `LinearDML`, CausalML and grf for R estimate the same quantity. Write it by hand once, then use a library you can check.

#### [day1#estimate-table] Regression with 60 controls missed by three points; double machine learning did not

*Lab 1.3, 40,000 decisions, true effect 10.5 points*

| Method | Estimate | 95% interval | Bias |
| --- | --- | --- | --- |
| Naive difference | 14.1 |  | +3.7 |
| Regression adjustment | 13.5 | 12.4 to 14.7 | +3.1 |
| DML by hand | 11.0 | 9.7 to 12.3 | +0.5 |
| DoubleML | 10.8 | 9.6 to 12.1 | +0.4 |

Percentage points of employment within six months. Average effect over all logged decisions.

#### [day1#check-crossfitting] Why fold the data before fitting the nuisance models?

*Checkpoint*

**Question:** A colleague fits the propensity and outcome models on all 40,000 decisions, then plugs their predictions into the same score. What is the main risk?

- A. The fit takes five times longer than with folds
- B. The estimate targets the effect on the offered instead of everyone
- C. No risk: the orthogonal score makes nuisance errors irrelevant
- D. Overfitted predictions shrink the residuals, biasing the estimate and its interval

#### [day1#lab-1-3] Double machine learning against regression

Lab 1.3 · 25 min

Estimate the average effect of coaching on the logs naively, by regression adjustment and by double machine learning, by hand and with DoubleML, and set each against the known effect.

[D1_3_dml.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D1_3_dml.ipynb)

- [ ] Naive and regression-adjusted estimates (5 min)
- [ ] Cross-fitting by hand (9 min)
- [ ] The same with DoubleML (6 min)
- [ ] One table: estimate, interval, bias (5 min)
- [ ] ✱ Stretch: The partially linear model against the interactive one

### Session 4: Nuisance models and hidden confounding

#### [day1#s4] Nuisance models and hidden confounding

*Session 4*

- Compare estimates across nuisance learners with their prediction losses
- Bound an estimate for a hidden confounder with sensitivity parameters
- Benchmark a bound against an observed covariate

#### [day1#learners-change-the-answer] The same score with four learners gave answers three points apart

*Lab 1.4, 40,000 decisions, true effect 10.5*

| Item | Value |
| --- | --- |
| Gradient boosting | 11.0 |
| Lasso logistic | 12.8 |
| Random forest | 12.9 |
| One split | 14.2 |

Estimates in points; standard errors 0.6 to 0.7. The learner with the lowest out-of-fold loss for the offer was the closest.

#### [day1#choose-by-loss] Choose nuisance learners by out-of-fold loss, before you see the estimate

*A rule you can defend*

| Learner | Offer log loss | Outcome log loss | Bias |
| --- | --- | --- | --- |
| Gradient boosting | 0.427 | 0.601 | +0.6 |
| Random forest | 0.439 | 0.608 | +2.5 |
| Lasso logistic | 0.447 | 0.603 | +2.3 |
| One split | 0.469 | 0.640 | +3.8 |

> **What goes wrong.** Trying learners until the estimate looks right is a forking path: the interval no longer means what it says.

#### [day1#the-2027-extract] The 2027 extract lost the caseworkers' motivation note, and the estimate rose by almost two points

*A system change*

**11.0 → 12.7** points, with and without motivation, cross-fitted by hand (truth 10.5)

- Motivated seekers were offered coaching more often.
- They also found work more often, coached or not.
- Without the column, coaching gets the credit.
- No learner can recover information that is not in the data.

#### [day1#omitted-variable-bounds] Two numbers describe a hidden confounder: how much it explains of the outcome and of the offer

*Sensitivity for causal machine learning*

- **`cf_y`.** Share of the outcome's remaining variation the confounder would explain.
- **`cf_d`.** How much it would change the weights of the offer model.

```python
# file: D1_4_sensitivity.py
irm27.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
print(irm27.sensitivity_summary)
```

#### [day1#reading-the-bound] A confounder of strength 0.03 could move the estimate anywhere from 8.4 to 17.2 points

*Lab 1.4 on the extract, 40,000 decisions*

- **12.8** DoubleML estimate on the extract
- **8.4 to 17.2** bound at cf_y = cf_d = 0.03
- **8.5%** robustness value

> **For the director.** "Coaching helped" survives a hidden factor weaker than 8.5% on both sides. "By 12.8 points" survives almost nothing.

#### [day1#what-the-bound-does-not-say] A bound tells you how strong something missing would have to be, not whether it exists

*Limits*

- Ask the people who made the decisions what they saw that the data do not hold.
- Benchmark against an observed covariate the caseworkers would recognise.
- Report the robustness value next to the estimate, every time.
- When the bound is wide, the answer is a randomised trial, which Day 4 designs.

#### [day1#check-hidden-confounder] Can a better learner fix the 2027 extract?

*Checkpoint*

**Question:** On the 2027 extract, without the motivation note, a colleague proposes to switch from gradient boosting to a deep neural network to remove the bias. What do you tell her?

- A. Yes: a more flexible model can learn motivation from the other columns
- B. No: report the estimate with a sensitivity bound and its robustness value
- C. Drop more columns so the model cannot overfit the remaining ones
- D. Raise the folds from five to ten to reduce the bias

#### [day1#lab-1-4] Swap learners, bound a hidden confounder

Lab 1.4 · 25 min

Re-estimate the effect with four nuisance learners, then take an extract without the motivation note and bound the estimate with DoubleML's sensitivity analysis.

[D1_4_sensitivity.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D1_4_sensitivity.ipynb)

- [ ] Four nuisance learners, one table (8 min)
- [ ] Nuisance losses beside each estimate (4 min)
- [ ] The extract without motivation (5 min)
- [ ] Bound it with a sensitivity analysis (8 min)
- [ ] ✱ Stretch: Benchmark the bound with an observed covariate

#### [day1#day-1-recap] An estimand first, overlap second, then an orthogonal score with honest folds

*Day 1 recap*

- **Risk is not benefit.** The top fifth by risk gained 9.5 points, by benefit 23.7.
- **Overlap first.** 4.6% of decisions sit in groups almost never offered: name them.
- **DML, then bounds.** 10.8 against a truth of 10.5; a missing column moves it, a bound says how far.

#### [day1#tomorrow-who-benefits] Who benefits: effects for each seeker, checked before anyone is targeted

*Tomorrow · Day 2*

Tomorrow you estimate conditional effects on the randomised pilot with three meta-learners and a causal forest, validate them on held-out data, and draw the uplift curves that turn them into places.

## Day 2: Who benefits

Conditional effects by three methods on the randomised pilot, checked on held-out data before anyone is targeted

#### [day2#title] Who benefits

*SDAIA Academy · Causal Machine Learning and Policy Evaluation*

Day 2: Conditional effects by three methods on the randomised pilot, checked on held-out data before anyone is targeted

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-402

#### [day2#review-day-1] Which number goes to the director for Q1?

*Review checkpoint · Day 1*

**Question:** On the logs, regression with 60 controls gives 13.5 points with a narrow interval; cross-fitted DML gives 10.8 points with a wider one. The DML robustness value is 8.5%. What do you report?

- A. The regression estimate, because its interval is narrower
- B. The average of the two estimates
- C. The DML estimate, its interval and its robustness value
- D. Nothing: the robustness value shows the effect may be zero

#### [day2#agenda] Who benefits

*Today*

- **Session 1 · Meta-learners**: S-, T- and X-learners turn any prediction model into an estimate of each seeker's effect, each with its own bias
- **Session 2 · Causal forests**: A causal forest splits on effect differences, honestly, after removing what predicts the outcome and the treatment
- **Session 3 · Validating effects**: An effect model is trusted only after held-out randomised data agree with it: calibration by group, the best linear predictor and ranking
- **Session 4 · Uplift for targeting**: Uplift and Qini curves show how much each extra place buys, and where a budget cuts the ranking

### Session 1: Meta-learners

#### [day2#s1] Meta-learners

*Session 1*

- Fit S-, T- and X-learners on a randomised pilot
- Explain why an S-learner shrinks effects towards zero
- Measure each learner against known effects and say why that is impossible on real data

#### [day2#why-the-pilot] A randomised pilot removes confounding, so every remaining problem is noise

*Where heterogeneous effects are learnt*

- A lottery within each office: the propensity is **0.5** for everyone.
- Any group fixed before the lottery can be compared directly.
- 6,000 seekers, a binary outcome: individual effects are never seen, and groups are noisy.

- **Real programmes.** Job-search programmes show effects that differ by person; Knaus, Lechner and Strittmatter (2022) estimated them with machine learning on Swiss administrative data.

#### [day2#three-meta-learners] Meta-learners turn a prediction model into an effect estimator

*Any classifier, three ways*

- **S-learner.** One model `μ(x, t)`; effect `μ(x,1) - μ(x,0)`.
- **T-learner.** Two models, one per arm; effect `μ1(x) - μ0(x)`.
- **X-learner.** Impute each seeker's effect with the other arm's model, then model the imputed effects.

#### [day2#s-learner-shrinks] One model with coaching as a feature tends to shrink effects towards zero

*S-learner*

```python
# file: D2_1_metalearners.py
m = clone(BASE).fit(np.column_stack([Xtr, ttr]), ytr)
one  = np.column_stack([X, np.ones(len(X))])
zero = np.column_stack([X, np.zeros(len(X))])
tau_s = m.predict_proba(one)[:, 1] - m.predict_proba(zero)[:, 1]
```

> **What goes wrong.** Regularised trees split on the strongest predictors of employment. Coaching is one column of nineteen; effects that differ by seeker need splits on coaching inside each branch.

#### [day2#t-learner-noise] Two separate models subtract their errors, and the difference looks like heterogeneity

*T-learner*

- **What it does well.** Lets each arm have its own shape; nothing forces effects to be small.
- **What it gets wrong.** Each model fits noise in its own arm; their difference adds the two noises.

**10.5 vs 10.1** spread of T-learner effects against the truth (points, sd)

#### [day2#x-learner-steps] The X-learner models imputed effects, which are smoother than outcomes

*X-learner*

1. **1 · T-learner** `μ1`, `μ0`
2. **2 · Impute** `y - μ0(x)` for the coached, `μ1(x) - y` for the others
3. **3 · Model** `τ1`, `τ0` on the imputed effects
4. **4 · Blend** `e·τ0 + (1-e)·τ1`

> **In a lottery** `e = 0.5`: the two halves count equally. With unequal arms, the imputations made with the larger arm's model count more.

#### [day2#learners-against-truth] Against the generator's truth, the X-learner made the smallest error

*Lab 2.1, 2,400 held-out seekers*

| Learner | Error (PEHE) | Correlation | Mean | Spread |
| --- | --- | --- | --- | --- |
| S-learner | 8.0 | 0.65 | 7.6 | 6.1 |
| T-learner | 8.9 | 0.64 | 8.5 | 10.5 |
| X-learner | 7.0 | 0.74 | 8.5 | 8.4 |
| Truth |  |  | 9.9 | 10.1 |

Points of employment within six months. Error and correlation need the truth; real data give you only the last two columns.

#### [day2#check-meta-learners] Why might the S-learner report almost no difference between seekers?

*Checkpoint*

**Question:** A colleague's S-learner on the pilot gives nearly the same effect, about 8 points, for every seeker. The trial was 1:1 and large. What is the most likely reason?

- A. Coaching truly has the same effect for everyone
- B. Confounding in the pilot has flattened the effects
- C. The pilot is too small for any learner to find differences
- D. Trees seldom split on coaching, so its interactions shrink

#### [day2#lab-2-1] S-, T- and X-learners on the pilot

Lab 2.1 · 25 min

Fit S-, T- and X-learners with gradient boosting on the pilot's training split and compare their effect estimates with the true effects the generator wrote.

[D2_1_metalearners.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D2_1_metalearners.ipynb)

- [ ] S-learner (6 min)
- [ ] T-learner (6 min)
- [ ] X-learner (8 min)
- [ ] Against the truth, one table (5 min)
- [ ] ✱ Stretch: Refit the three with a linear base model

### Session 2: Causal forests

#### [day2#s2] Causal forests

*Session 2*

- Fit a causal forest with local centring
- Read which covariates drive effect variation
- Describe how the effect changes with months unemployed and digital skills

#### [day2#split-on-effects] A causal forest splits where the effect changes, not where the outcome changes

*The causal forest's idea*

- **A prediction tree.** Splits on months unemployed because employment falls with it.
- **A causal tree.** Splits on months unemployed only if coaching works differently on each side.

#### [day2#local-centring] Local centring removes the outcome level; honesty keeps the splits from grading themselves

*Two ingredients*

- **Local centring.** Residualise `y` and `t` on `x` first, as in DML, so splits chase effects, not levels.
- **Honesty.** Half of each subsample picks the splits; the other half estimates the leaf effects.

#### [day2#causal-forest-code] CausalForestDML fits the nuisance models and the forest in one call

*EconML, as one example*

```python
# file: D2_2_causal_forest.py
cf = CausalForestDML(
    model_y=HistGradientBoostingRegressor(min_samples_leaf=40),
    model_t=HistGradientBoostingClassifier(min_samples_leaf=40),
    discrete_treatment=True, n_estimators=400,
    min_samples_leaf=20, cv=5, random_state=U.SEED)
cf.fit(y[tr], t[tr], X=X[tr])
tau = cf.effect(X)
```

> **Interchangeable.** grf for R is the reference implementation; CausalML offers uplift trees. The concepts carry over.

#### [day2#importance-of-drivers] Months unemployed, distance and age drive most of the variation in effects

*Lab 2.2, feature importances*

| Item | Value |
| --- | --- |
| Months unemployed | 0.23 |
| Distance to office | 0.20 |
| Age | 0.14 |
| Risk score | 0.10 |
| Prior wage | 0.09 |

Share of split importance, causal forest on the training split. The top five are the same with another seed.

#### [day2#effect-profiles] The effect peaks for seekers unemployed about a year; the forest sees the shape but flattens it

*Lab 2.2, held-out seekers*

Figure: Forest (solid) against the generator's truth (dashed). Right: forest effect by digital skills, in points.

#### [day2#importance-is-not-cause] A driver of effect variation is a description, not a lever

*What not to say*

- "Distance matters" does not mean moving seekers closer would raise the effect.
- Correlated covariates share importance: risk score and months unemployed.
- A zero importance may mean a correlated column took the splits.
- Say who gains more, and stop there until a trial tests a change.

#### [day2#check-causal-forest] What does honesty protect against?

*Checkpoint*

**Question:** An analyst grows a causal tree on all 3,600 training seekers and estimates each leaf's effect on the same seekers that chose the splits. What goes wrong?

- A. Leaf effects are inflated by the noise that chose the splits
- B. The splits follow employment levels rather than effects
- C. The estimates are confounded by who was coached
- D. The fit takes twice as long as it needs to

#### [day2#lab-2-2] A causal forest and its drivers

Lab 2.2 · 25 min

Fit a causal forest on the pilot's training split, rank the covariates that drive effect variation and draw the effect against months unemployed and digital skills.

[D2_2_causal_forest.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D2_2_causal_forest.ipynb)

- [ ] Fit the causal forest (8 min)
- [ ] Feature importances (5 min)
- [ ] Effect by months unemployed and digital skills (7 min)
- [ ] Name the drivers for the director (5 min)
- [ ] ✱ Stretch: Refit with another seed and compare the importances

### Session 3: Validating effects

#### [day2#s3] Validating effects

*Session 3*

- Check calibration of effect estimates by quantile group on held-out randomised data
- Test whether predictions carry signal with the best linear predictor
- Choose an estimator from calibration and ranking, with reasons

#### [day2#no-truth-column] Individual effects are never observed, so an effect model cannot be scored seeker by seeker

*The problem*

- No row has both outcomes: the usual test set metrics do not exist.
- Groups do have observable effects, if the groups were randomised.
- So validation compares **groups defined by the prediction** with their observed effects.

- **Held out and randomised.** 2,400 pilot seekers the models never saw, coached by the same lottery.

#### [day2#calibration-by-quintile] Sort by prediction into five groups and compare predicted with observed effects

*Check 1 · Calibration by group*

Figure: Held-out pilot, five groups of about 480 seekers; bars are 95% intervals. Dashed: perfect calibration.

#### [day2#best-linear-predictor] Regress an unbiased effect score on the prediction: a slope near 1 means right in scale

*Check 2 · The best linear predictor*

| Estimator | Slope | Standard error | Reading |
| --- | --- | --- | --- |
| S-learner | 1.35 | 0.31 | Signal; slightly timid |
| T-learner | 0.86 | 0.18 | Signal; slightly overconfident |
| X-learner | 1.20 | 0.23 | Signal; close to calibrated |
| Causal forest | 3.14 | 0.62 | Signal; far too timid |

#### [day2#ranking-top-fifth] Every effect estimator's top fifth gained more than twice what the risk score's did

*Check 3 · Ranking where the budget cuts*

| Item | Value |
| --- | --- |
| X-learner | 28.4 |
| S-learner | 28.1 |
| T-learner | 26.7 |
| Causal forest | 25.1 |
| Risk score | 11.5 |

Observed effect in each ranking's top fifth, held-out pilot, points. Each has a standard error near 4.4.

#### [day2#choose-with-reasons] Choose the estimator that ranks well and is right in scale, and say what the truth showed afterwards

*The choice*

- **Chosen: X-learner.** Top fifth 28.4 points; slope 1.20; quintiles rise in order.
- **Then the truth.** Smallest error (7.0) and true top-fifth gain of 20.8 points, against 21.0 for the forest.

> **What goes wrong.** Validating on the training seekers: every flexible estimator looks calibrated on the data it fitted.

#### [day2#check-validation] What does a best-linear-predictor slope of 3 tell you?

*Checkpoint*

**Question:** On held-out randomised seekers, an estimator's predicted effects range from 5 to 14 points and its best-linear-predictor slope is 3.1 (standard error 0.6). What do you conclude?

- A. The estimator has no useful signal
- B. It ranks well, but real differences are about three times larger
- C. It exaggerates differences between seekers
- D. Its average effect is three times too small

#### [day2#lab-2-3] Calibration and ranking per estimator

Lab 2.3 · 25 min

On the pilot's held-out split, check four supplied effect estimators for calibration by quintile, the best linear predictor slope and the gain in the top fifth, then choose one and justify it.

[D2_3_validate.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D2_3_validate.ipynb)

- [ ] Calibration by quintile, per estimator (8 min)
- [ ] The best linear predictor slope (5 min)
- [ ] Ranking: the gain in the top fifth (6 min)
- [ ] Choose and justify (6 min)
- [ ] ✱ Stretch: Repeat the checks on bootstrap draws of the held-out split

### Session 4: Uplift for targeting

#### [day2#s4] Uplift for targeting

*Session 4*

- Draw uplift and Qini curves from randomised data
- Compare targeting by estimated benefit with targeting by risk
- Set a threshold for one place per five seekers

#### [day2#uplift-by-share] The top tenth by estimated benefit gained 33 points; by risk, 5

*Lab 2.4, held-out pilot*

| Share coached | 10% | 20% | 30% | 50% | 100% |
| --- | --- | --- | --- | --- | --- |
| By estimated benefit | 33.4 | 25.1 | 28.6 | 22.7 | 13.5 |
| By risk score | 4.6 | 11.5 | 11.7 | 13.2 | 13.5 |

Observed effect among the top share, in points. Causal forest predictions; at 100% both rankings coach everyone.

#### [day2#qini-curves] The Qini curve of estimated benefit stays above random; the risk score's hugs the line

*The whole ranking at once*

Figure: Extra employed per 100 seekers in the population, when the top share is coached. Dotted line: one place in five.

#### [day2#threshold-for-the-budget] Offer coaching when the estimated benefit is at least 11.6 points

*From a rank to a rule*

**11.6** points: the threshold that selects one seeker in five

- Next year's seekers arrive one at a time; a threshold decides each.
- Watch the share above it: if it drifts from 20%, the population changed.
- Observed effect above it on the held-out pilot: 25.2 points.

#### [day2#places-change-hands] 84% of the places would go to different seekers, and about 117 more per thousand places would find work

*If the agency switched*

- **84%** of the benefit fifth is not in the risk fifth
- **21.0** true gain, benefit fifth (points)
- **9.3** true gain, risk fifth (points)

> **Who loses a place.** Most of the risk fifth: the long-term unemployed and the least qualified. Day 4 asks whether that is acceptable, and audits it.

#### [day2#check-qini] What does a Qini curve check?

*Checkpoint*

**Question:** An estimator's Qini curve lies well above the random line at every share. Which claim does that support?

- A. Its predicted effects are correct in size
- B. It will rank next year's seekers just as well
- C. Its top seekers gain more than a random pick would
- D. No other ranking could do better at any budget

#### [day2#lab-2-4] Uplift, Qini and a threshold

Lab 2.4 · 25 min

Draw uplift and Qini curves for targeting by estimated benefit and by risk on the held-out pilot, set the threshold at one place per five seekers and count the places that change hands.

[D2_4_qini.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D2_4_qini.ipynb)

- [ ] Uplift in the top shares (6 min)
- [ ] Qini curves against random and risk (7 min)
- [ ] The threshold at one place in five (6 min)
- [ ] Places that change hands (6 min)
- [ ] ✱ Stretch: Extra jobs per hundred places at three budgets

#### [day2#day-2-recap] Effects per seeker are estimated, then validated on randomised data, then turned into a threshold

*Day 2 recap*

- **Estimate.** S, T, X and a causal forest; regularise hard on a small pilot.
- **Validate.** Calibration by group, the best linear predictor, the top fifth, on held-out seekers.
- **Target.** Qini curves and a threshold; 84% of places would change hands.

#### [day2#tomorrow-rules-offline] Evaluating rules before using them

*Tomorrow · Day 3*

Tomorrow you value current practice and three candidate rules on 40,000 logged decisions with inverse propensity and augmented estimators, find the estimates you should not believe, and review an analyst's subgroup claim.

## Day 3: Evaluating rules before using them

The value of current practice and of candidate rules from logged decisions, with the diagnostics that say when not to believe it

#### [day3#title] Evaluating rules before using them

*SDAIA Academy · Causal Machine Learning and Policy Evaluation*

Day 3: The value of current practice and of candidate rules from logged decisions, with the diagnostics that say when not to believe it

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-402

#### [day3#review-day-2] Which check would you insist on before targeting with an effect model?

*Review checkpoint · Day 2*

**Question:** An effect model will decide next year's coaching places. Its builder shows a high correlation between predicted effects and observed outcomes on the training seekers. What do you ask for?

- A. Nothing more: a high correlation on 3,600 seekers is enough
- B. The correlation of its predictions with the risk score
- C. Bootstrap intervals for the same training correlation
- D. Observed effects by prediction group on held-out randomised seekers

#### [day3#agenda] Evaluating rules before using them

*Today*

- **Session 1 · Logged decisions**: Logged decisions with recorded probabilities can value rules never used, if the recorded probabilities are right
- **Session 2 · Weighted and augmented estimators**: Inverse propensity, self-normalised and augmented estimators value a rule from logs, each with an interval
- **Session 3 · Untrustworthy estimates**: A rule that treats people the logs almost never treated is valued by a few heavy weights: measure it and say so
- **Session 4 · Reviewing an effects claim**: A subgroup claim is reviewed for post-treatment groups, dredged cells, multiplicity and data used twice

### Session 1: Logged decisions

#### [day3#s1] Logged decisions

*Session 1*

- Compute the value of current practice from the logs
- Check recorded probabilities against the decisions they describe
- Decide what to do with an office whose records are wrong

#### [day3#what-the-logs-hold] Each logged decision carries what was known, what was done, how likely it was, and what happened

*Logged decisions*

1. **x** 60 confounders at registration
2. **p** offer probability the tool recorded
3. **a** offered or not
4. **y** employed within six months

> **The recorded `p` is the key.** When it is right, no propensity model is needed, and rules never used can be valued.

#### [day3#value-of-current-practice] Current practice needs no weights: its value is the employment rate in the logs

*The baseline every rule must beat*

**57.6%** employed within six months, 40,000 decisions (Lab 3.1)

- The logs were produced by current practice.
- Their mean outcome is its value, with a standard error of 0.2 points.
- Every candidate rule is compared with this number.

#### [day3#weights-average-to-one] If the recorded probabilities are right, the weights a/p and (1-a)/(1-p) average to 1

*A check that costs one line*

```python
# file: D3_1_logged.py
w1 = a / p
w0 = (1 - a) / (1 - p)
print(w1.mean(), w0.mean())
```

- **Fursa's logs.** Mean of `a/p`: 1.148, standard error 0.019. Eight standard errors from 1.
- **The other weight.** Mean of `(1-a)/(1-p)`: 0.976, standard error 0.003.

#### [day3#one-office-overrode] One office offered coaching two and a half times as often as its records say

*Lab 3.1, recorded against actual offers*

| Office | Decisions | Recorded | Offered | Mean a/p | z |
| --- | --- | --- | --- | --- | --- |
| RUM-2 | 3,961 | 13.6% | 33.6% | 2.61 | 38.9 |
| SAH-2 | 6,028 | 19.6% | 20.0% | 1.01 | 0.8 |
| RUM-1 | 4,028 | 13.5% | 13.2% | 1.03 | -0.6 |
| NIJ-2 | 1,588 | 8.7% | 7.3% | 0.75 | -2.0 |

Four of twelve offices. The other eight have z between -0.9 and 0.5.

#### [day3#exclude-or-re-estimate] Exclude the office or re-estimate its propensities, and say which

*What to do with a broken record*

- **Exclude.** Lose 10% of the data; every other office's records pass. The course's choice.
- **Re-estimate.** Fit a propensity model for RUM-2 alone, as on Day 1. Needed if a rule targets Rumh.
- **Keep.** Offer-to-no-one's IPW value: 54.0% with it, 55.5% without; truth 55.6%.

#### [day3#check-logged] What does a mean of a/p equal to 1.15 tell you?

*Checkpoint*

**Question:** In an agency's logs, the mean of offered divided by the recorded probability is 1.15, with a standard error of 0.02. What follows?

- A. Coaching raised employment by 15%
- B. More offers happened than recorded: some probabilities are too low
- C. Nothing: weights are noisy and a mean of 1.15 is normal
- D. The tool records probabilities that are too high

#### [day3#lab-3-1] Current practice from the logs

Lab 3.1 · 25 min

Recover the value of current practice from the logs, check that inverse weighting with the recorded probabilities gives it back, and find the office whose recorded probabilities do not match its offers.

[D3_1_logged.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D3_1_logged.ipynb)

- [ ] The value of current practice (4 min)
- [ ] Weight the logging rule by its own records (6 min)
- [ ] Recorded against actual offers, per office (9 min)
- [ ] Flag the office and decide (6 min)
- [ ] ✱ Stretch: A calibration plot of the recorded probabilities

### Session 2: Weighted and augmented estimators

#### [day3#s2] Weighted and augmented estimators

*Session 2*

- Write allocation rules as decisions per seeker
- Estimate a rule's value by inverse propensity and self-normalised weighting
- Estimate it by the augmented estimator and compare the intervals

#### [day3#ipw-value] Keep the decisions the rule agrees with, and weight each by one over its probability

*Inverse propensity value*

```python
# file: D3_2_offline.py
p_a = np.where(a == 1, p, 1 - p)     # probability of what the logs did
w   = (a == rule) / p_a               # zero where the rule disagrees
value = np.mean(w * y)
```

> **Recap from DS-303.** The same weights estimated an average effect there; a rule's value is the weighted mean outcome under one decision per seeker.

#### [day3#snipw-and-augmented] Self-normalising tames the weights; augmenting adds an outcome model and keeps the estimate honest

*Two refinements*

- **Self-normalised.** Divide by the sum of the weights, not by n. A little bias, much less variance.
- **Augmented.** Start from `μ` under the rule; add the weighted residual where logs and rule agree. Consistent if either part is right.

```python
# file: D3_2_offline.py
psi = np.where(rule == 1, mu1, mu0) + w * (y - np.where(a == 1, mu1, mu0))
```

#### [day3#three-candidate-rules] Three rules that each offer about one place in five

*The candidates*

- **Risk.** Top fifth by the agency's risk score.
- **Benefit.** Top fifth by the causal forest's estimated effect (fitted on the pilot).
- **Duration.** Months unemployed between 10 and 15. A caseworker can apply it by hand.

#### [day3#value-table] Benefit and duration both beat current practice by over two points; risk targeting does not

*Lab 3.2, 36,039 decisions without RUM-2*

| Rule | IPW | Self-normalised | Augmented | 95% interval | Truth |
| --- | --- | --- | --- | --- | --- |
| Duration 10 to 15 months | 59.7 | 59.7 | 59.9 | 59.0 to 60.7 | 59.9 |
| Benefit, top fifth | 59.8 | 59.6 | 59.8 | 59.0 to 60.7 | 59.7 |
| Current practice | 57.5 | 57.5 | 57.5 | 57.0 to 58.0 | 57.5 |
| Risk, top fifth | 56.8 | 57.7 | 57.0 | 56.1 to 57.9 | 57.4 |

% employed within six months. The truth column is the generator's; the other columns are what real logs give you.

#### [day3#simple-rules-compete] A rule a caseworker can read did as well as the black-box ranking

*What the table teaches*

- Risk targeting formalises what caseworkers already do: no gain.
- The duration rule captures the main driver of the effect in one line.
- Ties are decided by what you can explain, audit and defend: Day 4 learns such a rule on purpose.

#### [day3#check-augmented] When is the augmented value still right?

*Checkpoint*

**Question:** You value a rule with the augmented estimator. The recorded probabilities have been checked and are right; the outcome model is poor. What happens?

- A. The estimate is biased, because the outcome model is wrong
- B. The estimate is right and as precise as with a good model
- C. It stays consistent, with a wider interval
- D. The estimate equals the IPW value exactly

#### [day3#lab-3-2] Three rules offline

Lab 3.2 · 25 min

Evaluate risk targeting, benefit targeting and a simple duration rule on the logs with inverse propensity, self-normalised and augmented estimators, with intervals, against current practice.

[D3_2_offline.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D3_2_offline.ipynb)

- [ ] Write the three rules (5 min)
- [ ] Inverse propensity values with intervals (6 min)
- [ ] Self-normalised and augmented values (8 min)
- [ ] One value table and a ranking (6 min)
- [ ] ✱ Stretch: A bootstrap interval for the difference between two rules

### Session 3: Untrustworthy estimates

#### [day3#s3] Untrustworthy estimates

*Session 3*

- Compute the effective sample size of a rule's weights
- Find the decisions that carry a rule's value
- Show what clipping weights does to the estimate and its bias

#### [day3#reach-the-unreached] "Reach the unreached": coach the groups the logs almost never coached

*A rule from outside*

- Disabled seekers in Rumh and Nijad.
- Seekers unemployed more than 36 months.
- Seekers over 55.
- 8.1% of seekers; the logs offered them coaching in 2% to 5% of decisions.

- **A fair question.** The rule comes from an advocacy group. Its aim is legitimate; the question is whether the logs can value it.

#### [day3#effective-sample-size] 2,922 seekers to coach, 126 coached in the logs, worth 58 decisions

*Lab 3.3, the rule's weights*

- **126** of the 2,922 the logs actually coached
- **58** effective sample size of their weights
- **33%** of their weight in the ten heaviest decisions

> **Kish's effective sample size** `(Σw)² / Σw²`: how many equally weighted decisions the estimate is worth.

#### [day3#clipping-moves-the-answer] Clipping the weights moves the value by a point, which is the size of the differences between rules

*Lab 3.3, sensitivity to the heaviest weights*

| Item | Value |
| --- | --- |
| IPW | 55.9% |
| Clipped at 50 | 55.5% |
| Clipped at 20 | 54.8% |
| Augmented | 55.2% |

Employed within six months, all seekers, under the rule. The generator's truth: 56.0%.

#### [day3#diagnose-untrustworthy] Report the effective sample size, the heaviest weights and the clipped values with every rule's value

*A checklist for every value you report*

| Diagnostic | Warning sign |
| --- | --- |
| Share of rule-treated seekers the logs treated | Under one in ten |
| Effective sample size among them | Under a few hundred |
| Weight in the ten heaviest decisions | Over 10% |
| Change when clipping at 20 | Larger than the gap to the next rule |
| Recorded probabilities under 0.02 | Any decision the rule relies on |

#### [day3#check-heavy-weights] What do you tell the advocacy group?

*Checkpoint*

**Question:** The "reach the unreached" rule has an IPW value 1.6 points below current practice, an effective sample size of 58 among the seekers it would coach, and moves by a point when weights are clipped. What do you recommend?

- A. Reject the rule: the logs show it adds nothing
- B. Report the value clipped at 20 as the estimate
- C. Say the logs cannot value it; propose a small pilot
- D. Report the augmented value, which does not depend on the weights

#### [day3#lab-3-3] A rule resting on heavy weights

Lab 3.3 · 25 min

Value a rule that offers coaching to the groups the logs almost never offered it to, then show with effective sample sizes, the heaviest weights and clipping why its value cannot be trusted.

[D3_3_weights.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D3_3_weights.ipynb)

- [ ] The rule and its inverse propensity value (5 min)
- [ ] Effective sample size and the heaviest weights (7 min)
- [ ] Where the rule leaves the logs (6 min)
- [ ] Clip, compare and write the diagnosis (7 min)
- [ ] ✱ Stretch: The augmented value, and why it disagrees

### Session 4: Reviewing an effects claim

#### [day3#s4] Reviewing an effects claim

*Session 4*

- Recognise a subgroup defined after treatment
- Re-test a discovered subgroup on data not used to find it
- Name the effect of each fault on the claim

#### [day3#the-analysts-claim] An analyst found who benefits most, and recommends whom to coach first

*A draft on the director's desk*

- **1.** Seekers who attended four or more sessions gain 27.0 points.
- **2.** Men aged 40 to 44 in Sahil with secondary education gain 49.5 points (n = 36, p = 0.001).
- **3.** 8 of 60 cells had p < 0.05: the targets. In the others coaching showed no effect.
- **4.** Predicted and observed effects agree across deciles: correlation above 0.9.

#### [day3#post-treatment-subgroup] A subgroup defined after the lottery is selected, not randomised

*Fault 1*

**3.42 vs 2.50** motivation before the lottery: four or more sessions against fewer

- Only coached seekers have sessions; no comparable uncoached group exists.
- Attenders were more motivated before anything happened.
- The randomised effect for everyone offered is 10.6 points, not 27.0.

#### [day3#dredged-cell] The largest of 60 cells is mostly the noise that made it the largest

*Fault 2*

| Item | Value |
| --- | --- |
| The analyst's cell, claimed | 49.5 |
| The same cell, true effect | 9.7 |
| Honest search: top five, training | 23.3 |
| The same five, held out | 12.4 |

Points of employment. Discover on one split, evaluate on another.

#### [day3#many-cells] Eight significant cells out of sixty do not pick out the targets, and silence is not absence

*Fault 3*

- **The targets.** 8 cells at p < 0.05; after a Bonferroni correction, 2. Sixty tests at 0.05 give three by chance alone, even if coaching did nothing.
- **The "no effect" cells.** Their true effect averages 11.3 points: cells of 30 to 60 seekers cannot show it.

#### [day3#in-sample-agreement] A model checked on the seekers it was fitted to agrees with itself

*Fault 4*

- **0.99** decile correlation, fitted and checked on all seekers
- **0.81** fitted on training, checked on held-out seekers

> **The held-out number is still good.** The fault is not that the model is useless; it is that the evidence offered for it proves nothing.

#### [day3#review-memo] Name each fault, show its evidence, state its effect on the claim

*The review memo*

| Fault | Evidence | Effect on the claim |
| --- | --- | --- |
| Post-treatment subgroup | Motivation 3.42 against 2.50 | 27 points mixes coaching with motivation; randomised effect 11 |
| Same data for discovery and evaluation | Winners shrink from 23 to 12 points | The 50-point cell is mostly noise |
| Multiplicity; silence as absence | 2 survive correction; others gain 11 | Target list largely chance |
| In-sample validation | 0.99 against 0.81 | No evidence for the ranking offered |

#### [day3#check-review] Which fix answers the dredged cell?

*Checkpoint*

**Question:** A colleague wants to keep the analyst's best cell but make it credible. Which step does that?

- A. Report the p-value to three decimals so readers see it is 0.001
- B. Search more cells to show the finding is robust
- C. Bootstrap the cell's 36 seekers and report the interval
- D. Test the cell on seekers outside the search

#### [day3#lab-3-4] Four seeded faults

Lab 3.4 · 25 min

Review an analyst's subgroup claim on the pilot: find its four seeded faults, re-test each on held-out data or the truth, and write a review memo naming each fault and its effect.

[D3_4_review.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D3_4_review.ipynb)

- [ ] Read the claim and list what it rests on (4 min)
- [ ] The post-treatment subgroup (5 min)
- [ ] The dredged cell: search, then hold out (6 min)
- [ ] Multiplicity and the cells with no effect (5 min)
- [ ] In-sample against held-out agreement (5 min)
- [ ] ✱ Stretch: Tag each fault with its name in the national bias guide

#### [day3#day-3-recap] Check the records, value with intervals, refuse what rests on a few weights, review claims for reuse

*Day 3 recap*

- **Records.** Weights average to 1, by office.
- **Values.** IPW, self-normalised, augmented; intervals.
- **Trust.** Effective sample size, heaviest weights, clipping.
- **Review.** Post-treatment, dredging, multiplicity, in-sample.

#### [day3#tomorrow-choose-a-policy] Choosing and testing a policy

*Tomorrow · Day 4*

Tomorrow you learn a depth-two rule under one place per five seekers, decide which attributes may drive it, audit it by region, gender and disability status, size the trial that would confirm it, and choose your capstone brief.

## Day 4: Choosing and testing a policy

A rule learned under one place per five seekers, audited by region, gender and disability, and the trial that would confirm it

#### [day4#title] Choosing and testing a policy

*SDAIA Academy · Causal Machine Learning and Policy Evaluation*

Day 4: A rule learned under one place per five seekers, audited by region, gender and disability, and the trial that would confirm it

Data Science Track · 4 teaching days + capstone day · 4 sessions · 4 labs · DS-402

#### [day4#review-day-3] Which number do you put next to a rule's offline value?

*Review checkpoint · Day 3*

**Question:** A colleague reports a candidate rule's inverse propensity value as 61.2% employed, with a 95% interval. What else must the report show before anyone relies on it?

- A. Nothing: the interval already shows the uncertainty
- B. Its effective sample size, and how it moves when weights are clipped
- C. The outcome model's accuracy on the logs
- D. The rule's true value from the generator

#### [day4#agenda] Choosing and testing a policy

*Today*

- **Session 1 · Policy learning under a budget**: Doubly robust scores and an exhaustive shallow tree give a rule people can read, under a budget
- **Session 2 · Fair allocation**: Some attributes may drive a rule, some may only audit it: purpose limitation, the bias guide, an audit and a constraint with its price
- **Session 3 · The confirmatory trial**: An offline value is a hypothesis: the rule is confirmed by a trial against current practice, powered for the difference
- **Session 4 · Explaining a rule**: A rule is recommended in one page: what it does, what it is worth, whom it leaves out, who reviews the edge cases

### Session 1: Policy learning under a budget

#### [day4#s1] Policy learning under a budget

*Session 1*

- Compute doubly robust scores from randomised data
- Learn a depth-two policy tree under a budget
- Compare the rule with treat-all, risk targeting and current practice

#### [day4#ranking-or-rule] A ranking by estimated benefit is hard to explain; a short rule can be read at the counter

*Two ways to allocate*

- **Ranking.** Top fifth by a causal forest's estimate. Each seeker's place depends on 14 attributes and 400 trees.
- **Rule.** At most two questions about a seeker, each one a threshold. A caseworker can apply it, and a seeker can be told why.

> **Day 3 showed** a hand-written rule on months unemployed matched the ranking's value. Today the rule is learned, not guessed.

#### [day4#dr-scores] Doubly robust scores are noisy for one seeker and unbiased for any group

*The gain of each seeker*

```python
# file: D4_1_policy_tree.py
score = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)   # e = 0.5 in the pilot
```

- **8.8 points** mean score on the training split: the average effect
- **92 points** standard deviation of one seeker's score

#### [day4#exhaustive-tree] Search every pair of splits; keep the tree whose offered seekers have the largest summed gain

*Policy trees*

1. **Split 1** every allowed attribute, every threshold
2. **Split 2** the best split inside each side
3. **Leaves** offer where the summed score beats the cost

> **Unlike a CART tree,** it is not greedy: the first split is chosen for what the second makes possible.

#### [day4#budget-as-a-cost] A budget becomes a cost per place: raise it until the tree offers at most 20%

*One place per five seekers*

```python
# file: D4_1_policy_tree.py
for _ in range(12):
    cost = (lo + hi) / 2
    t = U.policy_tree(X, g, cost=cost)
    if U.predict_tree(t, X).mean() <= 0.2:
        best, hi = (t, cost), cost
    else:
        lo = cost
```

**17 points** the cost per place at which the tree fits the budget

A seeker earns a place only where coaching is expected to raise her chance of work by more than 17 points.

#### [day4#the-learned-rule] The search chose one band: unemployed more than 10 and at most 15 months

*Lab 4.1, the learned rule*

**months ≤ 10** no offer → **11 to 15 months** offer a place → **more than 15** no offer

Offers 16.3% of logged seekers. Six attributes were allowed; it used only months unemployed.

#### [day4#rule-against-alternatives] The rule finds about twice as many jobs per place as current practice

*Lab 4.1, on the logs*

| Rule | Offered | Employed | 95% interval | Jobs per 100 places |
| --- | --- | --- | --- | --- |
| Treat-all (five times the budget) | 100% | 65.0 | 63.3 to 66.7 | 9.5 |
| Depth-two rule | 16.3% | 58.9 | 58.1 to 59.7 | 20.7 |
| Current practice | 16.9% | 57.5 | 57.0 to 58.0 | 11.4 |
| Risk, top fifth | 20.0% | 57.0 | 56.1 to 57.9 | 7.4 |
| No one | 0% | 55.6 | 55.0 to 56.1 | · |

% employed within six months, augmented estimator, 36,039 decisions without RUM-2. Jobs per 100 places compared with offering no one.

#### [day4#check-policy-tree] Why learn the tree from scores, not from outcomes?

*Checkpoint*

**Question:** A colleague proposes to learn the allocation tree by predicting employment within six months and offering places to the leaves with the lowest predicted employment. What is wrong?

- A. It targets need, not gain: it rebuilds the risk score
- B. A depth-two tree is too simple to predict employment well
- C. The pilot is too small for prediction
- D. Nothing, if the leaves are chosen on held-out data

#### [day4#lab-4-1] A depth-two rule against the alternatives

Lab 4.1 · 25 min

Learn a depth-two policy tree from doubly robust scores on the pilot under one place per five seekers, and compare its value on the logs with treat-all, risk targeting and current practice.

[D4_1_policy_tree.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D4_1_policy_tree.ipynb)

- [ ] Doubly robust scores on the training split (6 min)
- [ ] An exhaustive depth-two tree under the budget (8 min)
- [ ] Value against treat-all, risk and current practice (7 min)
- [ ] Read the tree (4 min)
- [ ] ✱ Stretch: Depth one against depth two

### Session 2: Fair allocation

#### [day4#s2] Fair allocation

*Session 2*

- Decide which attributes may drive a rule and which may only audit it
- Audit a rule by region, gender and disability status
- Add a fairness constraint and measure its cost in value

#### [day4#attribute-register] Every attribute gets a role: drive the rule, only audit it, or stay out

*Before learning any rule*

- **Drive.** Months unemployed, digital skills, education, prior jobs, licence, distance: registered facts a caseworker can explain.
- **Audit.** Region, gender, disability, age: never used to decide; every result is checked across them.
- **Exclude.** Prior wage, dependants, the motivation rating, the risk score.

#### [day4#law-and-guidance] Purpose limitation, sensitive data and historical bias each remove an attribute

*What the register rests on*

| Source | What it says | Effect on the register |
| --- | --- | --- |
| PDPL, Article 10 | Process personal data only for the purposes they were collected for, with listed exceptions | Dependants out |
| PDPL, Article 1 | Health data are sensitive data | Disability audits, never drives |
| AI Bias Reference Guide | Historical bias: data that encode past unequal treatment carry it forward | Prior wage out |
| AI Ethics Principles | Fairness and transparency of automated decisions | Only attributes a caseworker can explain drive |

#### [day4#price-of-the-register] A tree allowed every attribute chose the same band: here the register costs nothing

*Lab 4.2, a tree free to use everything*

**58.9% = 58.9%** value with six allowed attributes, and with all thirteen

- The free tree added one split on the risk score that offers almost no one new.
- The two rules agree on practically every seeker.
- In another programme the excluded attributes could carry value: then report the register's price.

#### [day4#audit-selection-rates] The rule offers every group at similar rates; the benefit ranking leaves Nijad behind

*Lab 4.2, Fairlearn's MetricFrame*

| Rule | Attribute | Lowest group | Lowest rate | Highest | Ratio |
| --- | --- | --- | --- | --- | --- |
| Depth-two rule | Region | Nijad | 15.5% | 17.0% | 0.91 |
| Depth-two rule | Gender | Women | 16.2% | 16.4% | 0.99 |
| Depth-two rule | Disability | Without | 16.3% | 16.4% | 0.99 |
| Benefit, top fifth | Region | Nijad | 7.0% | 23.2% | 0.30 |

Selection rates: the share of each group offered a place.

#### [day4#proxies] The ranking reached Nijad through months unemployed and distance

*Leaving an attribute out does not remove it*

| Item | Value |
| --- | --- |
| Nijad: unemployed over 15 months | 52% |
| Other regions | 26% |
| Nijad: over 60 km from an office | 78% |
| Other regions | 1% |

#### [day4#regional-floor] A regional floor costs nothing measurable here: report it as an interval, not as free

*Lab 4.2, spending the last places*

- **National.** Rule, then the top estimated gains anywhere, to 20%. Nijad 15.9%.
- **Regional floor.** Rule, then the top gains in each region, to 20% of that region. Nijad 20.0%.

**−0.2 points** price of the floor, paired 95% interval −0.5 to 0.1; true price 0.0

#### [day4#check-fairness] What does an equal selection rate prove?

*Checkpoint*

**Question:** A rule offers places to disabled and non-disabled seekers at the same rate. A board member calls it fair to disabled seekers. What do you add?

- A. Nothing: equal rates are the definition of fairness
- B. The rule should use disability to offer more places
- C. Rates show who is offered, not who gains
- D. Equal rates show the rule ignores disability, which is discrimination

#### [day4#lab-4-2] Attributes, audit and a constraint

Lab 4.2 · 25 min

Decide which attributes may drive the rule and which may only audit it, audit the rule by region, gender and disability status, then add a regional floor and price it in jobs.

[D4_2_fairness.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D4_2_fairness.ipynb)

- [ ] Drive or audit: sort the attributes (6 min)
- [ ] Learn the rule on the allowed attributes (5 min)
- [ ] Audit by region, gender and disability (7 min)
- [ ] Add a regional floor and price it (7 min)
- [ ] ✱ Stretch: The same audit of the true gains with Fairlearn

### Session 3: The confirmatory trial

#### [day4#s3] The confirmatory trial

*Session 3*

- State the estimand of a trial of a rule against current practice
- Size the trial for the difference diluted by agreeing decisions
- Choose a randomisation unit and a stopping rule

#### [day4#trial-estimand] Randomise seekers to policies, not to coaching

*What the trial compares*

- **Arm A.** The depth-two rule decides whether the seeker is offered a place.
- **Arm B.** Current practice decides: the caseworker, with the allocation tool.

> **Estimand.** The difference in employment within six months between the arms, among all randomised seekers (intention to treat).

#### [day4#where-policies-disagree] The two policies disagree on a quarter of seekers: only they can show a difference

*Lab 4.3, rule against caseworker on the logs*

|  | Caseworker: no | Caseworker: offer |
| --- | --- | --- |
| Rule: no | 69.8% | 13.9% |
| Rule: offer | 13.3% | 3.0% |

5.3 points among the 27.2% who disagree is 1.4 points among everyone.

#### [day4#seekers-per-arm] A 1.4-point difference needs about 18,000 seekers per arm: nearly two years of registrations

*Lab 4.3, sizing*

- **18,246** seekers per arm, 5% two-sided, 80% power
- **1.8 years** of Fursa's registrations, about 19,900 a year
- **1,328** per arm if only disagreeing seekers could be enrolled

#### [day4#randomisation-unit] Randomising Fursa's 12 offices would multiply the sample by 109

*Seekers or offices?*

- **Offices.** No mixing of policies inside an office, but the design effect `1 + (m − 1) ICC` with ICC 0.036 and thousands of seekers per office is 109.
- **Seekers, stratified by office.** Feasible. Contamination is checked by caseworker compliance with the assigned policy.

> **What neither design sees:** displacement. Crépon and colleagues found job-search help for some seekers partly at the expense of others in the same labour markets.

#### [day4#stopping-rule] O'Brien–Fleming boundaries spend almost nothing at the halfway look

*Lab 4.3, one interim look*

- **|z| > 2.79** to stop at half the seekers
- **|z| > 1.97** at the end, instead of 1.96

Found by simulating pairs of correlated z-statistics until the chance of crossing either boundary under no difference is 5%. Simulated power with 1% more seekers: about 80%.

#### [day4#check-trial] Why is the trial so large when the rule finds twice the jobs per place?

*Checkpoint*

**Question:** The rule finds 20.7 jobs per 100 places against 11.4 for current practice, yet the trial needs about 18,000 seekers per arm. Why?

- A. The per-place gain is too small to detect
- B. The arms mostly agree, so across everyone the difference is 1.4 points
- C. The interim look doubles the sample size
- D. Offices are clustered, so the design effect inflates the sample

#### [day4#lab-4-3] Size the trial

Lab 4.3 · 25 min

Size a trial of the learned rule against current practice: the arms and the primary metric, where the two policies disagree, seekers per arm, the design effect of office clusters and a stopping rule.

[D4_3_trial.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D4_3_trial.ipynb)

- [ ] The estimand and the arms (4 min)
- [ ] Where the two policies disagree (6 min)
- [ ] Seekers per arm, then the design effect (8 min)
- [ ] A stopping rule and the protocol table (7 min)
- [ ] ✱ Stretch: Simulate the trial to check its power

### Session 4: Explaining a rule

#### [day4#s4] Explaining a rule

*Session 4*

- Summarise a rule, its value and its audit on one page
- Name the edge cases that go to human review
- Choose a capstone brief and plan its analysis

#### [day4#one-page] Five headings, in the order she will ask them

*What the director reads*

| Heading | What it says |
| --- | --- |
| What the rule does | The rule in one sentence; 16% of seekers offered a place |
| What it is worth | 58.9% employed against 57.5%: about 14 more in work per 1,000 seekers |
| Who it leaves out | The audit by group; the 29% unemployed over 15 months |
| Who reviews | The edge cases that go to caseworkers |
| What would confirm it | A trial of about two years against current practice |

#### [day4#edge-cases] A rule with thresholds needs human review where it is blind or arbitrary

*Lab 4.4, where the rule is weakest*

| Case | Seekers in the logs | Why review |
| --- | --- | --- |
| Exactly 10 or 16 months unemployed | 2,341 | One month from the band |
| Unemployed over 36 months | 1,505 | No evidence in the logs (Lab 3.3) |
| Disability, over 15 months | 721 | Gain more on average; the rule cannot see it |
| Months recorded as 0 | 624 | Often a re-registration |
| In the band, over 60 km away | 419 | May not be able to attend |

#### [day4#capstone-briefs] Choose one brief and run the whole week's chain on it

*The capstone*

- **Siraj Power Distribution.** Energy-saving visits; 15% of households; outcome: % of summer use saved.
- **Al-Nakheel food safety.** Unannounced inspections; 25% of restaurants; outcome: compliant at follow-up.
- **Fursa 2027.** A new pilot and logs; 20% of seekers; an independent check of the course.

Each has a randomised pilot, logs with recorded probabilities and a BRIEF.md with its budget, constraint and audit groups.

#### [day4#check-summary] Which sentence belongs on the director's page?

*Checkpoint*

**Question:** You are writing the "What it is worth" section. Which sentence is right for it?

- A. The policy's doubly robust value proves a CATE-driven uplift of 1.4 points
- B. The rule will employ 58.9% of seekers
- C. The rule guarantees more jobs than current practice
- D. On past records, about 14 more per 1,000 in work than today (6 to 22)

#### [day4#lab-4-4] One-page summary and the capstone brief

Lab 4.4 · 25 min

Draft a one-page summary of the rule for the director from the supplied results, list the edge cases for caseworker review, and choose the capstone brief.

[D4_4_summary.ipynb](https://sdaiaacademy.github.io/ds-402/labs/D4_4_summary.ipynb)

- [ ] Gather the numbers (5 min)
- [ ] Write the rule summary (10 min)
- [ ] Edge cases for human review (5 min)
- [ ] Choose the capstone brief (5 min)
- [ ] ✱ Stretch: Cut the summary to what the director would repeat

#### [day4#day-4-recap] Learn a readable rule, decide what may drive it, audit it, price constraints, plan the trial

*Day 4 recap*

- **Rule.** Depth two, under the budget, from DR scores.
- **Register.** Drive, audit, exclude; measure its price.
- **Audit.** Rates and gains by group; proxies.
- **Trial.** Dilution, unit, stopping, limits.

#### [day4#next-capstone] Your brief, your repository, your defence

*Next · Capstone*

The capstone asks for the week's chain on your brief, reported in a public repository, then a 15-minute defence in which the assessor changes the budget or the constraint and you predict what happens before you re-run.

## Capstone: Capstone: a policy evaluation and its defence

One brief, one constrained rule, its offline value, a subgroup audit and a trial plan, defended in 15 minutes

#### [capstone#title] Capstone: a policy evaluation and its defence

*SDAIA Academy · Causal Machine Learning and Policy Evaluation*

Capstone: One brief, one constrained rule, its offline value, a subgroup audit and a trial plan, defended in 15 minutes

Data Science Track · 4 teaching days + capstone day · 15 min talk and demo · DS-402

#### [capstone#agenda] How the defences run

*Capstone*

- **The briefs and final prep**: the rubric, the live change, a last rerun of each analysis
- **Defences**: fifteen minutes each, including the live change, one participant at a time
- **Review and closing**: peer feedback, the trainer's review, the follow-up task

#### [capstone#the-briefs] One brief, the whole chain of the week

*The briefs*

| Brief | Budget | Fairness constraint | Audit groups |
| --- | --- | --- | --- |
| Siraj Power Distribution: energy-saving visits | 15% of households | low-income households get at least their share of visits | district, income band, elderly member |
| Al-Nakheel food safety: unannounced inspections | 25% of restaurants | every district between half and twice the average rate | district, size band, chain |
| Fursa 2027: coaching places | 20% of seekers | every region at least 20% of its seekers | region, gender, disability |

Each brief: a randomised pilot, logs with recorded probabilities, a `BRIEF.md`. Submission rules: `labs/capstone/README.md`.

#### [capstone#readme-reports] The README reports measured results, in this order

*The submission*

1. The estimand, and the average effect with its overlap
2. Effect estimates validated on the held-out randomised split
3. A rule under the budget and the constraint, with the attribute register
4. Its offline value with intervals and diagnostics, against treat-all, risk or need targeting and current practice
5. The audit by the brief's groups, and the constraint's price
6. The confirmatory trial: arms, difference, seekers or units per arm, stopping rule

- **And with it.** Notebooks that rerun from the raw data with a fixed seed, the commit history, the run log, and a verification note on every assisted step.

#### [capstone#rubric] The rubric

*How evaluations are scored*

| Item | Value |
| --- | --- |
| Identification | 15% |
| Effects and validation | 25% |
| Offline evaluation | 25% |
| Rule and fairness audit | 20% |
| Trial plan and reproducibility | 15% |

> **Every number reruns.** A number the panel cannot regenerate from the repository scores as a claim, not a result.

#### [capstone#format] Fifteen minutes: the evaluation, then one live change

*Format*

- **1 · The question.** Who decides, the budget, the estimand. One slide.
- **2 · The evidence.** Effects validated, the rule, its value against the alternatives, the audit.
- **3 · The live change.** The panel changes the budget or the constraint; predict, then rerun.

> **Peer feedback.** After each defence, everyone writes one strength, one question and one suggestion.

#### [capstone#live-change] One change to the budget or the constraint, predicted before it is rerun

*What the panel changes*

- **The change.** Halve the budget, double it, or remove the fairness constraint. The participant first predicts the direction and rough size of the change in the rule's value and in the audit, then reruns and shows it.

- "With half the places, does the value per place rise or fall? Why?"
- "Without the constraint, which group loses places, and how many outcomes do you gain?"
- "What is the effective sample size of your rule's weights now?"
- "How many units per arm would the trial need at the new budget?"

#### [capstone#running-order] Who defends when

*Running order*

| # | Participant | Brief |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first defence*

Rerun your analysis from a clean clone, rehearse once, open your repository on this screen.

#### [capstone#now-presenting] Evaluation and live change

*Now defending*

#### [capstone#review] What the defences showed

*Trainer review*

- **Strongest patterns.** What several evaluations got right.
- **Common gaps.** What most evaluations would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **Follow-up task:** within three weeks, review one eligibility rule from your workplace or a public service (`labs/followup/README.md`).
- **Your evaluation stays yours:** keep it rerunnable and add the check you named as the next one.
- **Measuring programmes at scale:** DS-406 Measuring the Impact of Data and AI Programmes.

#### [capstone#thanks] Thank you

*SDAIA Academy*

Causal Machine Learning and Policy Evaluation
