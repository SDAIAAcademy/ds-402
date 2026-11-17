---
course: "Causal Machine Learning and Policy Evaluation"
code: "DS-402"
version: "2027.1.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-402/slides/day1.html"
---

# Causal Machine Learning and Policy Evaluation

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
