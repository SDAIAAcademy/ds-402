---
course: "Causal Machine Learning and Policy Evaluation"
code: "DS-402"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-402/slides/day2.html"
---

# Causal Machine Learning and Policy Evaluation

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
