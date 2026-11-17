---
course: "Causal Machine Learning and Policy Evaluation"
code: "DS-402"
version: "2027.1.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-402/slides/day3.html"
---

# Causal Machine Learning and Policy Evaluation

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
