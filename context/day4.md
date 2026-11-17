---
course: "Causal Machine Learning and Policy Evaluation"
code: "DS-402"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-402/slides/day4.html"
---

# Causal Machine Learning and Policy Evaluation

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
