---
course: "Causal Machine Learning and Policy Evaluation"
code: "DS-402"
version: "2027.1.0"
deck: "capstone"
day: 5
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ds-402/slides/capstone.html"
---

# Causal Machine Learning and Policy Evaluation

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
