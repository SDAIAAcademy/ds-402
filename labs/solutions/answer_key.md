# Answer key for trainers

What each lab plants, what a correct run shows, and the numbers to expect from the solutions run in full mode. `DS402_QUICK=1` samples the logs and coarsens the searches, so its numbers differ slightly. The complete solutions are the notebooks in this folder; the generator's true values are in `../data/truth.json`, `../data/truth_pilot.csv` and `../data/truth_logged.csv.gz`. Participants can read this folder: it is public with the rest of the repository.

Effects are in percentage points of the probability of employment within six months. Values of rules are employment rates in %.

## Pre-work

| Notebook | What a correct run shows |
|---|---|
| P0 Potential outcomes and propensity scores | Pilot: true ATE 10.2, ATT 9.9; difference in means 10.6 (8.2 to 13.1). Logs without RUM-2: naive 14.6 against a true ATT of 11.7. A logistic propensity model gives disabled seekers in Rumh and Nijad 12.8% although they were offered 2.0%; weighting on 14 covariates: 13.6 against a true ATE of 10.4; largest SMD 0.38 → 0.02. AUC 0.66 |

## Day 1 · Average effects at scale

| Lab | Planted | What a correct run shows |
|---|---|---|
| 1.1 Three estimands | Risk is not benefit (`data/requests.json`) | Q1 an ATT, Q2 a policy value and its gain, Q3 a difference in policy value at the same budget. Pilot employment 61.0%. Top fifth by risk truly gains 9.5, top fifth by benefit 23.7, everyone 10.2; the two fifths share 24% of seekers. Share harmed 18.2% (never observable) |
| 1.2 Overlap | Groups almost never offered | AUC 0.731; scores under 0.05 for 5.6% of the not offered and 0.6% of the offered; 12 mapped cells hold 4.6% of decisions (Nijad with disability or long durations, Rumh with disability, over-55s); trimming at 0.05 drops 5% |
| 1.3 DML against regression | Nonlinear confounding through administrative history | Truth 10.5. Naive 14.1, regression 13.5 (12.4 to 14.7), DML by hand 11.0 (9.7 to 12.3), DoubleML IRM 10.8 (9.6 to 12.1); PLR 11.9 |
| 1.4 Learners and a hidden confounder | Motivation missing from a later extract | Gradient boosting 11.0, lasso 12.8, random forest 12.9, one split 14.2. Without motivation 12.7; bound at cf_y = cf_d = 0.03: 8.4 to 17.2; robustness value 8.5%; a confounder as strong as months unemployed: 10.9 to 14.8 |

## Day 2 · Who benefits

| Lab | Planted | What a correct run shows |
|---|---|---|
| 2.1 Meta-learners | Effect peaks near 12 months unemployed; low digital skills, disability and youth add | Held-out truth: mean 9.9, sd 10.1. PEHE and correlation: S 8.0 / 0.65 (sd 6.1), T 8.9 / 0.64 (sd 10.5), X 7.0 / 0.74 (sd 8.4). Linear versions: correlations near 0.3 |
| 2.2 Causal forest | Same | Held-out correlation 0.75, sd 3.0 (too timid). Importances: months unemployed 0.23, distance 0.20, age 0.14, risk score 0.10, prior wage 0.09 |
| 2.3 Calibration and ranking | Same | Held-out observed ATE 13.5. BLP slopes: S 1.35 (0.31), T 0.86 (0.18), X 1.20 (0.23), forest 3.14 (0.62). Top fifth observed: S 28.1, T 26.7, X 28.4, forest 25.1, risk 11.5 |
| 2.4 Uplift and Qini | Same | Uplift at 10/20/30/50/100%: benefit 33.4/25.1/28.6/22.7/13.5, risk 4.6/11.5/11.7/13.2/13.5. Qini area 15.1 against 0.6 per thousand; threshold 11.6 points, 25.2 above it; 84% of places change hands; true gain 21.0 against 9.3 |

## Day 3 · Evaluating rules offline

| Lab | Planted | What a correct run shows |
|---|---|---|
| 3.1 Current practice | Office RUM-2 overrode the tool | Current practice 57.6% (se 0.2). Mean a/p 1.148 (se 0.019), mean (1-a)/(1-p) 0.976. RUM-2: recorded 13.6%, offered 33.6%, weight 2.61, z 38.9; all other offices within z of -2.05 (NIJ-2) to 0.77. Offer to no one: 54.0% with RUM-2, 55.5% without, truth 55.6% |
| 3.2 Three rules offline | Risk targeting gains nothing | 36,039 decisions. Augmented: duration 10 to 15 months 59.9 (59.0 to 60.7), benefit 59.8 (59.0 to 60.7), current 57.5, risk 57.0 (56.1 to 57.9); truths 59.9, 59.7, 57.5, 57.4. Augmented se 0.43 against IPW 0.73 for benefit. Stretch: benefit minus duration -0.1 (-0.7 to 0.6) |
| 3.3 Heavy weights | The never-offered groups | Rule offers 8.1%; IPW 55.9% (54.5 to 57.2). 2,922 seekers to coach, 126 coached in the logs, ESS 58, ten heaviest weights 33%, largest weight 100. Clipped at 50: 55.5, at 20: 54.8; augmented 55.2; truth 56.0. Diagnosis: the logs cannot value the rule; a small randomised pilot can |
| 3.4 Four seeded faults | `data/review/anwar_analysis.md` | 1: post-treatment subgroup, motivation 3.42 against 2.50, ITT 10.6 against a claimed 27.0. 2: dredged cell (men 40-44, Sahil, secondary, n = 36, claimed 49.5), true 9.7; top five training cells 23.3 → 12.4 held out. 3: 8 of 60 cells at p < 0.05, 2 after Bonferroni; "no effect" cells truly gain 11.3. 4: decile correlation 0.99 in sample, 0.81 held out. Bias guide: data-dredging bias (2, 3), evaluation bias (4) |

## Day 4 · Choosing and testing a policy

| Lab | Planted | What a correct run shows |
|---|---|---|
| 4.1 Depth-two rule | Effect peaks near a year unemployed | DR scores mean 8.8, sd 92. Cost per place 17 points; the tree offers months 11 to 15 (15.8% of training, 16.3% of logs). Values: treat-all 65.0, rule 58.9 (58.1 to 59.7), current 57.5, risk 57.0, no one 55.6; jobs per 100 places 9.5, 20.7, 11.4, 7.4. Truths 66.0, 59.0, 57.5, 57.4, 55.6. Stretch: depth one (digital skills ≤ 2) offers 14.5%, 58.3% |
| 4.2 Attributes and audit | Nijad: longer durations, farther from offices | A tree on all 13 numeric attributes agrees with the allowed-attribute rule on every seeker (58.9%). Selection-rate ratios of the rule: region 0.91, gender 0.99, disability 0.99. Benefit ranking: Nijad 7.0% against 23.2% (ratio 0.30). Nijad: 52% over 15 months, 78% over 60 km. Filling to 20%: national 59.5 (Nijad 15.9%), regional floor 59.7 (Nijad 20%); price -0.2 (-0.5 to 0.1), true 0.0. Stretch: true gain per offered seeker 21.8 in most regions, 16.2 in Nijad; disabled 26.2 against 21.0 |
| 4.3 Size the trial | Dilution by agreeing decisions | Disagreement 27.2%; current 57.5, rule 58.9; 1.4 overall, 5.3 among disagreeing seekers. 18,246 per arm, 1.8 years at about 19,900 registrations a year; ICC 0.036, design effect for office randomisation 119; 1,328 per arm if only disagreeing seekers could be enrolled. O'Brien-Fleming 2.79 and 1.97. Stretch: simulated power 78.7% (without the 1% inflation) |
| 4.4 One-page summary | | Facts: share 16.3%, value 58.9 (58.1 to 59.7), current 57.5, about 14 more in work per 1,000, region ratio 0.91, 29% unemployed over 15 months. The page has the five headings and about 275 words. Edge cases: 10 or 16 months 2,341; over 36 months 1,505; disabled over 15 months 721; months recorded 0: 624; in the band over 60 km: 419 |

## Capstone briefs

| Brief | True average effect (pilot / logs) | Planted |
|---|---|---|
| `siraj-energy-visits` | 2.68 / 2.71 points of summer use saved | Effect largest in homes over 15 years old, with four or more air conditioners and high use; smaller for tenants. Plateau district rarely visited in the logs |
| `nakheel-inspections` | 9.4 / 9.3 points of compliance | Effect largest for restaurants with one to three prior violations and those open two years or less; smaller for chains. Industrial district rarely inspected |
| `fursa-coaching-2027` | see `truth/` | A new draw of the course's generator (4,000 pilot seekers, 20,000 decisions), with the same planted problems, including the overriding office |
