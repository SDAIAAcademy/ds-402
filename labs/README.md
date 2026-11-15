# Causal Machine Learning and Policy Evaluation: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Estimands for allocation | [`D1_1_write_estimands_three.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_1_write_estimands_three.ipynb) |
| Day 1 | Session 2 · Overlap first | [`D1_2_map_groups_almost.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_2_map_groups_almost.ipynb) |
| Day 1 | Session 3 · Double machine learning | [`D1_3_estimate_average_effect.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_3_estimate_average_effect.ipynb) |
| Day 1 | Session 4 · Nuisance models and hidden confounding | [`D1_4_swap_nuisance_learners.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D1_4_swap_nuisance_learners.ipynb) |
| Day 2 | Session 1 · Meta-learners | [`D2_1_fit_s_t.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_1_fit_s_t.ipynb) |
| Day 2 | Session 2 · Causal forests | [`D2_2_fit_causal_forest.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_2_fit_causal_forest.ipynb) |
| Day 2 | Session 3 · Validating effects | [`D2_3_check_calibration_ranking.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_3_check_calibration_ranking.ipynb) |
| Day 2 | Session 4 · Uplift for targeting | [`D2_4_draw_uplift_qini.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D2_4_draw_uplift_qini.ipynb) |
| Day 3 | Session 1 · Logged decisions | [`D3_1_recover_value_current.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_1_recover_value_current.ipynb) |
| Day 3 | Session 2 · Weighted and augmented estimators | [`D3_2_evaluate_three_candidate.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_2_evaluate_three_candidate.ipynb) |
| Day 3 | Session 3 · Untrustworthy estimates | [`D3_3_diagnose_rule_whose.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_3_diagnose_rule_whose.ipynb) |
| Day 3 | Session 4 · Reviewing an effects claim | [`D3_4_find_seeded_faults.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D3_4_find_seeded_faults.ipynb) |
| Day 4 | Session 1 · Policy learning under a budget | [`D4_1_learn_depth_two.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_1_learn_depth_two.ipynb) |
| Day 4 | Session 2 · Fair allocation | [`D4_2_decide_which_attributes.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_2_decide_which_attributes.ipynb) |
| Day 4 | Session 3 · The confirmatory trial | [`D4_3_size_trial_learned.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_3_size_trial_learned.ipynb) |
| Day 4 | Session 4 · Explaining a rule | [`D4_4_draft_one_page.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ds-402/blob/main/labs/D4_4_draft_one_page.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
