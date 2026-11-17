# Brief · Al-Nakheel Municipality Food Safety Department: unannounced inspections

The Food Safety Department of Al-Nakheel Municipality (fictional) makes unannounced inspections of restaurants. Next year it can inspect **25% of licensed restaurants** in the first quarter. The head of the department asks: do inspections raise compliance, where do they raise it most, and which restaurants should be inspected?

## Data

| File | Rows | What it holds |
|---|---|---|
| `pilot.csv` | 3,000 | a randomised pilot: half the restaurants inspected by lottery; `split` marks a training (60%) and a held-out (40%) part |
| `logged.csv.gz` | 15,000 | last year's inspections as inspectors chose them, with the probability the scheduling tool recorded (`p_treat`) |
| `truth/` | | each restaurant's chance of compliance without and with an inspection, written by the generator |

Treatment `inspected` (0/1). Outcome `compliant_followup`: compliant at the follow-up visit three months later (0/1). Covariates: `district`, `seats`, `years_open`, `prior_violations`, `complaints_12m`, `chain`, `cuisine`, `size_band` and twenty licensing-history columns `lic_01` to `lic_20`.

## What the rule must respect

- **Budget**: at most 25% of restaurants inspected.
- **Attributes that may drive the rule**: decide and justify them, as in Lab 4.2.
- **Fairness constraint**: no district is inspected at less than half, or more than twice, the average rate. Price the constraint.
- **Audit groups**: `district`, `size_band`, `chain`.
- **Comparison rules**: inspect everyone (for scale only), risk targeting (the top 25% by `prior_violations`, then `complaints_12m`), and last year's practice.

## Notes

- One district was rarely inspected last year: check overlap before trusting any value there.
- A restaurant with the worst record is not necessarily the one an inspection changes most.
- `truth/` is for checking your estimates after you have made them. Your README reports estimates made without it, then may add a line comparing them with the truth.
