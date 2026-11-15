# Brief · Siraj Power Distribution: home energy-saving visits

Siraj Power Distribution (fictional) sends an adviser to homes before the summer to check air conditioning, insulation and settings. Next summer it can visit **15% of its household accounts**. The head of customer programmes asks: did the visits save energy, which homes save most, and which homes should be visited next summer?

## Data

| File | Rows | What it holds |
|---|---|---|
| `pilot.csv` | 5,000 | a randomised pilot: half the households visited by lottery; `split` marks a training (60%) and a held-out (40%) part |
| `logged.csv.gz` | 30,000 | last summer's visits as planners chose them, with the probability their scheduling tool recorded (`p_treat`) |
| `truth/` | | each household's expected saving without and with a visit, written by the generator |

Treatment `visited` (0/1). Outcome `kwh_saved_pct`: percentage of summer electricity saved against the household's own baseline (continuous, can be negative). Covariates: `district`, `household_size`, `home_age_years`, `summer_kwh_month`, `ac_units`, `elderly_member`, `income_band`, `owner_occupied`, `smart_meter` and twenty account-history columns `acct_01` to `acct_20`.

## What the rule must respect

- **Budget**: at most 15% of households visited.
- **Attributes that may drive the rule**: decide and justify them, as in Lab 4.2. `income_band` and `elderly_member` are audit attributes: they may not drive the rule.
- **Fairness constraint**: low-income households receive at least their share of visits (the share of visits going to `income_band == "low"` is at least their share of households). Price the constraint.
- **Audit groups**: `district`, `income_band`, `elderly_member`.
- **Comparison rules**: visit everyone (for scale only), need targeting (the top 15% by `summer_kwh_month`), and last summer's practice.

## Notes

- One district was rarely visited last summer: check overlap before trusting any value there.
- `truth/` is for checking your estimates after you have made them. Your README reports estimates made without it, then may add a line comparing them with the truth.
