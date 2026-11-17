# Brief · Fursa Employment Agency: coaching places for 2027

Fursa Employment Agency (fictional, the course's case) has run a second, smaller pilot and kept a new year of logs. For 2027 it again has **one coaching place for every five job seekers**. Wijdan Al-Hazzaa, director of employment services, asks for a fresh analysis on the new data: not a copy of the course's results, but an independent check of them.

## Data

| File | Rows | What it holds |
|---|---|---|
| `pilot.csv` | 4,000 | a new randomised pilot, coached by lottery; `split` marks a training (60%) and a held-out (40%) part |
| `logged.csv.gz` | 20,000 | new caseworker decisions with the probability the allocation tool recorded (`p_offer`) |
| `truth/` | | each seeker's chance of employment without and with coaching, written by the generator |

Same columns as the course data (`data/pilot.csv`, `data/logged.csv.gz`). Treatment `coached` in the pilot and `offered` in the logs; outcome `employed_6m`.

## What the rule must respect

- **Budget**: at most 20% of seekers offered a place.
- **Attributes that may drive the rule**: decide and justify them; region, gender, disability and age are audit attributes.
- **Fairness constraint**: every region receives at least 20% of its seekers in places, or a constraint you argue is better. Price it.
- **Audit groups**: `region`, `gender`, `disability`.
- **Comparison rules**: treat-all (for scale only), risk targeting (the top fifth by `risk_score`) and current practice.

## Notes

- Check the recorded probabilities office by office before using them (Lab 3.1).
- Choosing this brief means re-running the course's methods on new data. Credit goes to what you check and find, not to reproducing the course's numbers.
- `truth/` is for checking your estimates after you have made them. Your README reports estimates made without it, then may add a line comparing them with the truth.
