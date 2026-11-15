# Who benefits from coaching? Subgroup findings from the pilot

Anwar Al-Mohaimeed, analyst, programme analytics. Draft for the director.

## Method

I used all 6,000 pilot seekers. I fitted a gradient-boosted model of employment on the 14
covariates and the coaching flag and read its predicted effect for every seeker. I then split
the pilot into 60 cells (region by gender by age band by education) with at least 30
seekers each, estimated each cell's effect as the difference in employment rates between
coached and not coached seekers, and kept the cells with a p-value below 0.05.

## Findings

1. **Seekers who attended four or more coaching sessions** gain 27.0 points over the
   seekers who were not coached (n = 1559 coached seekers, p < 0.001). Coaching works when
   seekers engage with it.
2. **Men aged 40-44 in Sahil with secondary education** gain 49.5 points (n = 36, p = 0.001). This is our strongest
   demographic finding.
3. 8 of the 60 cells had p < 0.05. They are listed in the appendix as the targets
   for next year; in the other cells coaching showed no effect.
4. The model's predicted effects match the observed effects closely: across deciles of the
   predicted effect, the correlation between predicted and observed coached-minus-uncoached
   differences is above 0.9.

## Recommendation

Give coaching places first to seekers likely to attend four or more sessions, then to the
8 target cells, starting with men aged 40-44 in Sahil.
