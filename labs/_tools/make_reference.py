"""Write the supplied reference states of the DS-402 labs into labs/data/reference/ (fixed seeds).

    cd labs
    uv run python _tools/make_reference.py

These are the starting points that let a lab begin without an earlier one:
    cate_pilot.csv          four CATE estimates for every pilot seeker (S-, T-, X-learner, causal forest),
                            fitted on the training split; test rows are out of sample (Labs 2.3, 2.4, 3.4, 4.1)
    cate_logged.csv.gz      for every logged decision: the causal forest's CATE fitted on the whole pilot,
                            and cross-fitted outcome predictions mu0, mu1 on the logs (Labs 3.2, 3.3, 4.1, 4.2)
    dr_pilot.csv            doubly robust effect scores for the pilot's training split (Labs 4.1, 4.2)
    rule_tree.json          the reference depth-two rule on the attributes allowed to drive it (Labs 4.3, 4.4)
    results.json            the numbers of the reference rule that Lab 4.4 summarises
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import fursa as U  # noqa: E402

ALLOWED = ["months_unemployed", "digital_skills", "education", "prior_jobs", "has_licence", "distance_km"]


def meta_learners(Xtr, ttr, ytr, Xall):
    from sklearn.base import clone
    base = U.pilot_classifier(1)
    reg = U.pilot_regressor(2)
    # S
    s = clone(base).fit(np.column_stack([Xtr, ttr]), ytr)
    s_cate = s.predict_proba(np.column_stack([Xall, np.ones(len(Xall))]))[:, 1] - s.predict_proba(np.column_stack([Xall, np.zeros(len(Xall))]))[:, 1]
    # T
    m1 = clone(base).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = clone(base).fit(Xtr[ttr == 0], ytr[ttr == 0])
    t_cate = m1.predict_proba(Xall)[:, 1] - m0.predict_proba(Xall)[:, 1]
    # X (propensity 0.5 by design)
    d1 = ytr[ttr == 1] - m0.predict_proba(Xtr[ttr == 1])[:, 1]
    d0 = m1.predict_proba(Xtr[ttr == 0])[:, 1] - ytr[ttr == 0]
    tau1 = clone(reg).fit(Xtr[ttr == 1], d1)
    tau0 = clone(reg).fit(Xtr[ttr == 0], d0)
    x_cate = 0.5 * tau1.predict(Xall) + 0.5 * tau0.predict(Xall)
    return s_cate, t_cate, x_cate


def forest(Xtr, ttr, ytr, seed=0):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    cf = CausalForestDML(model_y=HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=40, random_state=U.SEED),
                         model_t=HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=40, random_state=U.SEED),
                         discrete_treatment=True, n_estimators=400, min_samples_leaf=20, max_depth=None,
                         cv=5, random_state=U.SEED + seed)
    cf.fit(ytr, ttr, X=Xtr)
    return cf


def main():
    t0 = time.time()
    U.REF.mkdir(parents=True, exist_ok=True)
    P = U.pilot()
    X = U.design(P)
    tr = (P["split"] == "train").to_numpy()
    y, t = P["employed_6m"].to_numpy(), P["coached"].to_numpy()
    s, tt, x = meta_learners(X.to_numpy()[tr], t[tr], y[tr], X.to_numpy())
    cf = forest(X.to_numpy()[tr], t[tr], y[tr])
    f = cf.effect(X.to_numpy())
    cate = pd.DataFrame({"seeker_id": P["seeker_id"], "split": P["split"], "s_learner": s.round(5), "t_learner": tt.round(5),
                         "x_learner": x.round(5), "causal_forest": f.round(5)})
    cate.to_csv(U.REF / "cate_pilot.csv", index=False, lineterminator="\n")
    print("pilot CATE written", round(time.time() - t0), "s")

    # DR scores on the training split (randomised: e = 0.5), with cross-fitted outcome models
    Ptr = P[tr].reset_index(drop=True)
    Xtr = U.design(Ptr)
    mu0, mu1 = U.outcome_models_crossfit(Ptr["employed_6m"], Ptr["coached"], Xtr, model=U.pilot_classifier(3), seed=3)
    g = U.dr_scores(Ptr["employed_6m"], Ptr["coached"], 0.5, mu0, mu1)
    pd.DataFrame({"seeker_id": Ptr["seeker_id"], "dr_score": g.round(5)}).to_csv(U.REF / "dr_pilot.csv", index=False, lineterminator="\n")

    # Logged: forest fitted on the whole pilot; cross-fitted outcome models on the logs
    cf_all = forest(X.to_numpy(), t, y, seed=1)
    L = U.logged()
    XL = U.design(L)
    f_l = cf_all.effect(XL.to_numpy())
    XL60 = U.design(L, U.CONFOUNDERS)
    m0l, m1l = U.outcome_models_crossfit(L["employed_6m"], L["offered"], XL60, seed=4)
    pd.DataFrame({"decision_id": L["decision_id"], "cate_hat": f_l.round(5), "mu0": m0l.round(5), "mu1": m1l.round(5)}).to_csv(
        U.REF / "cate_logged.csv.gz", index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0})
    print("logged written", round(time.time() - t0), "s")

    # The reference rule: depth two, allowed attributes only, within the budget
    tree, cost = U.budget_tree(Ptr[ALLOWED], g)
    tree["cost"] = cost
    tree["features"] = ALLOWED
    (U.REF / "rule_tree.json").write_text(json.dumps(tree, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(U.tree_text(tree))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
