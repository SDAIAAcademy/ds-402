"""Helpers for the DS-402 labs: the data of Fursa Employment Agency, the estimators the course
provides, checks and safety nets.

    import fursa as U

NumPy, pandas, scikit-learn and statsmodels. Nothing calls or downloads a model.

Environment variables
    DS402_QUICK=1      fewer rows, folds and trees, for a fast run on a laptop
    DS402_STRICT=1     solution runs: a failed check stops the run, a safety net is an error
    DS402_WRITE_REF=1  the trainer's run that writes data/reference/<lab>.json (the safety nets read it)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):  # a Windows console or a redirected run may not default to UTF-8
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
REF = DATA / "reference"
ART = LABS / "artifacts"

QUICK = os.environ.get("DS402_QUICK", "0") == "1"
STRICT = os.environ.get("DS402_STRICT", "0") == "1"
WRITE_REF = os.environ.get("DS402_WRITE_REF", "0") == "1"
MODE = "quick" if QUICK else "full"
SEED = 402
BUDGET = 0.20                       # one coaching place per five job seekers
REGIONS = ["Sahil", "Rumh", "Jaww", "Qaraa", "Mazari", "Nijad"]

NAMED = ["region", "age", "gender", "disability", "education", "months_unemployed", "prior_wage_k",
         "digital_skills", "prior_jobs", "distance_km", "has_licence", "dependants", "motivation"]
ADM = [f"adm_{j:02d}" for j in range(1, 47)]
CONFOUNDERS = NAMED + ["risk_score"] + ADM          # the 60 columns of the logged data
PILOT_X = NAMED + ["risk_score"]                     # the 14 covariates of the pilot


def pick(full, quick):
    """The full value, or the small one in quick mode."""
    return quick if QUICK else full


def rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


# ---------------------------------------------------------------- checks and safety nets

def check(ok, what: str) -> bool:
    """Print a tick or a cross. In a solution run (DS402_STRICT=1) a cross stops the run."""
    ok = bool(ok)
    print(("✅ " if ok else "❌ ") + what)
    if STRICT and not ok:
        raise AssertionError(what)
    return ok


def _jsonable(v):
    if isinstance(v, pd.DataFrame):
        return {"__frame__": json.loads(v.to_json(orient="split"))}
    if isinstance(v, pd.Series):
        return {"__series__": {"index": list(v.index), "data": _jsonable(v.to_numpy()), "name": v.name}}
    if isinstance(v, np.ndarray):
        return {"__array__": v.tolist()}
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _unjson(v):
    if isinstance(v, dict):
        if "__array__" in v:
            return np.array(v["__array__"])
        if "__frame__" in v:
            f = v["__frame__"]
            return pd.DataFrame(f["data"], index=f["index"], columns=f["columns"])
        if "__series__" in v:
            f = v["__series__"]
            return pd.Series(_unjson(f["data"]), index=f["index"], name=f.get("name"))
        return {k: _unjson(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unjson(x) for x in v]
    return v


def keep(lab: str, name: str, value):
    """In the trainer's reference run, store a result for the safety net of that exercise."""
    if not WRITE_REF:
        return
    REF.mkdir(parents=True, exist_ok=True)
    path = REF / f"{lab}.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    store[name] = _jsonable(value)
    path.write_text(json.dumps(store, indent=1) + "\n", encoding="utf-8", newline="\n")


def reference(lab: str, name: str):
    """The stored result of an exercise, for a participant who has not finished it yet."""
    if STRICT:
        raise AssertionError(f"a solution run took the safety net of {lab}/{name}")
    print(f"↪ using the reference result for {name}: finish the exercise to use your own")
    store = json.loads((REF / f"{lab}.json").read_text(encoding="utf-8"))
    return _unjson(store[name])


def artifacts(lab: str) -> Path:
    """A folder for this lab's outputs (labs/artifacts/<lab>/), never committed."""
    p = ART / lab
    p.mkdir(parents=True, exist_ok=True)
    return p


def pp(x: float) -> str:
    """A difference in probabilities as percentage points: 0.0912 -> '9.1 points'."""
    return f"{100 * x:.1f} points"


# ---------------------------------------------------------------- data

def truth() -> dict:
    """The true values the generator wrote (data/truth.json). Real data never give you these."""
    return json.loads((DATA / "truth.json").read_text(encoding="utf-8"))


def pilot() -> pd.DataFrame:
    """The randomised coaching pilot: 6,000 seekers, coached by lottery within each office."""
    return pd.read_csv(DATA / "pilot.csv")


def logged(drop_override: bool = False) -> pd.DataFrame:
    """40,000 logged decisions with the offer probability the allocation tool recorded.
    drop_override=True leaves out office RUM-2, whose recorded probabilities Lab 3.1 shows are wrong."""
    d = pd.read_csv(DATA / "logged.csv.gz")
    if drop_override:
        d = d[d["office"] != "RUM-2"].reset_index(drop=True)
    return d


def logged_sample(drop_override: bool = False, n: int | None = None) -> pd.DataFrame:
    """The logs, or in quick mode a seeded sample of 10,000 decisions."""
    d = logged(drop_override)
    n = n or pick(len(d), 10000)
    if n < len(d):
        d = d.sample(n=n, random_state=SEED).reset_index(drop=True)
    return d


def truth_for(ids, which: str = "logged") -> pd.DataFrame:
    """The true chances p0 and p1 for the given ids, in their order. Real data never give you these."""
    tr = truth_logged() if which == "logged" else truth_pilot()
    key = tr.columns[0]
    return tr.set_index(key).loc[list(ids), ["p0", "p1"]].reset_index()


def truth_pilot() -> pd.DataFrame:
    """Each pilot seeker's chance of employment without (p0) and with (p1) coaching."""
    return pd.read_csv(DATA / "truth_pilot.csv")


def truth_logged() -> pd.DataFrame:
    return pd.read_csv(DATA / "truth_logged.csv.gz")


def requests() -> dict:
    return json.loads((DATA / "requests.json").read_text(encoding="utf-8"))


def design(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """A numeric matrix: region as indicators (Sahil the base), gender as female 0/1."""
    cols = list(cols if cols is not None else PILOT_X)
    X = df[cols].copy()
    if "gender" in X:
        X["female"] = (X.pop("gender") == "female").astype(int)
    if "region" in X:
        reg = X.pop("region")
        for r in REGIONS[1:]:
            X[f"region_{r}"] = (reg == r).astype(int)
    return X.astype(float)


def supplied(name: str) -> pd.DataFrame:
    """A supplied reference state from data/reference/ (predictions every lab can start from)."""
    p = REF / name
    return pd.read_csv(p)


# ---------------------------------------------------------------- learners

def gbm_classifier(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=pick(200, 80), learning_rate=0.08, max_leaf_nodes=15,
                                          min_samples_leaf=40, random_state=SEED + seed)


def gbm_regressor(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=pick(200, 80), learning_rate=0.08, max_leaf_nodes=15,
                                         min_samples_leaf=40, random_state=SEED + seed)


def pilot_classifier(seed: int = 0):
    """The gradient boosting of the pilot labs: 3,600 training seekers and a binary outcome need strong regularisation."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_leaf_nodes=15,
                                          min_samples_leaf=200, random_state=SEED + seed)


def pilot_regressor(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_leaf_nodes=15,
                                         min_samples_leaf=200, random_state=SEED + seed)


def folds(n: int, k: int = 5, seed: int = 0) -> np.ndarray:
    """A fold number for each of n rows, balanced and seeded."""
    f = np.arange(n) % k
    return rng(seed).permutation(f)


def crossfit_aipw(y, t, X, model_y, model_t, k: int = 5, clip: float = 0.01, seed: int = 0) -> dict:
    """The course's reference: cross-fitted augmented IPW (the IRM score of double machine learning)."""
    from sklearn.base import clone
    y, t, X = np.asarray(y, float), np.asarray(t, int), np.asarray(X, float)
    n = len(y)
    fold = folds(n, k, seed)
    e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for f in range(k):
        tr, te = fold != f, fold == f
        mt = clone(model_t).fit(X[tr], t[tr])
        e[te] = mt.predict_proba(X[te])[:, 1]
        m1 = clone(model_y).fit(X[tr & (t == 1)], y[tr & (t == 1)])
        m0 = clone(model_y).fit(X[tr & (t == 0)], y[tr & (t == 0)])
        mu1[te] = _predict(m1, X[te])
        mu0[te] = _predict(m0, X[te])
    e = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return {"est": float(psi.mean()), "se": float(psi.std(ddof=1) / np.sqrt(n)), "psi": psi, "e": e, "mu0": mu0, "mu1": mu1}


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def ols_adjust(y, t, X) -> tuple[float, float]:
    """Regression adjustment: the coefficient of t in a linear regression on t and X (robust standard error)."""
    import statsmodels.api as sm
    Z = sm.add_constant(np.column_stack([np.asarray(t, float), np.asarray(X, float)]), has_constant="add")
    fit = sm.OLS(np.asarray(y, float), Z).fit(cov_type="HC1")
    return float(fit.params[1]), float(fit.bse[1])


def dr_scores(y, t, e, mu0, mu1) -> np.ndarray:
    """Doubly robust effect scores: unbiased for each seeker's effect on average."""
    y, t, e = np.asarray(y, float), np.asarray(t, float), np.asarray(e, float)
    return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)


def outcome_models_crossfit(y, t, X, model=None, k: int = 5, seed: int = 0):
    """Cross-fitted predictions of the outcome without (mu0) and with (mu1) treatment."""
    from sklearn.base import clone
    model = model or gbm_classifier(seed)
    y, t, X = np.asarray(y), np.asarray(t), np.asarray(X, float)
    fold = folds(len(y), k, seed)
    mu0, mu1 = np.zeros(len(y)), np.zeros(len(y))
    for f in range(k):
        tr, te = fold != f, fold == f
        m1 = clone(model).fit(X[tr & (t == 1)], y[tr & (t == 1)])
        m0 = clone(model).fit(X[tr & (t == 0)], y[tr & (t == 0)])
        mu1[te], mu0[te] = _predict(m1, X[te]), _predict(m0, X[te])
    return mu0, mu1


# ---------------------------------------------------------------- validating effect estimates

def calibration_table(cate_hat, y, t, e=0.5, groups: int = 5, mu0=None, mu1=None) -> pd.DataFrame:
    """Predicted against observed effect by quantile group of the prediction (held-out randomised data).
    The observed effect of a group is the difference in means within it."""
    cate_hat, y, t = np.asarray(cate_hat), np.asarray(y, float), np.asarray(t, int)
    g = pd.qcut(pd.Series(cate_hat).rank(method="first"), groups, labels=False).to_numpy()
    rows = []
    for k in range(groups):
        s = g == k
        a, b = y[s & (t == 1)], y[s & (t == 0)]
        obs = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        rows.append({"group": k + 1, "n": int(s.sum()), "predicted": float(cate_hat[s].mean()),
                     "observed": float(obs), "se": float(se)})
    return pd.DataFrame(rows)


def blp_slope(cate_hat, y, t, e=0.5, mu0=None, mu1=None) -> tuple[float, float]:
    """Best linear predictor check: regress a per-seeker effect score on the centred prediction.
    With mu0 and mu1 (outcome predictions fitted without these seekers' own rows) the score is
    doubly robust and much less noisy; without them it is the plain inverse-probability score.
    A slope near 1 says the predictions are calibrated in scale; near 0, they carry no signal."""
    import statsmodels.api as sm
    cate_hat, y, t = np.asarray(cate_hat, float), np.asarray(y, float), np.asarray(t, float)
    if mu0 is None:
        score = y * (t / e - (1 - t) / (1 - e))
    else:
        score = dr_scores(y, t, e, mu0, mu1)
    Z = sm.add_constant(cate_hat - cate_hat.mean())
    fit = sm.OLS(score, Z).fit(cov_type="HC1")
    return float(fit.params[1]), float(fit.bse[1])


def qini_curve(score, y, t, points: int = 50) -> pd.DataFrame:
    """Qini curve on randomised data: rank by score, and at each share treated, the extra employed
    in the top share (treated outcomes minus control outcomes scaled to the treated count), per seeker."""
    score, y, t = np.asarray(score, float), np.asarray(y, float), np.asarray(t, int)
    n = len(y)
    order = np.argsort(-score, kind="stable")
    yt, tt = y[order], t[order]
    c_t, c_c = np.cumsum(tt), np.cumsum(1 - tt)
    s_t, s_c = np.cumsum(yt * tt), np.cumsum(yt * (1 - tt))
    idx = np.unique(np.linspace(0, n - 1, points + 1).astype(int))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(c_c[idx] > 0, s_t[idx] - s_c[idx] * c_t[idx] / np.maximum(c_c[idx], 1), 0.0)
    share = (idx + 1) / n
    out = pd.DataFrame({"share": np.r_[0.0, share], "qini": np.r_[0.0, gain / n]})
    return out


def uplift_at(score, y, t, share: float) -> float:
    """The effect among the top share by score: treated minus control means (randomised data)."""
    score, y, t = np.asarray(score, float), np.asarray(y, float), np.asarray(t, int)
    k = int(round(share * len(y)))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top][t[top] == 1].mean() - y[top][t[top] == 0].mean())


def qini_area(curve: pd.DataFrame) -> float:
    """Area between the Qini curve and the straight line of random targeting."""
    s, q = curve["share"].to_numpy(), curve["qini"].to_numpy()
    rand = q[-1] * s
    return float(np.trapezoid(q - rand, s))


def diff_p(a, b) -> tuple[float, float]:
    """Difference in means of two 0/1 arrays and its two-sided p-value (normal approximation)."""
    from math import erf, sqrt
    a, b = np.asarray(a, float), np.asarray(b, float)
    p1, p0 = a.mean(), b.mean()
    se = sqrt(max(p1 * (1 - p1), 1e-9) / len(a) + max(p0 * (1 - p0), 1e-9) / len(b))
    z = abs(p1 - p0) / se
    return float(p1 - p0), float(2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))))


AGE_BANDS = ([17, 24, 29, 34, 39, 44, 49, 60], ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-60"])


def review_cells(d: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """The analyst's search of Lab 3.4: region x gender x age band x education cells with at least min_n
    seekers, each with its coached-minus-uncoached difference and p-value."""
    d = d.reset_index(drop=True)
    band = pd.cut(d["age"], AGE_BANDS[0], labels=AGE_BANDS[1])
    t, y = d["coached"].to_numpy(), d["employed_6m"].to_numpy()
    rows = []
    for key, idx in d.groupby([d["region"], d["gender"], band, d["education"]], observed=True).groups.items():
        s = np.zeros(len(d), bool)
        s[np.asarray(idx)] = True
        if s.sum() < min_n or (s & (t == 1)).sum() < 5 or (s & (t == 0)).sum() < 5:
            continue
        eff, pv = diff_p(y[s & (t == 1)], y[s & (t == 0)])
        rows.append({"region": key[0], "gender": key[1], "age_band": str(key[2]), "education": int(key[3]),
                     "n": int(s.sum()), "effect": eff, "p": pv})
    return pd.DataFrame(rows)


def in_cell(d: pd.DataFrame, cell) -> np.ndarray:
    """Rows of d in a cell of review_cells (a dict or a row with region, gender, age_band, education)."""
    band = pd.cut(d["age"], AGE_BANDS[0], labels=AGE_BANDS[1]).astype(str)
    return ((d["region"] == cell["region"]) & (d["gender"] == cell["gender"]) & (band == cell["age_band"])
            & (d["education"] == cell["education"])).to_numpy()


# ---------------------------------------------------------------- off-policy evaluation

def policy_weights(a, p1, pi) -> np.ndarray:
    """Importance weights of a rule: 1{logged action = rule's action} / P(logged action)."""
    a, p1, pi = np.asarray(a, int), np.asarray(p1, float), np.asarray(pi, int)
    p_a = np.where(a == 1, p1, 1 - p1)
    return (a == pi) / p_a


def ipw_value(y, a, p1, pi) -> tuple[float, float]:
    w = policy_weights(a, p1, pi)
    v = w * np.asarray(y, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def snipw_value(y, a, p1, pi) -> float:
    w = policy_weights(a, p1, pi)
    return float(np.sum(w * np.asarray(y, float)) / np.sum(w))


def dr_value(y, a, p1, pi, mu0, mu1) -> tuple[float, float]:
    """Augmented (doubly robust) value of a rule."""
    y, a, pi = np.asarray(y, float), np.asarray(a, int), np.asarray(pi, int)
    w = policy_weights(a, p1, pi)
    mu_pi = np.where(pi == 1, mu1, mu0)
    mu_a = np.where(a == 1, mu1, mu0)
    psi = mu_pi + w * (y - mu_a)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


def ess(w) -> float:
    """Kish's effective sample size of a set of weights."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def true_value(pi, which: str = "logged") -> float:
    """The true value of a rule: the mean chance of employment if the rule had decided. Real data never give you this."""
    tr = truth_logged() if which == "logged" else truth_pilot()
    pi = np.asarray(pi, int)
    return float(np.where(pi == 1, tr["p1"], tr["p0"]).mean())


def top_share(score, share: float = BUDGET) -> np.ndarray:
    """A rule that treats the top share by a score."""
    score = np.asarray(score, float)
    k = int(round(share * len(score)))
    pi = np.zeros(len(score), int)
    pi[np.argsort(-score, kind="stable")[:k]] = 1
    return pi


# ---------------------------------------------------------------- policy trees

def _thresholds(x: np.ndarray, n: int) -> np.ndarray:
    q = np.unique(np.quantile(x, np.linspace(0, 1, n + 2)[1:-1]))
    return q


def _best_leaf_pair(xk, r, cand):
    """For one feature: the threshold that best splits rows into two leaves, each treating or not."""
    best = (-np.inf, None, 0, 0)
    for c in cand:
        left = xk <= c
        sl, sr = r[left].sum(), r[~left].sum()
        v = max(sl, 0) + max(sr, 0)
        if v > best[0]:
            best = (v, float(c), int(sl > 0), int(sr > 0))
    return best


def policy_tree(X: pd.DataFrame, gain, cost: float = 0.0, n_thresholds: int = 12) -> dict:
    """An exhaustive depth-two policy tree (like policytree for R) that maximises the sum of
    (gain - cost) over the seekers it treats. gain: a doubly robust effect score per seeker."""
    cols = list(X.columns)
    M = X.to_numpy(float)
    r = np.asarray(gain, float) - cost
    cands = [_thresholds(M[:, j], n_thresholds) for j in range(M.shape[1])]
    best = {"value": -np.inf}
    for j in range(M.shape[1]):
        for c in cands[j]:
            left = M[:, j] <= c
            if left.sum() < 30 or (~left).sum() < 30:
                continue
            sub = []
            for mask in (left, ~left):
                bb = (max(r[mask].sum(), 0.0), None, None, int(r[mask].sum() > 0), int(r[mask].sum() > 0))
                for k in range(M.shape[1]):
                    v, ck, al, ar = _best_leaf_pair(M[mask, k], r[mask], cands[k])
                    if v > bb[0] + 1e-12:
                        bb = (v, k, ck, al, ar)
                sub.append(bb)
            value = sub[0][0] + sub[1][0]
            if value > best["value"]:
                best = {"value": value, "feature": cols[j], "threshold": float(c),
                        "left": _node(cols, sub[0]), "right": _node(cols, sub[1])}
    return best


def _node(cols, bb):
    v, k, c, al, ar = bb
    if k is None:
        return {"leaf": al}
    return {"feature": cols[k], "threshold": float(c), "left": {"leaf": al}, "right": {"leaf": ar}}


def predict_tree(tree: dict, X: pd.DataFrame) -> np.ndarray:
    """The rule's decision (1 = offer coaching) for each row of X."""
    def go(node, rows):
        if "leaf" in node:
            return np.full(rows.sum(), node["leaf"], int)
        out = np.zeros(rows.sum(), int)
        x = X.loc[rows, node["feature"]].to_numpy(float)
        sub = np.zeros(len(X), bool)
        idx = np.flatnonzero(rows)
        l = x <= node["threshold"]
        sub_l, sub_r = np.zeros(len(X), bool), np.zeros(len(X), bool)
        sub_l[idx[l]], sub_r[idx[~l]] = True, True
        out[l] = go(node["left"], sub_l)
        out[~l] = go(node["right"], sub_r)
        return out
    return go(tree, np.ones(len(X), bool))


def tree_text(tree: dict, indent: str = "") -> str:
    if "leaf" in tree:
        return indent + ("offer coaching" if tree["leaf"] else "no offer")
    f, c = tree["feature"], tree["threshold"]
    return (f"{indent}if {f} <= {c:g}:\n{tree_text(tree['left'], indent + '    ')}\n"
            f"{indent}else:\n{tree_text(tree['right'], indent + '    ')}")


def budget_tree(X: pd.DataFrame, gain, budget: float = BUDGET, n_thresholds: int = 12, iters: int = 14) -> tuple[dict, float]:
    """The depth-two tree whose treated share is at most the budget: raise a cost per place until it fits."""
    lo, hi = 0.0, float(np.quantile(gain, 0.99))
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        t = policy_tree(X, gain, cost=mid, n_thresholds=n_thresholds)
        share = predict_tree(t, X).mean()
        if share <= budget:
            best, hi = (t, mid), mid
        else:
            lo = mid
    if best is None:
        best = (policy_tree(X, gain, cost=hi, n_thresholds=n_thresholds), hi)
    return best


# ---------------------------------------------------------------- trials

def n_per_arm(p_control: float, p_treat: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Seekers per arm for a two-sided test of two proportions."""
    from scipy.stats import norm
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p_control + p_treat) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p_control * (1 - p_control) + p_treat * (1 - p_treat))) ** 2
    return int(np.ceil(num / (p_treat - p_control) ** 2))
