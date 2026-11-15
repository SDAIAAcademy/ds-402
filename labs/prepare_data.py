"""Write the synthetic data of Fursa Employment Agency into labs/data/ (fixed seed, same files on every run).

    uv run python prepare_data.py

Fursa Employment Agency is fictional, and so are its six regions, its twelve offices and every
job seeker. Nothing here is real data. The generator wrote every true effect, so it stores them
beside the data and each lab can set its estimate against the truth, which real data never allow.

Files
    data/pilot.csv              the randomised coaching pilot: 6,000 job seekers, coached by lottery
                                (one in two, balanced within each office), 14 covariates, the number
                                of coaching sessions attended and employment within six months
    data/logged.csv.gz          40,000 logged caseworker decisions of 2025 and 2026: 60 confounders,
                                the offer probability the allocation tool recorded, the offer, the outcome
    data/review/anwar_analysis.md   an analyst's subgroup claim with four seeded faults (Lab 3.4)
    data/requests.json          three questions from the director of employment services (Lab 1.1)
    data/truth.json             the true averages and the planted problems
    data/truth_pilot.csv        each pilot seeker's chance of employment without and with coaching
    data/truth_logged.csv.gz    the same for every logged decision
    capstone/<brief>/           the three capstone briefs: pilot.csv, logged.csv.gz, truth files, BRIEF.md

Planted problems (the lab that finds each one)
    risk is not benefit: the agency's risk score ranks the hopeless and the fine first    1.1, 2.4, 4.1
    groups almost never offered coaching (disability in Rumh and Nijad; over 36 months
    unemployed; over 55)                                                                    1.2, 3.3
    nonlinear confounding through administrative history: linear adjustment is biased     1.3
    a later extract without the motivation note: a hidden confounder                       1.4
    office RUM-2 overrode the tool: its recorded probabilities understate its offers       3.1
    a rule that reaches the never-offered rests on a few heavy weights                     3.3
    four faults in the analyst's subgroup claim                                            3.4
    seekers far from an office gain less, so an unconstrained rule leaves Nijad behind     4.2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 402
LF = chr(10)
LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
CAP = LABS / "capstone"

REGIONS = ["Sahil", "Rumh", "Jaww", "Qaraa", "Mazari", "Nijad"]
CODES = {"Sahil": "SAH", "Rumh": "RUM", "Jaww": "JAW", "Qaraa": "QAR", "Mazari": "MAZ", "Nijad": "NIJ"}
SHARE = [0.30, 0.20, 0.17, 0.14, 0.11, 0.08]
REGION_EFFECT = {"Sahil": 0.25, "Rumh": -0.05, "Jaww": 0.10, "Qaraa": 0.0, "Mazari": -0.10, "Nijad": -0.30}
N_ADM = 46
OVERRIDE_OFFICE = "RUM-2"


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def seekers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Covariates of n job seekers, the latent motivation and 46 administrative history fields."""
    region = rng.choice(REGIONS, size=n, p=SHARE)
    office = np.array([f"{CODES[r]}-{k}" for r, k in zip(region, rng.integers(1, 3, n))])
    age = np.clip(np.round(18 + rng.gamma(2.2, 6.5, n)), 18, 60).astype(int)
    female = (rng.random(n) < 0.46).astype(int)
    disability = (rng.random(n) < 0.07).astype(int)
    education = rng.choice(4, size=n, p=[0.18, 0.40, 0.17, 0.25])
    months = np.clip(np.round(rng.gamma(1.3, np.where(region == "Nijad", 16.0, 9.0))), 0, 60).astype(int)  # remote seekers stay unemployed longer
    prior_jobs = np.clip(rng.poisson(0.6 + 0.045 * (age - 18)), 0, 10)
    wage = np.where(prior_jobs == 0, 0.0, np.clip(np.round(rng.lognormal(1.6 + 0.12 * education, 0.4), 1), 3.0, 25.0))
    digital = np.clip(np.round(3.2 + 1.3 * education - 0.06 * (age - 30) + rng.normal(0, 1.8, n)), 0, 10).astype(int)
    remote = region == "Nijad"
    distance = np.where(remote, rng.lognormal(4.45, 0.45, n), rng.lognormal(2.7, 0.6, n))
    distance = np.clip(np.round(distance), 1, 250).astype(int)
    licence = (rng.random(n) < np.where(female == 1, 0.45, 0.62)).astype(int)
    dependants = np.clip(rng.poisson(1.5, n), 0, 8)
    m = rng.normal(0, 1, n)
    motivation = np.clip(np.round(3 + 1.1 * m), 1, 5).astype(int)
    df = pd.DataFrame({
        "region": region, "office": office, "age": age,
        "gender": np.where(female == 1, "female", "male"), "disability": disability,
        "education": education, "months_unemployed": months, "prior_wage_k": wage,
        "digital_skills": digital, "prior_jobs": prior_jobs, "distance_km": distance,
        "has_licence": licence, "dependants": dependants, "motivation": motivation,
    })
    adm = {}
    for j in range(1, N_ADM + 1):
        name = f"adm_{j:02d}"
        if j <= 5:
            v = rng.normal(0, 1, n) + (0.5 * m if j == 1 else 0.0)
            adm[name] = np.round(v, 2)
        elif j <= 15:
            adm[name] = (rng.random(n) < 0.12 + 0.03 * (j - 6)).astype(int)
        elif j <= 30:
            adm[name] = rng.poisson(0.8 + 0.2 * (j - 16), n)
        else:
            adm[name] = np.round(rng.normal(0, 1, n), 2)
    df = pd.concat([df, pd.DataFrame(adm)], axis=1)
    df["_m"] = m
    return df


def baseline_logit(df: pd.DataFrame) -> np.ndarray:
    """Log odds of employment within six months without coaching."""
    f = (df["gender"] == "female").to_numpy()
    g_adm = (0.35 * np.sin(1.5 * df["adm_02"]) + 0.30 * df["adm_06"] * df["adm_07"]
             + 0.25 * (df["adm_03"] > 0.8) - 0.04 * df["adm_16"] + 0.20 * np.tanh(df["adm_04"] * df["adm_05"]))
    return (-0.30 + 0.28 * df["education"] + 0.10 * df["digital_skills"] - 0.045 * df["months_unemployed"]
            - 0.035 * np.maximum(df["age"] - 40, 0) + 0.25 * np.sqrt(df["prior_jobs"]) + 0.25 * df["has_licence"]
            - 0.004 * df["distance_km"] - 0.55 * df["disability"] - 0.15 * f
            + df["region"].map(REGION_EFFECT) + 0.35 * df["_m"] + g_adm).to_numpy()


def effect(df: pd.DataFrame) -> np.ndarray:
    """Coaching's effect on the chance of employment, before clipping: it peaks near a year unemployed."""
    mo = df["months_unemployed"].to_numpy()
    bump = np.exp(-((mo - 12) / 8.0) ** 2)
    return (-0.02 + 0.22 * bump + 0.07 * (df["digital_skills"] <= 3) + 0.05 * df["disability"]
            + 0.04 * (df["age"] <= 29) - 0.04 * (df["education"] == 3) - 0.08 * (mo <= 2)
            - 0.08 * (df["distance_km"] > 60)).to_numpy()


def risk_score(df: pd.DataFrame) -> np.ndarray:
    """The agency's existing score: its estimate of the chance of NOT finding work in six months, 0 to 100."""
    z = (0.2 + 0.28 * df["education"] + 0.09 * df["digital_skills"] - 0.05 * df["months_unemployed"]
         - 0.03 * np.maximum(df["age"] - 40, 0) + 0.2 * np.sqrt(df["prior_jobs"]) - 0.5 * df["disability"])
    return np.round(100 * (1 - logistic(z))).astype(int)


def potential(df: pd.DataFrame):
    p0 = logistic(baseline_logit(df))
    p1 = np.clip(p0 + effect(df), 0.005, 0.995)
    return p0, p1


def offer_logit(df: pd.DataFrame) -> np.ndarray:
    """How the allocation tool and the caseworkers chose whom to offer coaching."""
    reg = df["region"].map({"Sahil": 0.15, "Rumh": -0.25, "Jaww": 0.0, "Qaraa": 0.05, "Mazari": -0.1, "Nijad": -0.9})
    never = (df["disability"] == 1) & df["region"].isin(["Rumh", "Nijad"])
    return (-1.25 + 0.035 * (df["risk_score"] - 50) + 0.45 * df["_m"]
            + 0.7 * np.sin(1.5 * df["adm_02"]) + 0.6 * df["adm_06"] * df["adm_07"] + 0.5 * (df["adm_03"] > 0.8)
            - 0.08 * df["adm_16"] + 0.4 * np.tanh(df["adm_04"] * df["adm_05"]) + reg
            - 3.2 * never - 2.6 * (df["months_unemployed"] > 36) - 2.0 * (df["age"] > 55)).to_numpy()


CORE = ["region", "office", "age", "gender", "disability", "education", "months_unemployed", "prior_wage_k",
        "digital_skills", "prior_jobs", "distance_km", "has_licence", "dependants", "motivation", "risk_score"]


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        df.to_csv(path, index=False, lineterminator=LF, compression={"method": "gzip", "mtime": 0})
    else:
        df.to_csv(path, index=False, lineterminator=LF)


def make_pilot(rng, n=6000, prefix="P"):
    df = seekers(n, rng)
    df["risk_score"] = risk_score(df)
    df.insert(0, "seeker_id", [f"{prefix}{i:05d}" for i in range(1, n + 1)])
    coached = np.zeros(n, dtype=int)
    for _, idx in df.groupby("office").groups.items():   # lottery: half of each office, exactly
        idx = np.array(sorted(idx))
        chosen = rng.permutation(idx)[: len(idx) // 2 + (rng.random() < 0.5) * (len(idx) % 2)]
        coached[chosen] = 1
    df["coached"] = coached
    p0, p1 = potential(df)
    y = (rng.random(n) < np.where(coached == 1, p1, p0)).astype(int)
    sessions = np.where(coached == 1, np.clip(rng.poisson(np.exp(0.9 + 0.35 * df["_m"] + 0.6 * (y == 1))), 0, 10), 0)
    df["coaching_sessions"] = sessions
    df["employed_6m"] = y
    split = np.where(rng.random(n) < 0.6, "train", "test")
    df["split"] = split
    truth = pd.DataFrame({"seeker_id": df["seeker_id"], "p0": np.round(p0, 5), "p1": np.round(p1, 5)})
    out = df[["seeker_id"] + CORE + ["coached", "coaching_sessions", "employed_6m", "split"]]
    return out, truth, df


def make_logged(rng, n=40000, prefix="L"):
    df = seekers(n, rng)
    df["risk_score"] = risk_score(df)
    df.insert(0, "decision_id", [f"{prefix}{i:05d}" for i in range(1, n + 1)])
    e = np.clip(logistic(offer_logit(df)), 0.01, 0.90)
    actual = e.copy()
    rum2 = (df["office"] == OVERRIDE_OFFICE).to_numpy()
    actual[rum2] = np.minimum(0.9, 2.6 * e[rum2])          # the override: the tool's number was not what happened
    offered = (rng.random(n) < actual).astype(int)
    p0, p1 = potential(df)
    y = (rng.random(n) < np.where(offered == 1, p1, p0)).astype(int)
    df["year"] = np.where(rng.random(n) < 0.5, 2025, 2026)
    df["p_offer"] = np.round(e, 4)
    df["offered"] = offered
    df["employed_6m"] = y
    adm = [f"adm_{j:02d}" for j in range(1, N_ADM + 1)]
    out = df[["decision_id", "year"] + CORE + adm + ["p_offer", "offered", "employed_6m"]]
    truth = pd.DataFrame({"decision_id": df["decision_id"], "p0": np.round(p0, 5), "p1": np.round(p1, 5),
                          "p_actual": np.round(actual, 4)})
    return out, truth, df


REQUESTS = {
    "from": "Wijdan Al-Hazzaa, director of employment services",
    "questions": [
        {"id": "Q1", "text": "Did coaching help the job seekers we offered it to in 2025 and 2026?"},
        {"id": "Q2", "text": "Next year we have one coaching place for every five job seekers. Who should get them?"},
        {"id": "Q3", "text": "Our risk score finds the people least likely to find work. Should places go to the highest scores?"},
    ],
}


REVIEW = """# Who benefits from coaching? Subgroup findings from the pilot

Anwar Al-Mohaimeed, analyst, programme analytics. Draft for the director.

## Method

I used all 6,000 pilot seekers. I fitted a gradient-boosted model of employment on the 14
covariates and the coaching flag and read its predicted effect for every seeker. I then split
the pilot into {f3_cells} cells (region by gender by age band by education) with at least 30
seekers each, estimated each cell's effect as the difference in employment rates between
coached and not coached seekers, and kept the cells with a p-value below 0.05.

## Findings

1. **Seekers who attended four or more coaching sessions** gain {f1_effect} points over the
   seekers who were not coached (n = {f1_n} coached seekers, p < 0.001). Coaching works when
   seekers engage with it.
2. **{f2_label}** gain {f2_effect} points (n = {f2_n}, p = {f2_p}). This is our strongest
   demographic finding.
3. {f3_k} of the {f3_cells} cells had p < 0.05. They are listed in the appendix as the targets
   for next year; in the other cells coaching showed no effect.
4. The model's predicted effects match the observed effects closely: across deciles of the
   predicted effect, the correlation between predicted and observed coached-minus-uncoached
   differences is above 0.9.

## Recommendation

Give coaching places first to seekers likely to attend four or more sessions, then to the
{f3_k} target cells, starting with {f2_short}.
"""

EDU_NAMES = ["no secondary", "secondary", "diploma", "bachelor"]


def _pval(a, b):
    from math import erf, sqrt
    p1, p0, n1, n0 = a.mean(), b.mean(), len(a), len(b)
    se = sqrt(max(p1 * (1 - p1), 1e-9) / n1 + max(p0 * (1 - p0), 1e-9) / n0)
    z = abs(p1 - p0) / se
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def review_cells(d: pd.DataFrame) -> pd.DataFrame:
    """The analyst's search: region x gender x age band x education cells with at least 30 seekers."""
    band = pd.cut(d["age"], [17, 24, 29, 34, 39, 44, 49, 60], labels=["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-60"])
    t, y = d["coached"].to_numpy(), d["employed_6m"].to_numpy()
    rows = []
    for key, idx in d.groupby([d["region"], d["gender"], band, d["education"]], observed=True).groups.items():
        s = np.zeros(len(d), bool)
        s[d.index.get_indexer(idx)] = True
        if s.sum() < 30 or (s & (t == 1)).sum() < 5 or (s & (t == 0)).sum() < 5:
            continue
        a, b = y[s & (t == 1)], y[s & (t == 0)]
        rows.append({"region": key[0], "gender": key[1], "age_band": str(key[2]), "education": int(key[3]),
                     "n": int(s.sum()), "effect": float(a.mean() - b.mean()), "p": float(_pval(a, b))})
    return pd.DataFrame(rows)


def write_review(pilot: pd.DataFrame, rng):
    d = pilot.reset_index(drop=True)
    t, y = d["coached"].to_numpy(), d["employed_6m"].to_numpy()
    eng = (d["coaching_sessions"] >= 4).to_numpy()
    f1 = y[eng & (t == 1)].mean() - y[t == 0].mean()
    cells = review_cells(d)
    top = cells.sort_values("effect", ascending=False).iloc[0]
    g = "women" if top["gender"] == "female" else "men"
    label = f"{g.capitalize()} aged {top['age_band']} in {top['region']} with {EDU_NAMES[top['education']]} education"
    short = f"{g} aged {top['age_band']} in {top['region']}"
    k = int((cells["p"] < 0.05).sum())
    text = REVIEW.format(f1_effect=f"{100 * f1:.1f}", f1_n=int((eng & (t == 1)).sum()), f2_label=label,
                         f2_short=short, f2_effect=f"{100 * top['effect']:.1f}", f2_n=int(top["n"]),
                         f2_p=f"{top['p']:.3f}", f3_k=k, f3_cells=len(cells))
    (DATA / "review").mkdir(parents=True, exist_ok=True)
    (DATA / "review" / "anwar_analysis.md").write_text(text, encoding="utf-8", newline=LF)
    return {"f1_effect": round(float(f1), 4), "f1_n": int((eng & (t == 1)).sum()),
            "f2": {"region": top["region"], "gender": top["gender"], "age_band": top["age_band"],
                   "education": int(top["education"]), "n": int(top["n"]), "effect": round(float(top["effect"]), 4),
                   "p": round(float(top["p"]), 4)},
            "cells": int(len(cells)), "cells_significant": k,
            "cells_bonferroni": int((cells["p"] < 0.05 / len(cells)).sum())}


# ---------------------------------------------------------------- capstone briefs

def brief_energy(rng, n_pilot=5000, n_log=30000):
    """Siraj Power Distribution: home energy-saving visits. Outcome: percentage of summer use saved."""
    def units(n):
        district = rng.choice(["North", "Centre", "Coast", "Valley", "Plateau"], size=n, p=[0.25, 0.3, 0.2, 0.15, 0.1])
        size = np.clip(rng.poisson(5, n), 1, 14)
        home_age = np.clip(np.round(rng.gamma(2, 8, n)), 0, 60).astype(int)
        kwh = np.round(rng.lognormal(7.2 + 0.05 * size, 0.35), 0)
        ac = np.clip(rng.poisson(2 + size / 4), 0, 10)
        elderly = (rng.random(n) < 0.22).astype(int)
        income = rng.choice(["low", "middle", "high"], size=n, p=[0.3, 0.5, 0.2])
        owner = (rng.random(n) < 0.6).astype(int)
        smart = (rng.random(n) < 0.4).astype(int)
        d = pd.DataFrame({"district": district, "household_size": size, "home_age_years": home_age,
                          "summer_kwh_month": kwh, "ac_units": ac, "elderly_member": elderly,
                          "income_band": income, "owner_occupied": owner, "smart_meter": smart})
        for j in range(1, 21):
            d[f"acct_{j:02d}"] = np.round(rng.normal(0, 1, n), 2) if j <= 10 else rng.poisson(1.5, n)
        return d
    def y0(d):
        return 2.0 + 0.02 * d["home_age_years"] + 0.5 * d["smart_meter"] + 0.8 * np.sin(d["acct_01"]) + 0.3 * d["acct_02"]
    def tau(d):
        return (1.0 + 3.0 * (d["home_age_years"] > 15) + 0.8 * (d["ac_units"] >= 4) - 1.0 * d["owner_occupied"].eq(0)
                + 1.5 * (np.log(d["summer_kwh_month"]) > 7.6)).to_numpy()
    def prop(d):
        return np.clip(logistic(-1.8 + 0.6 * (d["summer_kwh_month"] > 2000) + 0.5 * np.sin(d["acct_01"]) + 0.4 * d["acct_03"]
                                - 1.5 * (d["district"] == "Plateau")), 0.02, 0.9)
    return dict(name="siraj-energy-visits", units=units, y0=y0, tau=tau, prop=prop, outcome="kwh_saved_pct",
                treatment="visited", id="household_id", binary=False, noise=2.5, n_pilot=n_pilot, n_log=n_log,
                audit=["district", "income_band", "elderly_member"])


def brief_inspections(rng, n_pilot=3000, n_log=15000):
    """Al-Nakheel Municipality Food Safety Department: unannounced inspections. Outcome: compliant at follow-up."""
    def units(n):
        district = rng.choice(["Old Town", "Souq", "Harbour", "Garden", "New Town", "Industrial"], size=n)
        seats = np.clip(np.round(rng.lognormal(3.3, 0.7, n)), 4, 400).astype(int)
        years = np.clip(np.round(rng.gamma(2, 3, n)), 0, 40).astype(int)
        prior = np.clip(rng.poisson(1.0, n), 0, 9)
        complaints = np.clip(rng.poisson(0.7 + 0.3 * prior), 0, 12)
        chain = (rng.random(n) < 0.25).astype(int)
        cuisine = rng.choice(["grill", "bakery", "seafood", "cafe", "mixed"], size=n)
        size_band = np.where(seats < 20, "small", np.where(seats < 80, "medium", "large"))
        d = pd.DataFrame({"district": district, "seats": seats, "years_open": years, "prior_violations": prior,
                          "complaints_12m": complaints, "chain": chain, "cuisine": cuisine, "size_band": size_band})
        for j in range(1, 21):
            d[f"lic_{j:02d}"] = np.round(rng.normal(0, 1, n), 2) if j <= 10 else rng.poisson(1.2, n)
        return d
    def y0(d):
        return (0.6 - 0.35 * d["prior_violations"] - 0.2 * d["complaints_12m"] + 0.5 * d["chain"]
                + 0.4 * np.sin(1.5 * d["lic_01"]) + 0.2 * (d["cuisine"] == "seafood") * -1)
    def tau(d):
        return (0.02 + 0.12 * (d["prior_violations"].between(1, 3)) + 0.05 * (d["years_open"] <= 2)
                - 0.04 * d["chain"]).to_numpy()
    def prop(d):
        return np.clip(logistic(-1.6 + 0.5 * d["prior_violations"] + 0.3 * d["complaints_12m"] + 0.5 * np.sin(1.5 * d["lic_01"])
                                - 1.2 * (d["district"] == "Industrial")), 0.02, 0.9)
    return dict(name="nakheel-inspections", units=units, y0=y0, tau=tau, prop=prop, outcome="compliant_followup",
                treatment="inspected", id="restaurant_id", binary=True, n_pilot=n_pilot, n_log=n_log,
                audit=["district", "size_band", "chain"])


def build_brief(spec, rng):
    folder = CAP / spec["name"]
    out = {}
    for kind, n in [("pilot", spec["n_pilot"]), ("logged", spec["n_log"])]:
        d = spec["units"](n)
        d.insert(0, spec["id"], [f"{kind[0].upper()}{i:05d}" for i in range(1, n + 1)])
        if kind == "pilot":
            e = np.full(n, 0.5)
            t = (rng.random(n) < 0.5).astype(int)
            d["split"] = np.where(rng.random(n) < 0.6, "train", "test")
        else:
            e = spec["prop"](d)
            t = (rng.random(n) < e).astype(int)
            d["p_treat"] = np.round(e, 4)
        tau = spec["tau"](d)
        if spec["binary"]:
            p0 = logistic(spec["y0"](d).to_numpy())
            p1 = np.clip(p0 + tau, 0.005, 0.995)
            y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(int)
            truth = pd.DataFrame({spec["id"]: d[spec["id"]], "y0_mean": np.round(p0, 5), "y1_mean": np.round(p1, 5)})
        else:
            m0 = spec["y0"](d).to_numpy()
            m1 = m0 + tau
            y = np.round(np.where(t == 1, m1, m0) + rng.normal(0, spec["noise"], n), 2)
            truth = pd.DataFrame({spec["id"]: d[spec["id"]], "y0_mean": np.round(m0, 4), "y1_mean": np.round(m1, 4)})
        d[spec["treatment"]] = t
        d[spec["outcome"]] = y
        write_csv(d, folder / ("pilot.csv" if kind == "pilot" else "logged.csv.gz"))
        write_csv(truth, folder / "truth" / (f"truth_{kind}.csv.gz"))
        out[kind] = {"rows": int(n), "treated_share": round(float(t.mean()), 4),
                     "true_ate": round(float((truth["y1_mean"] - truth["y0_mean"]).mean()), 4)}
    return out


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(parents=True, exist_ok=True)
    pilot, truth_p, pfull = make_pilot(rng)
    logged, truth_l, lfull = make_logged(rng)
    write_csv(pilot, DATA / "pilot.csv")
    write_csv(truth_p, DATA / "truth_pilot.csv")
    write_csv(logged, DATA / "logged.csv.gz")
    write_csv(truth_l, DATA / "truth_logged.csv.gz")
    (DATA / "requests.json").write_text(json.dumps(REQUESTS, indent=2, ensure_ascii=False) + LF, encoding="utf-8", newline=LF)
    review = write_review(pilot, rng)

    tl = truth_l
    cate_l = tl["p1"] - tl["p0"]
    off = logged["offered"] == 1
    cate_p = truth_p["p1"] - truth_p["p0"]
    truth = {
        "about": "True values written by prepare_data.py. Real data never give you these.",
        "pilot": {"rows": len(pilot), "ate": round(float(cate_p.mean()), 4),
                  "ate_test": round(float(cate_p[pilot["split"] == "test"].mean()), 4),
                  "share_harmed": round(float((cate_p < 0).mean()), 4)},
        "logged": {"rows": len(logged), "ate": round(float(cate_l.mean()), 4), "att": round(float(cate_l[off].mean()), 4),
                   "offer_rate": round(float(off.mean()), 4),
                   "value_current_practice": round(float(np.where(off, tl["p1"], tl["p0"]).mean()), 4),
                   "value_treat_none": round(float(tl["p0"].mean()), 4),
                   "value_treat_all": round(float(tl["p1"].mean()), 4)},
        "planted": {
            "override_office": OVERRIDE_OFFICE,
            "never_offered": ["disability in Rumh or Nijad", "more than 36 months unemployed", "older than 55"],
            "hidden_confounder_in_2027_extract": "motivation",
            "far_from_office_km": 60,
            "effect_peaks_at_months_unemployed": 12,
        },
        "review": review,
    }
    briefs = {}
    for spec in [brief_energy(rng), brief_inspections(rng)]:
        briefs[spec["name"]] = build_brief(spec, rng)
    # The coaching brief: Fursa's 2027 cohort, a new draw from the same agency (fresh seed offset)
    rng2 = np.random.default_rng(SEED + 27)
    p2, tp2, _ = make_pilot(rng2, n=4000, prefix="C")
    l2, tl2, _ = make_logged(rng2, n=20000, prefix="K")
    folder = CAP / "fursa-coaching-2027"
    write_csv(p2, folder / "pilot.csv")
    write_csv(l2, folder / "logged.csv.gz")
    write_csv(tp2.rename(columns={"p0": "y0_mean", "p1": "y1_mean"}), folder / "truth" / "truth_pilot.csv.gz")
    write_csv(tl2.drop(columns=["p_actual"]).rename(columns={"p0": "y0_mean", "p1": "y1_mean"}), folder / "truth" / "truth_logged.csv.gz")
    briefs["fursa-coaching-2027"] = {"pilot": {"rows": 4000}, "logged": {"rows": 20000}}
    truth["capstone"] = briefs
    (DATA / "truth.json").write_text(json.dumps(truth, indent=2) + LF, encoding="utf-8", newline=LF)
    print(json.dumps(truth, indent=2))


if __name__ == "__main__":
    main()
