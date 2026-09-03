#!/usr/bin/env python3
"""03_baseline_eval.py — Honest baseline: does the network add anything?

- Link: Audit Analytics company_fkey == CIK; Compustat via funda.cik.
- Target: firm files a FRAUD restatement (res_fraud=1) in year t+1.
  (rare event; base rate reported, no SMOTE, class_weight instead)
- Features:
    financial (real, from funda year t): 8 Beneish-style ratios
    network  (leak-free, from panel year t)
- Eval: temporal split (train 2004-2017, test 2018-2023), PR-AUC + ROC-AUC.
  Compare financial-only vs financial+network. Report honestly.
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
rng = np.random.RandomState(42)

# ---------- labels ----------
res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
fraud_years = (res[res.res_fraud == 1].dropna(subset=["cik"])
               .groupby("cik").file_year.apply(set))

# ---------- financials: Beneish-style ratios (REAL data, year t vs t-1) ----------
f = pd.read_parquet(f"{DATA}/funda.parquet")
f["cik"] = pd.to_numeric(f.cik, errors="coerce")
f = f.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"])
NUM = ["rect", "sale", "cogs", "at", "act", "ppent", "dp", "xsga", "dltt", "lct", "ni", "oancf"]
f[NUM] = f[NUM].astype("float64")
f = f.drop_duplicates(["gvkey", "fyear"], keep="last")
g = f.groupby("gvkey")
prev = g[NUM].shift(1)


def safe(a, b):
    return np.where((b != 0) & np.isfinite(a) & np.isfinite(b), a / b, np.nan)


X = pd.DataFrame(index=f.index)
X["dsri"] = safe(safe(f.rect, f.sale), safe(prev.rect, prev.sale))
X["gmi"] = safe(safe(prev.sale - prev.cogs, prev.sale), safe(f.sale - f.cogs, f.sale))
X["aqi"] = safe(1 - safe(f.act + f.ppent, f["at"]), 1 - safe(prev.act + prev.ppent, prev["at"]))
X["sgi"] = safe(f.sale, prev.sale)
X["depi"] = safe(safe(prev.dp, prev.dp + prev.ppent), safe(f.dp, f.dp + f.ppent))
X["sgai"] = safe(safe(f.xsga, f.sale), safe(prev.xsga, prev.sale))
X["lvgi"] = safe(safe(f.dltt + f.lct, f["at"]), safe(prev.dltt + prev.lct, prev["at"]))
X["tata"] = safe(f.ni - f.oancf, f["at"])
X["cik"] = f.cik.values
X["year"] = f.fyear.values
X = X.dropna(subset=["year"])
X["year"] = X.year.astype(int)

# ---------- network features (leak-free) ----------
net = pd.read_parquet(f"{DATA}/network_panel.parquet")
net["cik"] = pd.to_numeric(net.company_fkey, errors="coerce")
net = net.dropna(subset=["cik"])

m = X.merge(net[["cik", "year", "degree", "pagerank", "fraud_neighbor_cnt",
                 "fraud_neighbor_ratio"]], on=["cik", "year"], how="inner")

# target: fraud restatement FILED in t+1
m["label"] = m.apply(lambda r: 1 if (r.cik in fraud_years.index
                     and (r.year + 1) in fraud_years.loc[r.cik]) else 0, axis=1)

FIN = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]
NET = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]
m[FIN] = m[FIN].clip(-10, 10)
m = m.dropna(subset=FIN)

print(f"analysis panel: {len(m):,} firm-years ({m.year.min()}-{m.year.max()}), "
      f"{m.cik.nunique():,} firms")
print(f"positives (fraud restatement in t+1): {m.label.sum():,} "
      f"({m.label.mean():.3%})  <- honest base rate")

train = m[m.year <= 2017]
test = m[(m.year >= 2018) & (m.year <= 2023)]
print(f"temporal split: train {len(train):,} ({train.label.sum()} pos) | "
      f"test {len(test):,} ({test.label.sum()} pos)")

results = {}
for name, cols in [("financial only", FIN), ("financial + network", FIN + NET)]:
    sc = StandardScaler().fit(train[cols])
    Xtr, Xte = sc.transform(train[cols]), sc.transform(test[cols])
    for mdl_name, mdl in [
        ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("gbm", HistGradientBoostingClassifier(random_state=42)),
    ]:
        mdl.fit(Xtr, train.label)
        pscore = mdl.predict_proba(Xte)[:, 1]
        pr = average_precision_score(test.label, pscore)
        roc = roc_auc_score(test.label, pscore)
        results[(name, mdl_name)] = (pr, roc)
        print(f"{name:22s} {mdl_name:6s}  PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}")

base = test.label.mean()
print(f"\nrandom-classifier PR-AUC baseline = base rate = {base:.4f}")
print("Interpretation rule: network features 'add value' only if "
      "financial+network PR-AUC exceeds financial-only consistently.")
