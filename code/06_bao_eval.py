#!/usr/bin/env python3
"""06_bao_eval.py — Literature-standard baseline (Bao et al. 2020) + network increment.

- Features: 28 raw Compustat items (year t, as published).
- Models: RUSBoost (literature standard for this task) and
  HistGradientBoosting(balanced) as a modern check.
- Labels: ongoing-fraud (primary) and broad (fraud|SEC investigation).
- Temporal split: train 2004-2017, test 2018-2023.
- Question: does adding leak-free network features improve over the
  literature-standard financial baseline?
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
NET = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]

# ---------- labels (same spell logic as 04) ----------
res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res = res.dropna(subset=["cik", "res_begin_date", "res_end_date"])
res["file_year"] = pd.to_datetime(res.file_date).dt.year
res["beg"] = pd.to_datetime(res.res_begin_date).dt.year
res["end"] = pd.to_datetime(res.res_end_date).dt.year


def spells(df):
    out = {}
    for r in df.itertuples():
        out.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))
    return out


fraud_sp = spells(res[res.res_fraud == 1])
broad_sp = spells(res[(res.res_fraud == 1) | (res.res_sec_investigation == 1)])


def lab(cik, year, sp):
    for b, e, fy in sp.get(cik, ()):
        if b <= year <= e and fy > year:
            return 1
    return 0


# ---------- features ----------
f = pd.read_parquet(f"{DATA}/funda_bao.parquet")
f["cik"] = pd.to_numeric(f.cik, errors="coerce")
f = f.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"])
f = f.drop_duplicates(["gvkey", "fyear"], keep="last")
f[RAW] = f[RAW].astype("float64")
# Bao: missing raw items -> 0
f[RAW] = f[RAW].fillna(0.0)
# scale by total assets to control size (common variant; keeps comparability)
for c in RAW:
    if c not in ("at", "prcc_f", "csho"):
        f[c] = f[c] / f["at"]
f["log_at"] = np.log1p(f["at"])
FEAT = [c for c in RAW if c != "at"] + ["log_at"]

f["year"] = f.fyear.astype(int)

net = pd.read_parquet(f"{DATA}/network_panel.parquet")
net["cik"] = pd.to_numeric(net.company_fkey, errors="coerce")
net = net.dropna(subset=["cik"])

m = f[["cik", "year"] + FEAT].merge(
    net[["cik", "year"] + NET], on=["cik", "year"], how="inner")
m = m.replace([np.inf, -np.inf], 0.0)

m["label"] = [lab(c, y, fraud_sp) for c, y in zip(m.cik, m.year)]
m["label_broad"] = [lab(c, y, broad_sp) for c, y in zip(m.cik, m.year)]

print(f"panel: {len(m):,} firm-years, {m.cik.nunique():,} firms")
print(f"ongoing fraud: {m.label.sum():,} ({m.label.mean():.3%}) | "
      f"broad: {m.label_broad.sum():,} ({m.label_broad.mean():.3%})")

train = m[(m.year >= 2004) & (m.year <= 2017)]
test = m[(m.year >= 2018) & (m.year <= 2023)]


def rusboost():
    return RUSBoostClassifier(
        estimator=DecisionTreeClassifier(min_samples_leaf=5),
        n_estimators=300, learning_rate=0.1, random_state=42)


for target in ["label", "label_broad"]:
    print(f"\n===== {target} | train pos {train[target].sum()} "
          f"| test pos {test[target].sum()} | test base {test[target].mean():.4f} =====")
    for name, cols in [("financial-28", FEAT), ("financial-28 + network", FEAT + NET)]:
        for mdl_name, mdl in [("rusboost", rusboost()),
                              ("hgb", HistGradientBoostingClassifier(
                                  random_state=42, class_weight="balanced"))]:
            mdl.fit(train[cols], train[target])
            s = mdl.predict_proba(test[cols])[:, 1]
            print(f"  {name:24s} {mdl_name:9s} "
                  f"PR-AUC={average_precision_score(test[target], s):.4f}  "
                  f"ROC-AUC={roc_auc_score(test[target], s):.4f}")
