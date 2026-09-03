#!/usr/bin/env python3
"""07_bootstrap_test.py — Is the network increment statistically distinguishable?

Paired bootstrap (2,000 resamples of the test set) of ΔROC-AUC and ΔPR-AUC
between financial-28 and financial-28+network, RUSBoost, primary fraud label.
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
rng = np.random.RandomState(42)

RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
NET = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]

res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res = res.dropna(subset=["cik", "res_begin_date", "res_end_date"])
res["file_year"] = pd.to_datetime(res.file_date).dt.year
res["beg"] = pd.to_datetime(res.res_begin_date).dt.year
res["end"] = pd.to_datetime(res.res_end_date).dt.year
sp = {}
for r in res[res.res_fraud == 1].itertuples():
    sp.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))

f = pd.read_parquet(f"{DATA}/funda_bao.parquet")
f["cik"] = pd.to_numeric(f.cik, errors="coerce")
f = f.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"]).drop_duplicates(["gvkey", "fyear"], keep="last")
f[RAW] = f[RAW].astype("float64").fillna(0.0)
for c in RAW:
    if c not in ("at", "prcc_f", "csho"):
        f[c] = f[c] / f["at"]
f["log_at"] = np.log1p(f["at"])
FEAT = [c for c in RAW if c != "at"] + ["log_at"]
f["year"] = f.fyear.astype(int)

net = pd.read_parquet(f"{DATA}/network_panel.parquet")
net["cik"] = pd.to_numeric(net.company_fkey, errors="coerce")
net = net.dropna(subset=["cik"])
m = f[["cik", "year"] + FEAT].merge(net[["cik", "year"] + NET], on=["cik", "year"], how="inner")
m = m.replace([np.inf, -np.inf], 0.0)
m["label"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp.get(c, ())) else 0
              for c, y in zip(m.cik, m.year)]

train = m[(m.year >= 2004) & (m.year <= 2017)]
test = m[(m.year >= 2018) & (m.year <= 2023)].reset_index(drop=True)
print(f"train pos {train.label.sum()}, test pos {test.label.sum()}")


def fit_score(cols):
    mdl = RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                             n_estimators=300, learning_rate=0.1, random_state=42)
    mdl.fit(train[cols], train.label)
    return mdl.predict_proba(test[cols])[:, 1]


s_fin = fit_score(FEAT)
s_net = fit_score(FEAT + NET)
y = test.label.values
print(f"point estimates: ROC fin={roc_auc_score(y, s_fin):.4f} "
      f"net={roc_auc_score(y, s_net):.4f} | "
      f"PR fin={average_precision_score(y, s_fin):.4f} "
      f"net={average_precision_score(y, s_net):.4f}")

d_roc, d_pr = [], []
n = len(y)
for _ in range(2000):
    idx = rng.randint(0, n, n)
    if y[idx].sum() < 5:
        continue
    d_roc.append(roc_auc_score(y[idx], s_net[idx]) - roc_auc_score(y[idx], s_fin[idx]))
    d_pr.append(average_precision_score(y[idx], s_net[idx])
                - average_precision_score(y[idx], s_fin[idx]))
d_roc, d_pr = np.array(d_roc), np.array(d_pr)
print(f"ΔROC-AUC: mean {d_roc.mean():+.4f}, 95% CI [{np.percentile(d_roc, 2.5):+.4f}, "
      f"{np.percentile(d_roc, 97.5):+.4f}], P(Δ>0)={np.mean(d_roc > 0):.3f}")
print(f"ΔPR-AUC : mean {d_pr.mean():+.4f}, 95% CI [{np.percentile(d_pr, 2.5):+.4f}, "
      f"{np.percentile(d_pr, 97.5):+.4f}], P(Δ>0)={np.mean(d_pr > 0):.3f}")
