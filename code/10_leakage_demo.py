#!/usr/bin/env python3
"""10_leakage_demo.py — How label leakage manufactures 'network alpha'.

Same pipeline, same model, ONE change: fraud_neighbor features computed with
ALL fraud spells (including those revealed at or after year t) instead of only
past-revealed fraud. This mimics the common mistake of building exposure
features from a completed fraud database without respecting reveal dates.
Contrast on the fixed temporal split, fraud label.
"""
import pandas as pd
import numpy as np
import networkx as nx
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
YEARS = range(2004, 2025)
WINDOW = 4

d = pd.read_parquet(f"{DATA}/directors.parquet")
d = d[d.is_bdmem_pers == 1].copy()
d["name"] = (d.first_name.fillna("") + "|" + d.middle_name.fillna("") + "|"
             + d.last_name.fillna("")).str.upper().str.strip()
d = d[(d.first_name.notna()) & (d.last_name.notna())]
d["year"] = pd.to_datetime(d.eff_date).dt.year
d = d.dropna(subset=["year"])
d["year"] = d.year.astype(int)
d = d[["company_fkey", "name", "year"]].drop_duplicates()

res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
fraud_reveal = res[res.res_fraud == 1].groupby("company_fkey").file_year.min()
fraud_all = set(fraud_reveal.index)  # leaky: known regardless of reveal timing

rows = []
for t in YEARS:
    active = d[(d.year >= t - WINDOW) & (d.year <= t)]
    by_name = active.groupby("name").company_fkey.apply(lambda s: list(set(s)))
    multi = by_name[by_name.str.len() >= 2]
    G = nx.Graph()
    G.add_nodes_from(active.company_fkey.unique())
    for comps in multi:
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                G.add_edge(comps[i], comps[j])
    past = set(fraud_reveal[fraud_reveal < t].index)
    for c in G.nodes():
        nbrs = set(G.neighbors(c))
        rows.append({
            "company_fkey": c, "year": t,
            "fnr_clean": len(nbrs & past) / len(nbrs) if nbrs else 0.0,
            "fnr_leaky": len(nbrs & fraud_all) / len(nbrs) if nbrs else 0.0,
            "self_leak": 1 if c in fraud_all else 0,  # extreme leak: own future fraud
        })
net = pd.DataFrame(rows)
net["cik"] = pd.to_numeric(net.company_fkey, errors="coerce")
net = net.dropna(subset=["cik"])

RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
f = pd.read_parquet(f"{DATA}/funda_bao.parquet")
f["cik"] = pd.to_numeric(f.cik, errors="coerce")
f = f.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"]).drop_duplicates(
    ["gvkey", "fyear"], keep="last")
f[RAW] = f[RAW].astype("float64").fillna(0.0)
for c in RAW:
    if c not in ("at", "prcc_f", "csho"):
        f[c] = f[c] / f["at"]
f["log_at"] = np.log1p(f["at"])
FEAT = [c for c in RAW if c != "at"] + ["log_at"]
f["year"] = f.fyear.astype(int)

m = f[["cik", "year"] + FEAT].merge(net[["cik", "year", "fnr_clean", "fnr_leaky",
                                         "self_leak"]], on=["cik", "year"], how="inner")
m = m.replace([np.inf, -np.inf], 0.0)

resl = res.dropna(subset=["res_begin_date", "res_end_date"]).copy()
resl["cik"] = pd.to_numeric(resl.company_fkey, errors="coerce")
resl["beg"] = pd.to_datetime(resl.res_begin_date).dt.year
resl["end"] = pd.to_datetime(resl.res_end_date).dt.year
sp = {}
for r in resl[resl.res_fraud == 1].itertuples():
    sp.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))
m["label"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp.get(c, ())) else 0
              for c, y in zip(m.cik, m.year)]

train = m[(m.year >= 2004) & (m.year <= 2017)]
test = m[(m.year >= 2018) & (m.year <= 2023)]
print(f"test pos {test.label.sum()}")


def run(cols, tag):
    mdl = RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                             n_estimators=300, learning_rate=0.1, random_state=42)
    mdl.fit(train[cols], train.label)
    s = mdl.predict_proba(test[cols])[:, 1]
    print(f"  {tag:34s} ROC={roc_auc_score(test.label, s):.4f} "
          f"PR={average_precision_score(test.label, s):.4f}")


print("LEAKAGE DEMONSTRATION (fixed split, fraud label):")
run(FEAT, "FIN only")
run(FEAT + ["fnr_clean"], "FIN + neighbor ratio (leak-free)")
run(FEAT + ["fnr_leaky"], "FIN + neighbor ratio (LEAKY)")
run(FEAT + ["fnr_leaky", "self_leak"], "FIN + leaky ratio + self flag")
