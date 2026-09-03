#!/usr/bin/env python3
"""11_homonym_sensitivity.py — Is the network increment robust to homonym risk?

Three person-matching variants:
  V0 baseline: full name (first|middle|last), as in the main analysis
  V1 strict  : additionally drop names linked to >=5 firms over the full sample
               (most homonym-prone)
  V2 middle  : require a non-empty middle name (sharply reduces false merges)
For each variant: rebuild NET4 panel, fixed temporal split, RUSBoost,
report FIN+NET4 - FIN ROC delta. Fraud label.
"""
import pandas as pd
import numpy as np
import networkx as nx
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
YEARS = range(2004, 2025)
WINDOW = 4

d0 = pd.read_parquet(f"{DATA}/directors.parquet")
d0 = d0[d0.is_bdmem_pers == 1].copy()
d0["name"] = (d0.first_name.fillna("") + "|" + d0.middle_name.fillna("") + "|"
              + d0.last_name.fillna("")).str.upper().str.strip()
d0 = d0[(d0.first_name.notna()) & (d0.last_name.notna())]
d0["year"] = pd.to_datetime(d0.eff_date).dt.year
d0 = d0.dropna(subset=["year"])
d0["year"] = d0.year.astype(int)
d0 = d0[["company_fkey", "name", "year", "middle_name"]].drop_duplicates()

res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
reveal = res[res.res_fraud == 1].groupby("company_fkey").file_year.min()

nfirms = d0.groupby("name").company_fkey.nunique()
variants = {
    "V0_baseline": d0,
    "V1_drop5plus": d0[d0.name.map(nfirms) < 5],
    "V2_middlereq": d0[d0.middle_name.notna() & (d0.middle_name.str.strip() != "")],
}


def build_panel(d):
    rows = []
    for t in YEARS:
        act = d[(d.year >= t - WINDOW) & (d.year <= t)]
        by = act.groupby("name").company_fkey.apply(lambda s: list(set(s)))
        by = by[by.str.len() >= 2]
        G = nx.Graph()
        G.add_nodes_from(act.company_fkey.unique())
        for comps in by:
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    G.add_edge(comps[i], comps[j])
        past = set(reveal[reveal < t].index)
        deg = dict(G.degree())
        pr = nx.pagerank(G) if G.number_of_edges() else {}
        for c in G.nodes():
            nb = set(G.neighbors(c))
            rows.append({"company_fkey": c, "year": t, "degree": deg.get(c, 0),
                         "pagerank": pr.get(c, 0.0),
                         "fraud_neighbor_cnt": len(nb & past),
                         "fraud_neighbor_ratio": len(nb & past) / len(nb) if nb else 0.0})
    p = pd.DataFrame(rows)
    p["cik"] = pd.to_numeric(p.company_fkey, errors="coerce")
    return p.dropna(subset=["cik"])


RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
NET4 = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]
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

resl = res.dropna(subset=["res_begin_date", "res_end_date"]).copy()
resl["cik"] = pd.to_numeric(resl.company_fkey, errors="coerce")
resl["beg"] = pd.to_datetime(resl.res_begin_date).dt.year
resl["end"] = pd.to_datetime(resl.res_end_date).dt.year
sp = {}
for r in resl[resl.res_fraud == 1].itertuples():
    sp.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))


def rus():
    return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                              n_estimators=300, learning_rate=0.1, random_state=42)


for vname, d in variants.items():
    print(f"\n===== {vname}: {d.name.nunique():,} names, {len(d):,} links =====", flush=True)
    net = build_panel(d)
    m = f[["cik", "year"] + FEAT].merge(net[["cik", "year"] + NET4],
                                        on=["cik", "year"], how="inner")
    m = m.replace([np.inf, -np.inf], 0.0)
    m["label"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp.get(c, ())) else 0
                  for c, y in zip(m.cik, m.year)]
    tr = m[(m.year >= 2004) & (m.year <= 2017)]
    te = m[(m.year >= 2018) & (m.year <= 2023)]
    out = {}
    for name, cols in [("FIN", FEAT), ("FIN+NET4", FEAT + NET4)]:
        mdl = rus()
        mdl.fit(tr[cols], tr.label)
        s = mdl.predict_proba(te[cols])[:, 1]
        out[name] = roc_auc_score(te.label, s)
    print(f"  panel {len(m):,} | test pos {te.label.sum()} | "
          f"FIN {out['FIN']:.4f} | FIN+NET4 {out['FIN+NET4']:.4f} | "
          f"Δ {out['FIN+NET4'] - out['FIN']:+.4f}")
