#!/usr/bin/env python3
"""13_construction_grid.py — Systematic ablation of network construction choices.
Grid: activity window {3,4,5} x edge weighting {unweighted, weighted}.
Report Δ(FIN+NET4 − FIN) ROC on the fixed split for each cell (R1-3)."""
import pandas as pd
import numpy as np
import networkx as nx
import os
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
YEARS = range(2004, 2025)

RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
NET4 = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]

res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
reveal = res[res.res_fraud == 1].groupby("company_fkey").file_year.min()
resl = res.dropna(subset=["cik", "res_begin_date", "res_end_date"]).copy()
resl["beg"] = pd.to_datetime(resl.res_begin_date).dt.year
resl["end"] = pd.to_datetime(resl.res_end_date).dt.year
sp = {}
for r in resl[resl.res_fraud == 1].itertuples():
    sp.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))

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

d0 = pd.read_parquet(f"{DATA}/directors.parquet")
d0 = d0[d0.is_bdmem_pers == 1].copy()
d0["name"] = (d0.first_name.fillna("") + "|" + d0.middle_name.fillna("") + "|"
              + d0.last_name.fillna("")).str.upper().str.strip()
d0 = d0[(d0.first_name.notna()) & (d0.last_name.notna())]
d0["year"] = pd.to_datetime(d0.eff_date).dt.year
d0 = d0.dropna(subset=["year"])
d0["year"] = d0.year.astype(int)
d0 = d0[["company_fkey", "name", "year"]].drop_duplicates()


def run_cell(window, weighted):
    rows = []
    for t in YEARS:
        act = d0[(d0.year >= t - window) & (d0.year <= t)]
        by = act.groupby("name").company_fkey.apply(lambda s: list(set(s)))
        by = by[by.str.len() >= 2]
        G = nx.Graph()
        G.add_nodes_from(act.company_fkey.unique())
        for comps in by:
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    if weighted and G.has_edge(comps[i], comps[j]):
                        G[comps[i]][comps[j]]["weight"] += 1
                    else:
                        G.add_edge(comps[i], comps[j], weight=1)
        past = set(reveal[reveal < t].index)
        deg = dict(G.degree(weight="weight" if weighted else None))
        pr = (nx.pagerank(G, weight="weight" if weighted else None)
              if G.number_of_edges() else {})
        for c in G.nodes():
            nb = set(G.neighbors(c))
            rows.append({"company_fkey": c, "year": t, "degree": deg.get(c, 0),
                         "pagerank": pr.get(c, 0.0),
                         "fraud_neighbor_cnt": len(nb & past),
                         "fraud_neighbor_ratio": len(nb & past) / len(nb) if nb else 0})
    pv = pd.DataFrame(rows)
    pv["cik"] = pd.to_numeric(pv.company_fkey, errors="coerce")
    pv = pv.dropna(subset=["cik"])
    m = f[["cik", "year"] + FEAT].merge(pv[["cik", "year"] + NET4],
                                        on=["cik", "year"], how="inner")
    m = m.replace([np.inf, -np.inf], 0.0)
    m["label"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp.get(c, ()))
                  else 0 for c, y in zip(m.cik, m.year)]
    tr = m[(m.year >= 2004) & (m.year <= 2017)]
    te = m[(m.year >= 2018) & (m.year <= 2023)]
    out = {}
    for name, cols in [("FIN", FEAT), ("NET", FEAT + NET4)]:
        mdl = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(min_samples_leaf=5),
            n_estimators=300, learning_rate=0.1, random_state=42)
        mdl.fit(tr[cols], tr.label)
        out[name] = roc_auc_score(te.label, mdl.predict_proba(te[cols])[:, 1])
    return out["FIN"], out["NET"], te.label.sum()


print("window | edges      | FIN ROC | NET ROC | Δ")
for w in [3, 4, 5]:
    for wt in [False, True]:
        fin, netv, pos = run_cell(w, wt)
        print(f"  {w}    | {'weighted' if wt else 'unweighted':10s} | "
              f"{fin:.4f} | {netv:.4f} | {netv - fin:+.4f} (pos {pos})", flush=True)
