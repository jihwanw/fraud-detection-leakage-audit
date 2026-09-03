#!/usr/bin/env python3
"""12_referee_experiments.py — Referee-response batch.

A. Multi-seed x multi-model increment distribution (R1-1,2)
B. Leakage dose-response: reveal horizon allowed into features (R1-4)
C. Industry adjustment: 1-digit SIC dummies + industry-year adjusted
   fraud_neighbor_ratio (R2-3)
D. Homonym variants on a COMMON test sample + precision@top-1% (R1-5,6)
All: fixed temporal split (train 2004-2017 / test 2018-2023), fraud label.
"""
import pandas as pd
import numpy as np
import networkx as nx
import os
import builtins
import getpass
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import RUSBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
YEARS = range(2004, 2025)
WINDOW = 4

RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
NET4 = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]

# ---------- shared inputs ----------
res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
reveal = res[res.res_fraud == 1].groupby("company_fkey").file_year.min()

resl = res.dropna(subset=["cik", "res_begin_date", "res_end_date"]).copy()
resl["beg"] = pd.to_datetime(resl.res_begin_date).dt.year
resl["end"] = pd.to_datetime(resl.res_end_date).dt.year
SPELLS = {}
for r in resl[resl.res_fraud == 1].itertuples():
    SPELLS.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))

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

net = pd.read_parquet(f"{DATA}/network_panel_ext.parquet")
net["cik"] = pd.to_numeric(net.company_fkey, errors="coerce")
net = net.dropna(subset=["cik"])
M = f[["cik", "year", "gvkey"] + FEAT].merge(
    net[["cik", "year"] + NET4], on=["cik", "year"], how="inner")
M = M.replace([np.inf, -np.inf], 0.0)
M["label"] = [1 if any(b <= y <= e and fy > y for b, e, fy in SPELLS.get(c, ())) else 0
              for c, y in zip(M.cik, M.year)]
TR = M[(M.year >= 2004) & (M.year <= 2017)]
TE = M[(M.year >= 2018) & (M.year <= 2023)].reset_index(drop=True)
print(f"panel {len(M):,} | test pos {TE.label.sum()}", flush=True)


def prec_at_top(y, s, frac=0.01):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-s)[:k]
    return y[idx].mean()


# ========== A. multi-seed x multi-model ==========
print("\n[A] multi-seed x multi-model (Δ = FIN+NET4 − FIN, ROC)", flush=True)
for mdl_name in ["rusboost", "hgb", "logit"]:
    deltas, p1s = [], []
    seeds = range(10) if mdl_name != "logit" else [0]
    for sd in seeds:
        out = {}
        for name, cols in [("FIN", FEAT), ("NET", FEAT + NET4)]:
            if mdl_name == "rusboost":
                mdl = RUSBoostClassifier(
                    estimator=DecisionTreeClassifier(min_samples_leaf=5),
                    n_estimators=300, learning_rate=0.1, random_state=sd)
                mdl.fit(TR[cols], TR.label)
                s = mdl.predict_proba(TE[cols])[:, 1]
            elif mdl_name == "hgb":
                mdl = HistGradientBoostingClassifier(random_state=sd,
                                                     class_weight="balanced")
                mdl.fit(TR[cols], TR.label)
                s = mdl.predict_proba(TE[cols])[:, 1]
            else:
                sc = StandardScaler().fit(TR[cols])
                mdl = LogisticRegression(max_iter=3000, class_weight="balanced")
                mdl.fit(sc.transform(TR[cols]), TR.label)
                s = mdl.predict_proba(sc.transform(TE[cols]))[:, 1]
            out[name] = s
        deltas.append(roc_auc_score(TE.label, out["NET"])
                      - roc_auc_score(TE.label, out["FIN"]))
        p1s.append((prec_at_top(TE.label.values, out["FIN"]),
                    prec_at_top(TE.label.values, out["NET"])))
    d = np.array(deltas)
    print(f"  {mdl_name:9s} ΔROC mean {d.mean():+.4f} sd {d.std():.4f} "
          f"min {d.min():+.4f} max {d.max():+.4f} (n={len(d)}) | "
          f"prec@1% FIN {np.mean([a for a, _ in p1s]):.3f} "
          f"NET {np.mean([b for _, b in p1s]):.3f}", flush=True)

# ========== B. leakage dose-response ==========
print("\n[B] leakage dose-response (reveal horizon allowed into neighbor ratio)",
      flush=True)
d0 = pd.read_parquet(f"{DATA}/directors.parquet")
d0 = d0[d0.is_bdmem_pers == 1].copy()
d0["name"] = (d0.first_name.fillna("") + "|" + d0.middle_name.fillna("") + "|"
              + d0.last_name.fillna("")).str.upper().str.strip()
d0 = d0[(d0.first_name.notna()) & (d0.last_name.notna())]
d0["year"] = pd.to_datetime(d0.eff_date).dt.year
d0 = d0.dropna(subset=["year"])
d0["year"] = d0.year.astype(int)
d0 = d0[["company_fkey", "name", "year"]].drop_duplicates()

nbrs_cache = {}
for t in YEARS:
    act = d0[(d0.year >= t - WINDOW) & (d0.year <= t)]
    by = act.groupby("name").company_fkey.apply(lambda s: list(set(s)))
    by = by[by.str.len() >= 2]
    G = nx.Graph()
    G.add_nodes_from(act.company_fkey.unique())
    for comps in by:
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                G.add_edge(comps[i], comps[j])
    nbrs_cache[t] = {c: set(G.neighbors(c)) for c in G.nodes()}

for horizon in [0, 1, 2, 3, 99]:
    rows = []
    for t in YEARS:
        allowed = set(reveal[reveal < t + horizon].index) if horizon < 99 \
            else set(reveal.index)
        for c, nb in nbrs_cache[t].items():
            rows.append({"company_fkey": c, "year": t,
                         "fnr": len(nb & allowed) / len(nb) if nb else 0.0})
    nv = pd.DataFrame(rows)
    nv["cik"] = pd.to_numeric(nv.company_fkey, errors="coerce")
    mm = M[["cik", "year", "label"] + FEAT].merge(
        nv[["cik", "year", "fnr"]], on=["cik", "year"], how="inner")
    tr = mm[(mm.year >= 2004) & (mm.year <= 2017)]
    te = mm[(mm.year >= 2018) & (mm.year <= 2023)]
    mdl = RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                             n_estimators=300, learning_rate=0.1, random_state=42)
    cols = FEAT + ["fnr"]
    mdl.fit(tr[cols], tr.label)
    s = mdl.predict_proba(te[cols])[:, 1]
    tag = "clean (<t)" if horizon == 0 else (f"reveal < t+{horizon}"
                                             if horizon < 99 else "all (no constraint)")
    print(f"  {tag:22s} ROC={roc_auc_score(te.label, s):.4f}", flush=True)

# ========== C. industry adjustment ==========
print("\n[C] industry adjustment (1-digit SIC)", flush=True)
u = p = None
for line in open("/Users/jihwanw/PhD/.wrds_config"):
    if "USERNAME" in line:
        u = line.split("=")[1].strip()
    elif "PASSWORD" in line:
        p = line.split("=")[1].strip()
os.environ["PGPASSWORD"] = p
builtins.input = lambda x="": u
getpass.getpass = lambda x="": p
import wrds  # noqa: E402
db = wrds.Connection(wrds_username=u)
sic = db.raw_sql("SELECT gvkey, sic FROM comp.company")
db.close()
Mi = M.merge(sic, on="gvkey", how="left")
Mi["sic1"] = Mi.sic.astype(str).str[0].fillna("9")
ind_mean = Mi.groupby(["sic1", "year"]).fraud_neighbor_ratio.transform("mean")
Mi["fnr_indadj"] = Mi.fraud_neighbor_ratio - ind_mean
dums = pd.get_dummies(Mi.sic1, prefix="sic").astype(float)
Mi = pd.concat([Mi, dums], axis=1)
DUM = list(dums.columns)
tr = Mi[(Mi.year >= 2004) & (Mi.year <= 2017)]
te = Mi[(Mi.year >= 2018) & (Mi.year <= 2023)]
for name, cols in [
    ("FIN + SIC dummies", FEAT + DUM),
    ("FIN + SIC + NET4", FEAT + DUM + NET4),
    ("FIN + SIC + ind-adj fnr", FEAT + DUM + ["fnr_indadj", "degree", "pagerank"]),
]:
    mdl = RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                             n_estimators=300, learning_rate=0.1, random_state=42)
    mdl.fit(tr[cols], tr.label)
    s = mdl.predict_proba(te[cols])[:, 1]
    print(f"  {name:26s} ROC={roc_auc_score(te.label, s):.4f}", flush=True)

# ========== D. homonym variants on COMMON sample ==========
print("\n[D] homonym variants, COMMON test sample", flush=True)
nf = d0.groupby("name").company_fkey.nunique()
mid_ok = d0.name.str.split("|").str[1] != ""
variants = {"V0": d0, "V1": d0[d0.name.map(nf) < 5], "V2": d0[mid_ok]}
panels = {}
for vn, dv in variants.items():
    rows = []
    for t in YEARS:
        act = dv[(dv.year >= t - WINDOW) & (dv.year <= t)]
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
                         "fraud_neighbor_ratio": len(nb & past) / len(nb) if nb else 0})
    pv = pd.DataFrame(rows)
    pv["cik"] = pd.to_numeric(pv.company_fkey, errors="coerce")
    panels[vn] = pv.dropna(subset=["cik"])

common = None
for vn, pv in panels.items():
    k = set(map(tuple, pv[["cik", "year"]].values))
    common = k if common is None else (common & k)
print(f"  common firm-years: {len(common):,}")
base = M[["cik", "year", "label"] + FEAT]
base = base[[tuple(x) in common for x in base[["cik", "year"]].values]]
for vn, pv in panels.items():
    mm = base.merge(pv[["cik", "year"] + NET4], on=["cik", "year"], how="inner")
    tr = mm[(mm.year >= 2004) & (mm.year <= 2017)]
    te = mm[(mm.year >= 2018) & (mm.year <= 2023)]
    out = {}
    for name, cols in [("FIN", FEAT), ("NET", FEAT + NET4)]:
        mdl = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(min_samples_leaf=5),
            n_estimators=300, learning_rate=0.1, random_state=42)
        mdl.fit(tr[cols], tr.label)
        out[name] = roc_auc_score(te.label, mdl.predict_proba(te[cols])[:, 1])
    print(f"  {vn}: test pos {te.label.sum()} | FIN {out['FIN']:.4f} | "
          f"NET {out['NET']:.4f} | Δ {out['NET'] - out['FIN']:+.4f}", flush=True)
