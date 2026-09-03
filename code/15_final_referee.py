#!/usr/bin/env python3
"""15_final_referee.py — Remaining referee experiments.
C. Industry adjustment (SIC cached to file, WRDS retry)
D. Homonym variants on COMMON test sample
E. AAER-anchored alternative label: accounting-restatement spell covering t,
   filed after t, at a firm matched to a public SEC AAER (label validity check)
"""
import pandas as pd
import numpy as np
import networkx as nx
import os
import time
import builtins
import getpass
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
YEARS = range(2004, 2025)
WINDOW = 4
RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
NET4 = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]


def rus():
    return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                              n_estimators=300, learning_rate=0.1, random_state=42)


res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
reveal = res[res.res_fraud == 1].groupby("company_fkey").file_year.min()
resl = res.dropna(subset=["cik", "res_begin_date", "res_end_date"]).copy()
resl["beg"] = pd.to_datetime(resl.res_begin_date).dt.year
resl["end"] = pd.to_datetime(resl.res_end_date).dt.year


def spells(df):
    out = {}
    for r in df.itertuples():
        out.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))
    return out


sp_fraud = spells(resl[resl.res_fraud == 1])

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
M = f[["cik", "year", "gvkey"] + FEAT].merge(net[["cik", "year"] + NET4],
                                             on=["cik", "year"], how="inner")
M = M.replace([np.inf, -np.inf], 0.0)
M["label"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp_fraud.get(c, ()))
              else 0 for c, y in zip(M.cik, M.year)]

# ---------- C. industry ----------
print("[C] industry adjustment", flush=True)
try:
    sic_path = f"{DATA}/sic.parquet"
    if not os.path.exists(sic_path):
        from wrds_auth import connect
        for attempt in range(3):
            try:
                db = connect()
                sic = db.raw_sql("SELECT gvkey, sic FROM comp.company")
                db.close()
                sic.to_parquet(sic_path)
                break
            except Exception as e:
                print(f"  wrds attempt {attempt}: {e}")
                time.sleep(20)
    sic = pd.read_parquet(sic_path)
    Mi = M.merge(sic, on="gvkey", how="left")
    Mi["sic1"] = Mi.sic.astype(str).str[0].where(Mi.sic.notna(), "9")
    ind_mean = Mi.groupby(["sic1", "year"]).fraud_neighbor_ratio.transform("mean")
    Mi["fnr_indadj"] = Mi.fraud_neighbor_ratio - ind_mean
    dums = pd.get_dummies(Mi.sic1, prefix="sic").astype(float)
    Mi = pd.concat([Mi, dums], axis=1)
    DUM = list(dums.columns)
    tr = Mi[(Mi.year >= 2004) & (Mi.year <= 2017)]
    te = Mi[(Mi.year >= 2018) & (Mi.year <= 2023)]
    for name, cols in [("FIN+SIC", FEAT + DUM), ("FIN+SIC+NET4", FEAT + DUM + NET4),
                       ("FIN+SIC+indadj-fnr", FEAT + DUM + ["fnr_indadj", "degree", "pagerank"])]:
        mdl = rus()
        mdl.fit(tr[cols], tr.label)
        print(f"  {name:20s} ROC={roc_auc_score(te.label, mdl.predict_proba(te[cols])[:, 1]):.4f}",
              flush=True)


except Exception as e:
    print(f"  [C] SKIPPED: {str(e)[:100]}")

# ---------- D. homonym common sample ----------
print("\n[D] homonym variants, common test sample", flush=True)
d0 = pd.read_parquet(f"{DATA}/directors.parquet")
d0 = d0[d0.is_bdmem_pers == 1].copy()
d0["name"] = (d0.first_name.fillna("") + "|" + d0.middle_name.fillna("") + "|"
              + d0.last_name.fillna("")).str.upper().str.strip()
d0 = d0[(d0.first_name.notna()) & (d0.last_name.notna())]
d0["year"] = pd.to_datetime(d0.eff_date).dt.year
d0 = d0.dropna(subset=["year"])
d0["year"] = d0.year.astype(int)
d0 = d0[["company_fkey", "name", "year"]].drop_duplicates()
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
for pv in panels.values():
    k = set(map(tuple, pv[["cik", "year"]].values))
    common = k if common is None else (common & k)
print(f"  common firm-years: {len(common):,}", flush=True)
base = M[["cik", "year", "label"] + FEAT]
mask = [tuple(x) in common for x in base[["cik", "year"]].values]
base = base[mask]
for vn, pv in panels.items():
    mm = base.merge(pv[["cik", "year"] + NET4], on=["cik", "year"], how="inner")
    tr = mm[(mm.year >= 2004) & (mm.year <= 2017)]
    te = mm[(mm.year >= 2018) & (mm.year <= 2023)]
    out = {}
    for name, cols in [("FIN", FEAT), ("NET", FEAT + NET4)]:
        mdl = rus()
        mdl.fit(tr[cols], tr.label)
        out[name] = roc_auc_score(te.label, mdl.predict_proba(te[cols])[:, 1])
    print(f"  {vn}: pos {te.label.sum()} | FIN {out['FIN']:.4f} | "
          f"NET {out['NET']:.4f} | Δ {out['NET'] - out['FIN']:+.4f}", flush=True)

# ---------- E. AAER-anchored label ----------
print("\n[E] AAER-anchored alternative label", flush=True)
import re
import json
import urllib.request
aaer = pd.read_parquet(f"{DATA}/aaer_public.parquet")
req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json",
                             headers={"User-Agent": "Jihwan Woo research jihwan.woo@gmail.com"})
tick = json.loads(urllib.request.urlopen(req, timeout=30).read())
comp = pd.DataFrame(tick).T
comp["norm"] = (comp.title.str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True)
                .str.replace(r"\b(INC|CORP|CO|LTD|LLC|PLC|THE|COMPANY|CORPORATION|"
                             r"INCORPORATED|HOLDINGS?|GROUP)\b", "", regex=True).str.strip())
comp = comp[comp.norm.str.len() >= 5]
name2cik = dict(zip(comp.norm, comp.cik_str.astype(int)))


def norm(s):
    s = re.sub(r"\(.*?\)", "", s.upper())
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\b(INC|CORP|CO|LTD|LLC|PLC|THE|COMPANY|CORPORATION|INCORPORATED|"
               r"HOLDINGS?|GROUP)\b", "", s)
    return s.strip()


aaer_ciks = set()
for r in aaer.itertuples():
    n = norm(r.respondent)
    for cand, cik in name2cik.items():
        if cand and cand in n:
            aaer_ciks.add(cik)
sp_aaer = spells(resl[(resl.res_accounting == 1)
                      & (resl.cik.isin(aaer_ciks))])
M["label_aaer"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp_aaer.get(c, ()))
                   else 0 for c, y in zip(M.cik, M.year)]
print(f"  AAER-anchored positives: {M.label_aaer.sum():,} ({M.label_aaer.mean():.3%})")
tr = M[(M.year >= 2004) & (M.year <= 2017)]
te = M[(M.year >= 2018) & (M.year <= 2023)]
for name, cols in [("FIN", FEAT), ("FIN+NET4", FEAT + NET4)]:
    mdl = rus()
    mdl.fit(tr[cols], tr.label_aaer)
    print(f"  {name:9s} ROC={roc_auc_score(te.label_aaer, mdl.predict_proba(te[cols])[:, 1]):.4f} "
          f"(test pos {te.label_aaer.sum()})", flush=True)
