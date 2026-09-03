#!/usr/bin/env python3
"""02_build_panel.py — Leak-free board-interlock network features per firm-year.

Design (documented honestly):
- Person ID: normalized full name (first+middle+last, uppercase). The feed's
  do_off_pers_key is row-unique and unusable. Name matching risks homonym
  false positives; we require both first and last name present, and we report
  the share of high-frequency names as a data limitation.
- Board membership only (is_bdmem_pers == 1).
- Activity window: a person-company link is considered active in year t if the
  person has any board event at that company with eff_date in [t-4, t]
  (approximation: the feed records changes, not full tenure spells).
- LEAK RULE: fraud exposure features for prediction year t use only fraud
  restatements with file_date < Jan 1 of year t (publicly revealed before t).
- Output: data/network_panel.parquet (company_fkey x year features).
"""
import pandas as pd
import numpy as np
import networkx as nx
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

YEARS = range(2004, 2025)
WINDOW = 4  # years of board events considered "active"

d = pd.read_parquet(f"{DATA}/directors.parquet")
d = d[d.is_bdmem_pers == 1].copy()
d["name"] = (d.first_name.fillna("") + "|" + d.middle_name.fillna("") + "|"
             + d.last_name.fillna("")).str.upper().str.strip()
d = d[(d.first_name.notna()) & (d.last_name.notna())]
d["year"] = pd.to_datetime(d.eff_date).dt.year
d = d.dropna(subset=["year"])
d["year"] = d.year.astype(int)
d = d[["company_fkey", "name", "year"]].drop_duplicates()
print(f"board-member links: {len(d):,} rows, {d.name.nunique():,} names, "
      f"{d.company_fkey.nunique():,} firms")

# homonym diagnostics (top of the name frequency distribution)
freq = d.groupby("name").company_fkey.nunique().sort_values(ascending=False)
print(f"names at 5+ firms: {(freq >= 5).sum():,} "
      f"({(freq >= 5).mean():.2%} of names) — potential homonyms, reported as limitation")

res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
fraud_events = res[res.res_fraud == 1][["company_fkey", "file_year"]].drop_duplicates()
print(f"fraud restatement events: {len(fraud_events):,} firm-filings")

rows = []
for t in YEARS:
    active = d[(d.year >= t - WINDOW) & (d.year <= t)]
    # bipartite person-company -> company projection
    by_name = active.groupby("name").company_fkey.apply(list)
    by_name = by_name[by_name.str.len() >= 2]
    G = nx.Graph()
    G.add_nodes_from(active.company_fkey.unique())
    for comps in by_name:
        comps = list(set(comps))
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                w = G[comps[i]][comps[j]]["weight"] + 1 if G.has_edge(comps[i], comps[j]) else 1
                G.add_edge(comps[i], comps[j], weight=w)

    # LEAK RULE: only fraud revealed strictly before year t
    past_fraud = set(fraud_events[fraud_events.file_year < t].company_fkey)

    deg = dict(G.degree())
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {}
    for c in G.nodes():
        nbrs = list(G.neighbors(c))
        rows.append({
            "company_fkey": c, "year": t,
            "degree": deg.get(c, 0),
            "pagerank": pr.get(c, 0.0),
            "n_neighbors": len(nbrs),
            "fraud_neighbor_cnt": sum(1 for n in nbrs if n in past_fraud),
            "fraud_neighbor_ratio": (sum(1 for n in nbrs if n in past_fraud) / len(nbrs))
                                    if nbrs else 0.0,
        })
    print(f"  {t}: {G.number_of_nodes():,} firms, {G.number_of_edges():,} edges, "
          f"past-revealed fraud firms: {len(past_fraud):,}")

panel = pd.DataFrame(rows)
panel.to_parquet(f"{DATA}/network_panel.parquet")
print(f"saved network_panel.parquet: {len(panel):,} firm-years")
