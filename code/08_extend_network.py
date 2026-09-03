#!/usr/bin/env python3
"""08_extend_network.py — Extended leak-free network features.

Adds to the 02 panel, all using ONLY fraud revealed before year t:
  - fraud_neighbor_2hop_ratio: share of 2-hop neighbors with past-revealed fraud
  - fraud_dir_cnt / fraud_dir_share: directors on firm c's current board who were
    linked (board event) to ANY other firm whose fraud was revealed before t,
    with the person-firm link occurring before the revelation year
    (the "fraud-associated director" channel; cf. director recidivism).
Output: data/network_panel_ext.parquet
"""
import pandas as pd
import numpy as np
import networkx as nx
import os

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
fraud_first_reveal = (res[res.res_fraud == 1]
                      .groupby("company_fkey").file_year.min())

# person -> earliest year they were linked to each firm
first_link = d.groupby(["name", "company_fkey"]).year.min().reset_index()
first_link = first_link.merge(fraud_first_reveal.rename("reveal_year"),
                              left_on="company_fkey", right_index=True, how="inner")
# person joined the fraud firm's board before its fraud was revealed
first_link = first_link[first_link.year <= first_link.reveal_year]
print(f"fraud-firm director links (joined before reveal): {len(first_link):,}")

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

    past = set(fraud_first_reveal[fraud_first_reveal < t].index)
    # fraud-associated persons as of t: linked before reveal AND revealed before t
    assoc = set(first_link[first_link.reveal_year < t].name)
    firm_board = active.groupby("company_fkey").name.apply(set)

    deg = dict(G.degree())
    pr = nx.pagerank(G) if G.number_of_edges() else {}
    for c in G.nodes():
        nbrs = set(G.neighbors(c))
        two_hop = set()
        for n in nbrs:
            two_hop |= set(G.neighbors(n))
        two_hop -= (nbrs | {c})
        board = firm_board.get(c, set())
        n_assoc = len(board & assoc)
        rows.append({
            "company_fkey": c, "year": t,
            "degree": deg.get(c, 0), "pagerank": pr.get(c, 0.0),
            "fraud_neighbor_cnt": len(nbrs & past),
            "fraud_neighbor_ratio": len(nbrs & past) / len(nbrs) if nbrs else 0.0,
            "fraud_2hop_ratio": len(two_hop & past) / len(two_hop) if two_hop else 0.0,
            "fraud_dir_cnt": n_assoc,
            "fraud_dir_share": n_assoc / len(board) if board else 0.0,
        })
    print(f"  {t}: firms {G.number_of_nodes():,}, past-fraud {len(past):,}, "
          f"assoc-directors {len(assoc):,}", flush=True)

panel = pd.DataFrame(rows)
panel.to_parquet(f"{DATA}/network_panel_ext.parquet")
print(f"saved network_panel_ext.parquet: {len(panel):,} rows")
