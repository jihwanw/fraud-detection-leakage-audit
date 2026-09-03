#!/usr/bin/env python3
"""17_network_visual.py — Network figure + graph-property table for the paper.

Fig 5: 2018 interlock network, largest connected component, nodes sized by
degree; firms with past-revealed fraud (reveal < 2018) in red; a zoom panel
shows the 2-hop neighborhood of the highest-exposure firm.
Table: topology statistics for snapshot years 2008/2013/2018/2023.
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "paper", "figures")
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
reveal = res[res.res_fraud == 1].groupby("company_fkey").file_year.min()


def build(t):
    act = d[(d.year >= t - WINDOW) & (d.year <= t)]
    by = act.groupby("name").company_fkey.apply(lambda s: list(set(s)))
    by = by[by.str.len() >= 2]
    G = nx.Graph()
    G.add_nodes_from(act.company_fkey.unique())
    for comps in by:
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                G.add_edge(comps[i], comps[j])
    return G


# ---------- topology table ----------
print("year & nodes & edges & density(x1e4) & mean deg & max deg & "
      "clustering & giant comp % & components & fraud firms")
rows = []
for t in [2008, 2013, 2018, 2023]:
    G = build(t)
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len)
    Gg = G.subgraph(giant)
    degs = [dd for _, dd in G.degree()]
    past = set(reveal[reveal < t].index) & set(G.nodes())
    row = dict(
        year=t, n=G.number_of_nodes(), e=G.number_of_edges(),
        dens=nx.density(G) * 1e4, mdeg=np.mean(degs), xdeg=max(degs),
        clust=nx.average_clustering(G), giant=len(giant) / G.number_of_nodes(),
        ncomp=len(comps), fraud=len(past))
    rows.append(row)
    print(f"{t} & {row['n']:,} & {row['e']:,} & {row['dens']:.2f} & "
          f"{row['mdeg']:.2f} & {row['xdeg']} & {row['clust']:.3f} & "
          f"{row['giant']:.1%} & {row['ncomp']:,} & {row['fraud']}")
pd.DataFrame(rows).to_csv(f"{DATA}/topology_table.csv", index=False)

# ---------- figure: 2018 giant component ----------
t = 2018
G = build(t)
past = set(reveal[reveal < t].index)
giant = max(nx.connected_components(G), key=len)
Gg = G.subgraph(giant).copy()
print(f"\n2018 giant component: {Gg.number_of_nodes():,} nodes, "
      f"{Gg.number_of_edges():,} edges, fraud in giant: {len(set(Gg) & past)}")

pos = nx.spring_layout(Gg, seed=42, k=0.06, iterations=60)
deg = dict(Gg.degree())
fraud_nodes = [n for n in Gg if n in past]
clean_nodes = [n for n in Gg if n not in past]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.2),
                             gridspec_kw={"width_ratios": [1.35, 1]})
nx.draw_networkx_edges(Gg, pos, ax=a1, alpha=0.08, width=0.4)
nx.draw_networkx_nodes(Gg, pos, nodelist=clean_nodes, ax=a1,
                       node_size=[4 + deg[n] * 1.2 for n in clean_nodes],
                       node_color="#4878a8", alpha=0.55, linewidths=0)
nx.draw_networkx_nodes(Gg, pos, nodelist=fraud_nodes, ax=a1,
                       node_size=[14 + deg[n] * 2 for n in fraud_nodes],
                       node_color="#c0392b", alpha=0.95, linewidths=0)
a1.set_title(f"(a) Board interlock network, {t} snapshot\n"
             f"giant component: {Gg.number_of_nodes():,} firms, "
             f"{Gg.number_of_edges():,} edges; "
             f"red = fraud revealed before {t} (n={len(fraud_nodes)})",
             fontsize=10)
a1.axis("off")

# zoom: 2-hop neighborhood of the firm with highest fraud-neighbor count
best, cnt = None, -1
for n in Gg:
    c = sum(1 for x in Gg.neighbors(n) if x in past)
    if c > cnt:
        best, cnt = n, c
hood = {best} | set(Gg.neighbors(best))
for n in list(hood):
    hood |= set(Gg.neighbors(n))
Gz = Gg.subgraph(hood).copy()
posz = nx.spring_layout(Gz, seed=7, k=0.35)
fz = [n for n in Gz if n in past]
cz = [n for n in Gz if n not in past and n != best]
nx.draw_networkx_edges(Gz, posz, ax=a2, alpha=0.25, width=0.7)
nx.draw_networkx_nodes(Gz, posz, nodelist=cz, ax=a2, node_size=60,
                       node_color="#4878a8", alpha=0.7, linewidths=0)
nx.draw_networkx_nodes(Gz, posz, nodelist=fz, ax=a2, node_size=110,
                       node_color="#c0392b", alpha=0.95, linewidths=0)
nx.draw_networkx_nodes(Gz, posz, nodelist=[best], ax=a2, node_size=220,
                       node_color="#e8a13a", edgecolors="black", linewidths=1.2)
a2.set_title(f"(b) Two-hop neighborhood of the highest-exposure firm\n"
             f"(orange; {cnt} of its neighbors had revealed fraud)",
             fontsize=10)
a2.axis("off")
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_network.pdf", dpi=250, bbox_inches="tight")
plt.savefig(f"{FIG}/fig5_network.png", dpi=200, bbox_inches="tight")
print("saved fig5_network")
