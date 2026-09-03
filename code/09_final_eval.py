#!/usr/bin/env python3
"""09_final_eval.py — Comprehensive honest evaluation.

Feature sets: FIN(28 raw), FIN+NET4(basic), FIN+NET7(extended incl. 2-hop,
fraud-associated directors). Model: RUSBoost.
1) Fixed temporal split (train 2004-2017 / test 2018-2023) + paired bootstrap
2) Rolling-origin: 4 splits, test windows 2012-14, 2015-17, 2018-20, 2021-23
Primary label: ongoing fraud. Secondary: broad. All results printed for paper.
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
NET4 = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]
NET7 = NET4 + ["fraud_2hop_ratio", "fraud_dir_cnt", "fraud_dir_share"]

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


SP = {"label": spells(res[res.res_fraud == 1]),
      "label_broad": spells(res[(res.res_fraud == 1) | (res.res_sec_investigation == 1)])}

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
m = f[["cik", "year"] + FEAT].merge(net[["cik", "year"] + NET7],
                                    on=["cik", "year"], how="inner")
m = m.replace([np.inf, -np.inf], 0.0)
for tgt, sp in SP.items():
    m[tgt] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp.get(c, ())) else 0
              for c, y in zip(m.cik, m.year)]
print(f"panel {len(m):,} firm-years | fraud {m.label.sum()} "
      f"({m.label.mean():.3%}) | broad {m.label_broad.sum()} ({m.label_broad.mean():.3%})")

SETS = [("FIN", FEAT), ("FIN+NET4", FEAT + NET4), ("FIN+NET7", FEAT + NET7)]


def rus():
    return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                              n_estimators=300, learning_rate=0.1, random_state=42)


def evaluate(train, test, target):
    out = {}
    for name, cols in SETS:
        mdl = rus()
        mdl.fit(train[cols], train[target])
        s = mdl.predict_proba(test[cols])[:, 1]
        out[name] = (roc_auc_score(test[target], s),
                     average_precision_score(test[target], s), s)
    return out


# ---------- 1) fixed split + bootstrap ----------
train = m[(m.year >= 2004) & (m.year <= 2017)]
test = m[(m.year >= 2018) & (m.year <= 2023)].reset_index(drop=True)
for target in ["label", "label_broad"]:
    print(f"\n===== FIXED SPLIT | {target} | test pos {test[target].sum()} =====")
    r = evaluate(train, test, target)
    for name, (roc, pr, _) in r.items():
        print(f"  {name:9s} ROC={roc:.4f} PR={pr:.4f}")
    y = test[target].values
    for cand in ["FIN+NET4", "FIN+NET7"]:
        d = []
        for _ in range(2000):
            idx = rng.randint(0, len(y), len(y))
            if y[idx].sum() < 5:
                continue
            d.append(roc_auc_score(y[idx], r[cand][2][idx])
                     - roc_auc_score(y[idx], r["FIN"][2][idx]))
        d = np.array(d)
        print(f"  Δ({cand}-FIN) ROC: {d.mean():+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] "
              f"P(>0)={np.mean(d > 0):.3f}")

# ---------- 2) rolling origin ----------
print("\n===== ROLLING ORIGIN (label) =====")
wins = [(2012, 2014), (2015, 2017), (2018, 2020), (2021, 2023)]
deltas = []
for a, b in wins:
    tr = m[(m.year >= 2004) & (m.year < a)]
    te = m[(m.year >= a) & (m.year <= b)]
    if te.label.sum() < 5:
        print(f"  {a}-{b}: skipped (test pos {te.label.sum()})")
        continue
    r = evaluate(tr, te, "label")
    d4 = r["FIN+NET4"][0] - r["FIN"][0]
    d7 = r["FIN+NET7"][0] - r["FIN"][0]
    deltas.append((d4, d7))
    print(f"  {a}-{b}: pos {te.label.sum():3d} | FIN {r['FIN'][0]:.3f} | "
          f"+NET4 Δ{d4:+.3f} | +NET7 Δ{d7:+.3f}")
if deltas:
    d4s, d7s = zip(*deltas)
    print(f"  mean ΔROC: NET4 {np.mean(d4s):+.4f} | NET7 {np.mean(d7s):+.4f} "
          f"| NET4>0 in {sum(x > 0 for x in d4s)}/{len(d4s)} windows"
          f" | NET7>0 in {sum(x > 0 for x in d7s)}/{len(d7s)}")
