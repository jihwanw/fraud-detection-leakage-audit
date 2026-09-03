#!/usr/bin/env python3
"""04_ongoing_fraud_eval.py — Literature-standard label (Dechow 2011; Bao 2020).

Label: firm-year t is positive if a FRAUD restatement (res_fraud=1) filed
LATER (file_year > t) covers year t in its misstatement period
(res_begin_date.year <= t <= res_end_date.year).
This is the standard "detect ongoing, not-yet-revealed misstatement" task.

Timing discipline:
  - Financial features of year t are the AS-PUBLISHED (possibly misstated)
    numbers — legitimate: this is what a detector would observe at the time.
  - Network fraud-exposure features use only fraud revealed BEFORE year t
    (unchanged from 02/03).
  - Temporal split: train 2004-2017, test 2018-2023. For test years we require
    file_year > t, so test labels come from filings after the prediction year
    (mostly 2019+), keeping the evaluation honest.

Outputs PR-AUC / ROC-AUC for financial-only vs financial+network,
plus the secondary broader label (fraud OR SEC investigation) for reference.
"""
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

# ---------- misstatement spells ----------
res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res = res.dropna(subset=["cik", "res_begin_date", "res_end_date"])
res["file_year"] = pd.to_datetime(res.file_date).dt.year
res["beg"] = pd.to_datetime(res.res_begin_date).dt.year
res["end"] = pd.to_datetime(res.res_end_date).dt.year


def spell_lookup(df):
    """cik -> list of (beg, end, file_year) misstatement spells."""
    out = {}
    for r in df.itertuples():
        out.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))
    return out


fraud_spells = spell_lookup(res[res.res_fraud == 1])
broad_spells = spell_lookup(res[(res.res_fraud == 1) | (res.res_sec_investigation == 1)])
print(f"fraud spells: {sum(len(v) for v in fraud_spells.values()):,} "
      f"({len(fraud_spells):,} firms); broad: "
      f"{sum(len(v) for v in broad_spells.values()):,} ({len(broad_spells):,} firms)")


def label_row(cik, year, spells):
    for beg, end, fy in spells.get(cik, ()):
        if beg <= year <= end and fy > year:   # ongoing at t, revealed after t
            return 1
    return 0


# ---------- features (same construction as 03) ----------
f = pd.read_parquet(f"{DATA}/funda.parquet")
f["cik"] = pd.to_numeric(f.cik, errors="coerce")
f = f.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"])
NUM = ["rect", "sale", "cogs", "at", "act", "ppent", "dp", "xsga", "dltt", "lct", "ni", "oancf"]
f[NUM] = f[NUM].astype("float64")
f = f.drop_duplicates(["gvkey", "fyear"], keep="last")
g = f.groupby("gvkey")
prev = g[NUM].shift(1)


def safe(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((b != 0) & np.isfinite(a) & np.isfinite(b), a / b, np.nan)


X = pd.DataFrame(index=f.index)
X["dsri"] = safe(safe(f.rect, f.sale), safe(prev.rect, prev.sale))
X["gmi"] = safe(safe(prev.sale - prev.cogs, prev.sale), safe(f.sale - f.cogs, f.sale))
X["aqi"] = safe(1 - safe(f.act + f.ppent, f["at"]), 1 - safe(prev.act + prev.ppent, prev["at"]))
X["sgi"] = safe(f.sale, prev.sale)
X["depi"] = safe(safe(prev.dp, prev.dp + prev.ppent), safe(f.dp, f.dp + f.ppent))
X["sgai"] = safe(safe(f.xsga, f.sale), safe(prev.xsga, prev.sale))
X["lvgi"] = safe(safe(f.dltt + f.lct, f["at"]), safe(prev.dltt + prev.lct, prev["at"]))
X["tata"] = safe(f.ni - f.oancf, f["at"])
X["cik"] = f.cik.values
X["year"] = f.fyear.values
X = X.dropna(subset=["year"])
X["year"] = X.year.astype(int)

net = pd.read_parquet(f"{DATA}/network_panel.parquet")
net["cik"] = pd.to_numeric(net.company_fkey, errors="coerce")
net = net.dropna(subset=["cik"])

m = X.merge(net[["cik", "year", "degree", "pagerank", "fraud_neighbor_cnt",
                 "fraud_neighbor_ratio"]], on=["cik", "year"], how="inner")

m["label"] = [label_row(c, y, fraud_spells) for c, y in zip(m.cik, m.year)]
m["label_broad"] = [label_row(c, y, broad_spells) for c, y in zip(m.cik, m.year)]

FIN = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]
NET = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]
m[FIN] = m[FIN].clip(-10, 10)
m = m.dropna(subset=FIN)

print(f"panel: {len(m):,} firm-years, {m.cik.nunique():,} firms")
print(f"ongoing-fraud positives: {m.label.sum():,} ({m.label.mean():.3%})")
print(f"broad-label positives:   {m.label_broad.sum():,} ({m.label_broad.mean():.3%})")

train = m[m.year <= 2017]
test = m[(m.year >= 2018) & (m.year <= 2023)]

for target in ["label", "label_broad"]:
    print(f"\n===== target: {target} =====")
    print(f"train {len(train):,} ({train[target].sum()} pos) | "
          f"test {len(test):,} ({test[target].sum()} pos) | "
          f"test base rate {test[target].mean():.4f}")
    for name, cols in [("financial only", FIN), ("financial + network", FIN + NET)]:
        sc = StandardScaler().fit(train[cols])
        Xtr, Xte = sc.transform(train[cols]), sc.transform(test[cols])
        for mdl_name, mdl in [
            ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("gbm", HistGradientBoostingClassifier(random_state=42,
                                                   class_weight="balanced")),
        ]:
            mdl.fit(Xtr, train[target])
            s = mdl.predict_proba(Xte)[:, 1]
            print(f"  {name:22s} {mdl_name:6s}  "
                  f"PR-AUC={average_precision_score(test[target], s):.4f}  "
                  f"ROC-AUC={roc_auc_score(test[target], s):.4f}")

m.to_parquet(f"{DATA}/analysis_panel.parquet")
print("\nsaved analysis_panel.parquet")
