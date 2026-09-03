#!/usr/bin/env python3
"""19_missingness_sensitivity.py — Missing-data specification audit.

Compares the literature-standard zero fill with train-only median imputation,
median imputation plus missingness indicators, and complete cases. Each method
uses the same real-data construction, temporal split, and network features.
Uncertainty uses ten training seeds and paired cluster bootstrap by firm.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "results" / "missingness_sensitivity.json"
RAW = [
    "act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
    "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
    "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f",
]
NET4 = ["degree", "pagerank", "fraud_neighbor_cnt", "fraud_neighbor_ratio"]


def classifier(seed: int):
    return RUSBoostClassifier(
        estimator=DecisionTreeClassifier(min_samples_leaf=5),
        n_estimators=300,
        learning_rate=0.1,
        random_state=seed,
    )


def cluster_interval(frame: pd.DataFrame, score_fin, score_net, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    labels = frame.label.to_numpy()
    clusters = {
        cik: np.asarray(indices, dtype=int)
        for cik, indices in frame.reset_index(drop=True).groupby("cik").indices.items()
    }
    firm_ids = np.asarray(list(clusters))
    deltas = []
    for _ in range(2000):
        sampled = rng.choice(firm_ids, size=len(firm_ids), replace=True)
        index = np.concatenate([clusters[cik] for cik in sampled])
        if labels[index].sum() < 5:
            continue
        deltas.append(
            roc_auc_score(labels[index], score_net[index])
            - roc_auc_score(labels[index], score_fin[index])
        )
    values = np.asarray(deltas)
    return {
        "mean": round(float(values.mean()), 4),
        "lo": round(float(np.percentile(values, 2.5)), 4),
        "hi": round(float(np.percentile(values, 97.5)), 4),
        "p_gt_zero": round(float((values > 0).mean()), 3),
        "replicates": int(len(values)),
    }


restatements = pd.read_parquet(DATA / "restatements.parquet")
restatements["cik"] = pd.to_numeric(restatements.company_fkey, errors="coerce")
restatements = restatements.dropna(subset=["cik", "res_begin_date", "res_end_date"])
restatements["file_year"] = pd.to_datetime(restatements.file_date).dt.year
restatements["beg"] = pd.to_datetime(restatements.res_begin_date).dt.year
restatements["end"] = pd.to_datetime(restatements.res_end_date).dt.year
spells = {}
for row in restatements[restatements.res_fraud == 1].itertuples():
    spells.setdefault(row.cik, []).append((row.beg, row.end, row.file_year))

fund = pd.read_parquet(DATA / "funda_bao.parquet")
fund["cik"] = pd.to_numeric(fund.cik, errors="coerce")
fund = fund.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"]).drop_duplicates(
    ["gvkey", "fyear"], keep="last"
)
fund[RAW] = fund[RAW].astype("float64")
for column in RAW:
    if column not in ("at", "prcc_f", "csho"):
        fund[column] = fund[column] / fund["at"]
fund["log_at"] = np.log1p(fund["at"])
features = [column for column in RAW if column != "at"] + ["log_at"]
fund["year"] = fund.fyear.astype(int)

network = pd.read_parquet(DATA / "network_panel_ext.parquet")
network["cik"] = pd.to_numeric(network.company_fkey, errors="coerce")
network = network.dropna(subset=["cik"])
panel = fund[["cik", "year"] + features].merge(
    network[["cik", "year"] + NET4], on=["cik", "year"], how="inner"
)
panel = panel.replace([np.inf, -np.inf], np.nan)
panel["label"] = [
    int(any(b <= year <= e and filed > year for b, e, filed in spells.get(cik, ())))
    for cik, year in zip(panel.cik, panel.year)
]

result = {
    "panel_rows": int(len(panel)),
    "rows_with_any_missing_financial_feature": int(panel[features].isna().any(axis=1).sum()),
    "missing_row_pct": round(float(panel[features].isna().any(axis=1).mean()) * 100, 2),
    "methods": {},
}

for method in ("zero", "train_median", "median_plus_indicators", "complete_case"):
    if method == "complete_case":
        analysis = panel.dropna(subset=features).copy()
    else:
        analysis = panel.copy()
    train = analysis[(analysis.year >= 2004) & (analysis.year <= 2017)].copy()
    test = analysis[(analysis.year >= 2018) & (analysis.year <= 2023)].copy().reset_index(drop=True)
    method_result = {
        "panel_rows": int(len(analysis)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_positives": int(test.label.sum()),
    }

    transformed = {}
    for name, columns in (("FIN", features), ("FIN_NET4", features + NET4)):
        if method == "zero":
            transformed[name] = (
                train[columns].fillna(0.0).to_numpy(),
                test[columns].fillna(0.0).to_numpy(),
            )
        elif method == "complete_case":
            transformed[name] = (train[columns].to_numpy(), test[columns].to_numpy())
        else:
            imputer = SimpleImputer(
                strategy="median", add_indicator=(method == "median_plus_indicators")
            )
            transformed[name] = (
                imputer.fit_transform(train[columns]),
                imputer.transform(test[columns]),
            )

    seed_deltas = []
    point_scores = {}
    for seed in list(range(10)) + [42]:
        scores = {}
        for name in ("FIN", "FIN_NET4"):
            x_train, x_test = transformed[name]
            model = classifier(seed)
            model.fit(x_train, train.label)
            scores[name] = model.predict_proba(x_test)[:, 1]
        delta = roc_auc_score(test.label, scores["FIN_NET4"]) - roc_auc_score(
            test.label, scores["FIN"]
        )
        if seed < 10:
            seed_deltas.append(delta)
        if seed == 42:
            point_scores = scores

    method_result["FIN"] = {
        "roc": round(float(roc_auc_score(test.label, point_scores["FIN"])), 4),
        "pr": round(float(average_precision_score(test.label, point_scores["FIN"])), 4),
    }
    method_result["FIN_NET4"] = {
        "roc": round(float(roc_auc_score(test.label, point_scores["FIN_NET4"])), 4),
        "pr": round(float(average_precision_score(test.label, point_scores["FIN_NET4"])), 4),
    }
    method_result["delta_roc_seed42"] = round(
        method_result["FIN_NET4"]["roc"] - method_result["FIN"]["roc"], 4
    )
    method_result["seed_delta_mean"] = round(float(np.mean(seed_deltas)), 4)
    method_result["seed_delta_sd"] = round(float(np.std(seed_deltas)), 4)
    method_result["cluster_bootstrap"] = cluster_interval(
        test, point_scores["FIN"], point_scores["FIN_NET4"]
    )
    result["methods"][method] = method_result

OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print(f"saved -> {OUT}")
