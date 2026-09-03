#!/usr/bin/env python3
"""20_build_numbers_json.py — Canonical single source of truth for the paper.

Every empirical value is either recomputed from licensed source parquets or
parsed from an execution log / aggregate diagnostic JSON. No result statistic
is entered as a literal. The script then verifies that every manuscript result
string is generated from the canonical JSON and appears in the TeX sources.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
LOGS = BASE / "logs"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)
J = {"_sources": {}}


def read_log(name: str) -> str:
    candidates = (LOGS / name, DATA / name)
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Required execution log missing from logs/ and data/: {name}"
        )
    return path.read_text(errors="replace")


def read_optional_log(name: str) -> str:
    for candidate in (LOGS / name, DATA / name):
        if candidate.exists():
            return candidate.read_text(errors="replace")
    return ""


def match(pattern: str, text: str, label: str, flags: int = 0):
    found = re.search(pattern, text, flags)
    if not found:
        raise AssertionError(f"Could not parse {label}: {pattern}")
    return found


def fnum(value: str) -> float:
    return float(value.replace(",", ""))


# ================= A. facts recomputed from source parquets =================
res = pd.read_parquet(DATA / "restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
J["restatements"] = {
    "n": int(len(res)),
    "firms": int(res.company_fkey.nunique()),
    "res_fraud": int((res.res_fraud == 1).sum()),
    "res_fraud_pct_of_restatements": round(float((res.res_fraud == 1).mean()) * 100, 2),
    "res_sec_investigation": int((res.res_sec_investigation == 1).sum()),
    "res_accounting": int((res.res_accounting == 1).sum()),
}
J["_sources"]["restatements"] = "recomputed from data/restatements.parquet"

directors = pd.read_parquet(DATA / "directors.parquet")
J["directors"] = {
    "events": int(len(directors)),
    "board_member_rows": int((directors.is_bdmem_pers == 1).sum()),
}
J["_sources"]["directors"] = "recomputed from data/directors.parquet"

fund = pd.read_parquet(DATA / "funda_bao.parquet")
fund["cik"] = pd.to_numeric(fund.cik, errors="coerce")
fund = fund.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"]).drop_duplicates(
    ["gvkey", "fyear"], keep="last"
)
fund["year"] = fund.fyear.astype(int)
network = pd.read_parquet(DATA / "network_panel_ext.parquet")
network["cik"] = pd.to_numeric(network.company_fkey, errors="coerce")
network = network.dropna(subset=["cik"])
panel = fund[["cik", "year"]].merge(
    network[["cik", "year"]], on=["cik", "year"], how="inner"
)

res_labeled = res.dropna(subset=["cik", "res_begin_date", "res_end_date"]).copy()
res_labeled["beg"] = pd.to_datetime(res_labeled.res_begin_date).dt.year
res_labeled["end"] = pd.to_datetime(res_labeled.res_end_date).dt.year


def spells(frame: pd.DataFrame) -> dict:
    output = {}
    for row in frame.itertuples():
        output.setdefault(row.cik, []).append((row.beg, row.end, row.file_year))
    return output


fraud_spells = spells(res_labeled[res_labeled.res_fraud == 1])
broad_spells = spells(
    res_labeled[
        (res_labeled.res_fraud == 1) | (res_labeled.res_sec_investigation == 1)
    ]
)
panel["label"] = [
    int(any(b <= year <= e and filed > year for b, e, filed in fraud_spells.get(cik, ())))
    for cik, year in zip(panel.cik, panel.year)
]
panel["label_broad"] = [
    int(any(b <= year <= e and filed > year for b, e, filed in broad_spells.get(cik, ())))
    for cik, year in zip(panel.cik, panel.year)
]
early = panel[(panel.year >= 2004) & (panel.year <= 2013)]
late = panel[(panel.year >= 2014) & (panel.year <= 2024)]
J["panel"] = {
    "firm_years": int(len(panel)),
    "firms": int(panel.cik.nunique()),
    "fraud_pos": int(panel.label.sum()),
    "fraud_base_pct": round(float(panel.label.mean()) * 100, 3),
    "broad_pos": int(panel.label_broad.sum()),
    "broad_base_pct": round(float(panel.label_broad.mean()) * 100, 3),
    "early": {
        "firm_years": int(len(early)),
        "pos": int(early.label.sum()),
        "base_pct": round(float(early.label.mean()) * 100, 2),
    },
    "late": {
        "firm_years": int(len(late)),
        "pos": int(late.label.sum()),
        "base_pct": round(float(late.label.mean()) * 100, 2),
    },
}
J["_sources"]["panel"] = (
    "recomputed from funda_bao.parquet + network_panel_ext.parquet + "
    "restatements.parquet"
)

topology = pd.read_csv(DATA / "topology_table.csv")
J["topology"] = {
    str(int(row.year)): {
        "nodes": int(row.n),
        "edges": int(row.e),
        "density_e4": round(float(row.dens), 2),
        "mean_deg": round(float(row.mdeg), 2),
        "max_deg": int(row.xdeg),
        "clustering": round(float(row.clust), 3),
        "giant_pct": round(float(row.giant) * 100, 1),
        "components": int(row.ncomp),
        "fraud_firms": int(row.fraud),
    }
    for row in topology.itertuples()
}
J["_sources"]["topology"] = "data/topology_table.csv generated by code/17"

panel_log = read_log("panel_build.log")
year_rows = re.findall(
    r"(\d{4}): ([\d,]+) firms, ([\d,]+) edges, past-revealed fraud firms: (\d+)",
    panel_log,
)
year_stats = {
    int(year): (int(firms.replace(",", "")), int(edges.replace(",", "")), int(fraud))
    for year, firms, edges, fraud in year_rows
}
if len(year_stats) != 21:
    raise AssertionError(f"Expected 21 network years, found {len(year_stats)}")


def period_aggregate(start: int, end: int) -> dict:
    firms = [year_stats[year][0] for year in range(start, end + 1)]
    edges = [year_stats[year][1] for year in range(start, end + 1)]
    return {
        "avg_firms": round(sum(firms) / len(firms)),
        "avg_edges": round(sum(edges) / len(edges)),
        "fraud_start": year_stats[start][2],
        "fraud_end": year_stats[end][2],
    }


J["network_periods"] = {
    "2004_2013": period_aggregate(2004, 2013),
    "2014_2024": period_aggregate(2014, 2024),
}
J["_sources"]["network_periods"] = "data/panel_build.log generated by code/02"

# ================= B. metrics parsed from execution artifacts =================
final_log = read_log("final.log")
fraud_block = match(
    r"===== FIXED SPLIT \| label \|.*?(?====== FIXED SPLIT \| label_broad)",
    final_log,
    "fraud fixed-split block",
    re.S,
).group(0)
broad_block = match(
    r"===== FIXED SPLIT \| label_broad \|.*?(?====== ROLLING ORIGIN)",
    final_log,
    "broad fixed-split block",
    re.S,
).group(0)


def performance(block: str, feature_name: str) -> dict:
    parsed = match(
        rf"^\s*{re.escape(feature_name)}\s+ROC=([\d.]+) PR=([\d.]+)",
        block,
        f"{feature_name} performance",
        re.M,
    )
    return {"roc": fnum(parsed.group(1)), "pr": fnum(parsed.group(2))}


def bootstrap(block: str, feature_name: str) -> dict:
    parsed = match(
        rf"(?:cluster )?Δ\({re.escape(feature_name)}-FIN\) ROC: ([+\-][\d.]+) "
        rf"\[([+\-][\d.]+), ([+\-][\d.]+)\] P\(>0\)=([\d.]+)",
        block,
        f"{feature_name} bootstrap",
    )
    return {
        "d": fnum(parsed.group(1)),
        "lo": fnum(parsed.group(2)),
        "hi": fnum(parsed.group(3)),
        "p_gt_zero": fnum(parsed.group(4)),
    }


J["fixed_split"] = {
    "FIN": performance(fraud_block, "FIN"),
    "FIN_NET4": performance(fraud_block, "FIN+NET4"),
    "FIN_NET7": performance(fraud_block, "FIN+NET7"),
    "broad_FIN": performance(broad_block, "FIN"),
    "boot_net4": bootstrap(fraud_block, "FIN+NET4"),
    "boot_net7": bootstrap(fraud_block, "FIN+NET7"),
}
J["_sources"]["fixed_split"] = "data/final.log generated by code/09"

rolling_rows = re.findall(
    r"(\d{4})-(\d{4}): pos\s+(\d+) \| FIN ([\d.]+) \| "
    r"\+NET4 Δ([+\-][\d.]+) \| \+NET7 Δ([+\-][\d.]+)",
    final_log,
)
if len(rolling_rows) != 4:
    raise AssertionError(f"Expected four rolling windows, found {len(rolling_rows)}")
rolling_mean = match(
    r"mean ΔROC: NET4 ([+\-][\d.]+) \| NET7 ([+\-][\d.]+)",
    final_log,
    "rolling means",
)
J["rolling"] = {
    "windows": [f"{start[-2:]}-{end[-2:]}" for start, end, *_ in rolling_rows],
    "pos": [int(row[2]) for row in rolling_rows],
    "fin_roc": [fnum(row[3]) for row in rolling_rows],
    "d_net4": [fnum(row[4]) for row in rolling_rows],
    "d_net7": [fnum(row[5]) for row in rolling_rows],
    "mean_d_net4": fnum(rolling_mean.group(1)),
    "mean_d_net7": fnum(rolling_mean.group(2)),
}
J["_sources"]["rolling"] = "data/final.log generated by code/09"

referee_log = read_log("referee.log")
seed_results = {}
for model in ("rusboost", "hgb", "logit"):
    parsed = match(
        rf"{model}\s+ΔROC mean ([+\-][\d.]+) sd ([\d.]+) min ([+\-][\d.]+) "
        rf"max ([+\-][\d.]+) \(n=(\d+)\) \| prec@1% FIN ([\d.]+) NET ([\d.]+)",
        referee_log,
        f"{model} seed results",
    )
    seed_results[model] = {
        "mean": fnum(parsed.group(1)),
        "sd": fnum(parsed.group(2)),
        "min": fnum(parsed.group(3)),
        "max": fnum(parsed.group(4)),
        "n": int(parsed.group(5)),
        "prec_top1_fin": fnum(parsed.group(6)),
        "prec_top1_net": fnum(parsed.group(7)),
    }
J["seeds"] = seed_results

dose_patterns = {
    "clean": r"clean \(<t\)\s+ROC=([\d.]+)",
    "t1": r"reveal < t\+1\s+ROC=([\d.]+)",
    "t2": r"reveal < t\+2\s+ROC=([\d.]+)",
    "t3": r"reveal < t\+3\s+ROC=([\d.]+)",
    "all": r"all \(no constraint\)\s+ROC=([\d.]+)",
}
J["dose_response"] = {
    name: fnum(match(pattern, referee_log, f"dose {name}").group(1))
    for name, pattern in dose_patterns.items()
}
leak_log = read_log("leak.log")
J["dose_response"]["self_leak"] = fnum(
    match(r"self flag\s+ROC=([\d.]+)", leak_log, "self leakage").group(1)
)
J["_sources"]["seeds/dose"] = "data/referee.log (code/12) + data/leak.log (code/10)"

grid_log = read_log("grid.log")
grid_rows = re.findall(
    r"^\s*(\d)\s+\|\s+(unweighted|weighted)\s+\|\s+([\d.]+)\s+\|\s+"
    r"([\d.]+)\s+\|\s+([+\-][\d.]+) \(pos (\d+)\)",
    grid_log,
    re.M,
)
if len(grid_rows) != 6:
    raise AssertionError(f"Expected six construction-grid cells, found {len(grid_rows)}")
J["grid"] = {}
J["grid_full"] = []
for window, weighting, fin_roc, net_roc, delta, positives in grid_rows:
    suffix = "unw" if weighting == "unweighted" else "wt"
    J["grid"][f"w{window}_{suffix}"] = fnum(delta)
    J["grid_full"].append(
        {
            "window": int(window),
            "weighting": weighting,
            "fin": fnum(fin_roc),
            "net": fnum(net_roc),
            "d": fnum(delta),
            "positives": int(positives),
        }
    )
J["_sources"]["grid"] = "data/grid.log generated by code/13"

final_ref_log = read_log("final_ref.log") + "\n" + read_optional_log("final_ref2.log")
common = match(r"common firm-years: ([\d,]+)", final_ref_log, "homonym common sample")
homonym = {
    "common_firm_years": int(common.group(1).replace(",", "")),
    "names": {},
}
for variant in ("V0", "V1", "V2"):
    parsed = match(
        rf"{variant}: pos (\d+) \| FIN ([\d.]+) \| NET ([\d.]+) \| Δ ([+\-][\d.]+)",
        final_ref_log,
        f"homonym {variant}",
    )
    homonym[variant] = {
        "test_pos": int(parsed.group(1)),
        "fin": fnum(parsed.group(2)),
        "net": fnum(parsed.group(3)),
        "d": fnum(parsed.group(4)),
    }
homonym_log = read_log("homonym.log")
for variant, log_name in (("V0", "V0_baseline"), ("V1", "V1_drop5plus"), ("V2", "V2_middlereq")):
    parsed = match(rf"{log_name}: ([\d,]+) names", homonym_log, f"{variant} name count")
    homonym["names"][variant] = int(parsed.group(1).replace(",", ""))
J["homonym_common"] = homonym

aaer_label = match(
    r"AAER-anchored positives: ([\d,]+) \(([\d.]+)%\).*?"
    r"FIN\s+ROC=([\d.]+) \(test pos (\d+)\).*?"
    r"FIN\+NET4\s+ROC=([\d.]+) \(test pos (\d+)\)",
    final_ref_log,
    "AAER-anchored results",
    re.S,
)
if aaer_label.group(4) != aaer_label.group(6):
    raise AssertionError("AAER test-positive counts differ between models")
J["aaer_label"] = {
    "pos": int(aaer_label.group(1).replace(",", "")),
    "base_pct": fnum(aaer_label.group(2)),
    "FIN_roc": fnum(aaer_label.group(3)),
    "test_pos": int(aaer_label.group(4)),
    "NET4_roc": fnum(aaer_label.group(5)),
}
J["_sources"]["homonym/aaer_label"] = "data/final_ref.log + data/homonym.log"

industry_log = final_ref_log
J["industry"] = {}
for key, label in (
    ("FIN_SIC", "FIN+SIC"),
    ("FIN_SIC_NET4", "FIN+SIC+NET4"),
    ("FIN_SIC_indadj", "FIN+SIC+indadj-fnr"),
):
    J["industry"][key] = fnum(
        match(rf"{re.escape(label)}\s+ROC=([\d.]+)", industry_log, label).group(1)
    )
J["_sources"]["industry"] = "data/final_ref2.log generated by code/15"

aaer_log = read_log("aaer.log")
aaer_head = match(
    r"scraped AAERs: ([\d,]+) \| coverage ([\d-]+).*?"
    r"matched to at least one public-company CIK: ([\d,]+) \(([\d.]+)%\) "
    r"\| distinct CIKs ([\d,]+)",
    aaer_log,
    "AAER coverage",
    re.S,
)
fraud_overlap = match(
    r"AAER∩res_fraud: ([\d,]+) \(([\d.]+)%", aaer_log, "AAER fraud overlap"
)
sec_overlap = match(
    r"AAER∩res_sec_investigation: ([\d,]+) \(([\d.]+)%",
    aaer_log,
    "AAER SEC overlap",
)
acct_overlap = match(
    r"AAER∩any accounting restatement: ([\d,]+) \(([\d.]+)%",
    aaer_log,
    "AAER accounting overlap",
)
J["aaer_public"] = {
    "scraped": int(aaer_head.group(1).replace(",", "")),
    "coverage": aaer_head.group(2),
    "matched_releases": int(aaer_head.group(3).replace(",", "")),
    "matched_pct": fnum(aaer_head.group(4)),
    "matched_ciks": int(aaer_head.group(5).replace(",", "")),
    "overlap_fraud": int(fraud_overlap.group(1).replace(",", "")),
    "overlap_fraud_pct": fnum(fraud_overlap.group(2)),
    "overlap_sec": int(sec_overlap.group(1).replace(",", "")),
    "overlap_sec_pct": fnum(sec_overlap.group(2)),
    "overlap_acct": int(acct_overlap.group(1).replace(",", "")),
    "overlap_acct_pct": fnum(acct_overlap.group(2)),
}
J["_sources"]["aaer_public"] = "data/aaer.log generated by code/14"

boot_log = read_log("boot.log")
weighted = match(
    r"ΔROC-AUC: mean ([+\-][\d.]+), 95% CI \[([+\-][\d.]+), ([+\-][\d.]+)\]",
    boot_log,
    "earlier weighted variant",
)
J["earlier_weighted_variant"] = {
    "d": fnum(weighted.group(1)),
    "lo": fnum(weighted.group(2)),
    "hi": fnum(weighted.group(3)),
}
J["_sources"]["earlier_weighted_variant"] = "data/boot.log generated by code/07"

netviz_log = read_log("netviz.log")
giant = match(
    r"2018 giant component: ([\d,]+) nodes, ([\d,]+) edges, fraud in giant: ([\d,]+)",
    netviz_log,
    "2018 giant component",
)
J["network_2018_giant"] = {
    "nodes": int(giant.group(1).replace(",", "")),
    "edges": int(giant.group(2).replace(",", "")),
    "fraud_in_giant": int(giant.group(3).replace(",", "")),
}
J["_sources"]["network_2018_giant"] = "data/netviz.log generated by code/17"

missingness_path = OUT / "missingness_sensitivity.json"
if not missingness_path.exists():
    raise FileNotFoundError(
        "Missingness sensitivity results missing; run code/19_missingness_sensitivity.py"
    )
J["missingness"] = json.loads(missingness_path.read_text())
J["_sources"]["missingness"] = (
    "results/missingness_sensitivity.json generated by code/19"
)

boardex_path = DATA / "boardex_trial_diag.json"
if not boardex_path.exists():
    raise FileNotFoundError(
        "BoardEx aggregate diagnostic missing; run code/18_boardex_trial_diagnostic.py"
    )
boardex = json.loads(boardex_path.read_text())
J["boardex_trial_diag"] = {
    "rows": int(boardex["rows"]),
    "persons": int(boardex["persons"]),
    "companies": int(boardex["companies"]),
    "false_split_pct": float(boardex["full_name"]["false_split_pct"]),
    "false_merge_pct_firstlast": float(boardex["first_last"]["false_merge_pct"]),
    "false_merge_pct_fullname": float(boardex["full_name"]["false_merge_pct"]),
}
J["_sources"]["boardex_trial_diag"] = (
    "data/boardex_trial_diag.json generated from WRDS BoardEx trial by code/18"
)

(OUT / "paper_numbers.json").write_text(
    json.dumps(J, indent=1, ensure_ascii=False), encoding="utf-8"
)
print(f"saved -> {OUT / 'paper_numbers.json'}")

# ================= C. JSON-derived manuscript checks =================
# The under-review TeX sources are not distributed in the public repository.
# In that environment, JSON reconstruction completes above and the archived
# internal 50/50 audit report documents the manuscript cross-check.
if not (BASE / "paper" / "main.tex").exists():
    print("Manuscript TeX not present: skipping internal manuscript cross-check.")
    raise SystemExit(0)

tex_files = [BASE / "paper" / "main.tex", BASE / "paper" / "supplementary.tex"]
tex_files.extend(Path(path) for path in sorted(glob.glob(str(BASE / "paper" / "tables" / "*.tex"))))
tex = "".join(path.read_text() for path in tex_files).replace("{,}", ",")

checks = [
    ("panel firm-years", f"{J['panel']['firm_years']:,}"),
    ("panel firms", f"{J['panel']['firms']:,}"),
    ("fraud positives", f"({J['panel']['fraud_pos']} positive firm years)"),
    ("fraud base rate", f"{J['panel']['fraud_base_pct']:.2f} percent"),
    ("broad positives", f"({J['panel']['broad_pos']})"),
    ("restatement records", f"{J['restatements']['n']:,}"),
    ("fraud restatements", str(J['restatements']['res_fraud'])),
    ("director events", f"{J['directors']['events']:,}"),
    ("board-member rows", f"{J['directors']['board_member_rows']:,}"),
    ("early firm-years", f"{J['panel']['early']['firm_years']:,}"),
    ("late firm-years", f"{J['panel']['late']['firm_years']:,}"),
    ("early base rate", f"{J['panel']['early']['base_pct']:.2f}\\%"),
    ("late base rate", f"{J['panel']['late']['base_pct']:.2f}\\%"),
    ("FIN ROC", f"{J['fixed_split']['FIN']['roc']:.4f}"),
    ("NET4 ROC", f"{J['fixed_split']['FIN_NET4']['roc']:.4f}"),
    ("NET7 ROC", f"{J['fixed_split']['FIN_NET7']['roc']:.4f}"),
    ("broad baseline", f"{J['fixed_split']['broad_FIN']['roc']:.3f}"),
    ("AAER baseline", f"{J['aaer_label']['FIN_roc']:.3f}"),
    ("clean dose", f"{J['dose_response']['clean']:.3f}"),
    ("all-history dose", f"{J['dose_response']['all']:.3f}"),
    ("self leakage", f"{J['dose_response']['self_leak']:.3f}"),
    ("grid minimum", f"${J['grid']['w3_wt']:+.3f}$"),
    ("grid maximum", f"${J['grid']['w5_wt']:+.3f}$"),
    ("industry baseline", f"{J['industry']['FIN_SIC']:.3f}"),
    ("industry network", f"{J['industry']['FIN_SIC_NET4']:.3f}"),
    ("homonym V0", f"${J['homonym_common']['V0']['d']:+.3f}$"),
    ("homonym V2", f"${J['homonym_common']['V2']['d']:+.3f}$"),
    ("AAER scraped", f"{J['aaer_public']['scraped']:,}"),
    ("AAER matched CIKs", f"{J['aaer_public']['matched_ciks']:,}"),
    ("AAER fraud overlap", f"{J['aaer_public']['overlap_fraud_pct']:.0f} percent"),
    ("AAER accounting overlap", f"{J['aaer_public']['overlap_acct_pct']:.0f} percent"),
    ("AAER label positives", f"({J['aaer_label']['pos']} positives"),
    ("giant-component nodes", f"{J['network_2018_giant']['nodes']:,}"),
    ("giant-component edges", f"{J['network_2018_giant']['edges']:,}"),
    ("rolling mean", f"${J['rolling']['mean_d_net4']:+.3f}$"),
    (
        "NET4 bootstrap interval",
        f"[{J['fixed_split']['boot_net4']['lo']:+.3f}, "
        f"{J['fixed_split']['boot_net4']['hi']:+.3f}]",
    ),
    ("weighted variant", f"${J['earlier_weighted_variant']['d']:+.3f}$"),
    ("trial persons", f"{J['boardex_trial_diag']['persons']:,}"),
    (
        "trial merge range",
        f"{J['boardex_trial_diag']['false_merge_pct_fullname']:.2f}--"
        f"{J['boardex_trial_diag']['false_merge_pct_firstlast']:.2f}\\%",
    ),
    (
        "NET7 bootstrap interval",
        f"[{J['fixed_split']['boot_net7']['lo']:+.3f}, "
        f"{J['fixed_split']['boot_net7']['hi']:+.3f}]",
    ),
    (
        "missingness zero row",
        f"Zero fill (primary; Bao et al.) & "
        f"{J['missingness']['methods']['zero']['test_rows']:,} & "
        f"{J['missingness']['methods']['zero']['test_positives']} & "
        f"{J['missingness']['methods']['zero']['FIN']['roc']:.4f} & "
        f"{J['missingness']['methods']['zero']['FIN_NET4']['roc']:.4f}",
    ),
    (
        "missingness median row",
        f"Train-median imputation & "
        f"{J['missingness']['methods']['train_median']['test_rows']:,} & "
        f"{J['missingness']['methods']['train_median']['test_positives']} & "
        f"{J['missingness']['methods']['train_median']['FIN']['roc']:.4f} & "
        f"{J['missingness']['methods']['train_median']['FIN_NET4']['roc']:.4f}",
    ),
    (
        "missingness indicator row",
        f"Train median + missingness indicators & "
        f"{J['missingness']['methods']['median_plus_indicators']['test_rows']:,} & "
        f"{J['missingness']['methods']['median_plus_indicators']['test_positives']} & "
        f"{J['missingness']['methods']['median_plus_indicators']['FIN']['roc']:.4f} & "
        f"{J['missingness']['methods']['median_plus_indicators']['FIN_NET4']['roc']:.4f}",
    ),
    (
        "missingness complete-case row",
        f"Complete cases & "
        f"{J['missingness']['methods']['complete_case']['test_rows']:,} & "
        f"{J['missingness']['methods']['complete_case']['test_positives']} & "
        f"{J['missingness']['methods']['complete_case']['FIN']['roc']:.4f} & "
        f"{J['missingness']['methods']['complete_case']['FIN_NET4']['roc']:.4f}",
    ),
]
checks.extend(
    (
        f"construction-grid cell {index + 1}",
        f"{cell['fin']:.4f} & {cell['net']:.4f}",
    )
    for index, cell in enumerate(J["grid_full"])
)

missing = []
normalized_tex = tex.replace("\\%", "%")
for name, expected in checks:
    present = expected.replace("\\%", "%") in normalized_tex
    print(f"  {'OK' if present else 'MISSING'} {name}: {expected}")
    if not present:
        missing.append((name, expected))
print(f"\n{len(checks) - len(missing)}/{len(checks)} JSON-derived checks present")
if missing:
    raise SystemExit(f"Manuscript numeric audit failed: {missing}")
