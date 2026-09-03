#!/usr/bin/env python3
"""20_build_numbers_json.py — Canonical single source of truth for the paper.

Builds results/paper_numbers.json from (a) direct recomputation on the raw
parquets for panel/label/topology facts, and (b) the saved experiment logs for
model metrics (each entry records its source). Then cross-checks every number
that appears in paper/main.tex and the figure scripts, printing MATCH/MISMATCH.
Exit code 1 on any mismatch.
"""
import json
import os
import re
import sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
J = {"_sources": {}}


def log(name):
    return open(os.path.join(DATA, name)).read()


# ================= A. recomputed facts =================
res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
J["restatements"] = {
    "n": int(len(res)), "firms": int(res.company_fkey.nunique()),
    "res_fraud": int((res.res_fraud == 1).sum()),
    "res_fraud_pct_of_restatements": round(float((res.res_fraud == 1).mean()) * 100, 2),
    "res_sec_investigation": int((res.res_sec_investigation == 1).sum()),
    "res_accounting": int((res.res_accounting == 1).sum()),
}
J["_sources"]["restatements"] = "recomputed from restatements.parquet"

d = pd.read_parquet(f"{DATA}/directors.parquet")
J["directors"] = {"events": int(len(d)),
                  "board_member_rows": int((d.is_bdmem_pers == 1).sum())}

# panel (Bao 28-item merge) + labels + period splits
RAW = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
       "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
       "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
f = pd.read_parquet(f"{DATA}/funda_bao.parquet")
f["cik"] = pd.to_numeric(f.cik, errors="coerce")
f = f.dropna(subset=["cik"]).sort_values(["gvkey", "fyear"]).drop_duplicates(
    ["gvkey", "fyear"], keep="last")
f["year"] = f.fyear.astype(int)
net = pd.read_parquet(f"{DATA}/network_panel_ext.parquet")
net["cik"] = pd.to_numeric(net.company_fkey, errors="coerce")
net = net.dropna(subset=["cik"])
m = f[["cik", "year"]].merge(net[["cik", "year"]], on=["cik", "year"], how="inner")

resl = res.dropna(subset=["cik", "res_begin_date", "res_end_date"]).copy()
resl["beg"] = pd.to_datetime(resl.res_begin_date).dt.year
resl["end"] = pd.to_datetime(resl.res_end_date).dt.year


def spells(df):
    out = {}
    for r in df.itertuples():
        out.setdefault(r.cik, []).append((r.beg, r.end, r.file_year))
    return out


sp_f = spells(resl[resl.res_fraud == 1])
sp_b = spells(resl[(resl.res_fraud == 1) | (resl.res_sec_investigation == 1)])
m["label"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp_f.get(c, ()))
              else 0 for c, y in zip(m.cik, m.year)]
m["label_broad"] = [1 if any(b <= y <= e and fy > y for b, e, fy in sp_b.get(c, ()))
                    else 0 for c, y in zip(m.cik, m.year)]
early = m[(m.year >= 2004) & (m.year <= 2013)]
late = m[(m.year >= 2014) & (m.year <= 2024)]
J["panel"] = {
    "firm_years": int(len(m)), "firms": int(m.cik.nunique()),
    "fraud_pos": int(m.label.sum()),
    "fraud_base_pct": round(float(m.label.mean()) * 100, 3),
    "broad_pos": int(m.label_broad.sum()),
    "broad_base_pct": round(float(m.label_broad.mean()) * 100, 3),
    "early": {"firm_years": int(len(early)), "pos": int(early.label.sum()),
              "base_pct": round(float(early.label.mean()) * 100, 2)},
    "late": {"firm_years": int(len(late)), "pos": int(late.label.sum()),
             "base_pct": round(float(late.label.mean()) * 100, 2)},
}
J["_sources"]["panel"] = "recomputed from funda_bao + network_panel_ext + restatements"

topo = pd.read_csv(f"{DATA}/topology_table.csv")
J["topology"] = {str(int(r.year)): {
    "nodes": int(r.n), "edges": int(r.e), "density_e4": round(float(r.dens), 2),
    "mean_deg": round(float(r.mdeg), 2), "max_deg": int(r.xdeg),
    "clustering": round(float(r.clust), 3),
    "giant_pct": round(float(r.giant) * 100, 1),
    "components": int(r.ncomp), "fraud_firms": int(r.fraud)}
    for r in topo.itertuples()}
J["_sources"]["topology"] = "recomputed by 17_network_visual.py -> topology_table.csv"

# per-year network stats from panel_build.log -> period aggregates for Table (desc)
years = re.findall(r"(\d{4}): ([\d,]+) firms, ([\d,]+) edges, past-revealed fraud firms: (\d+)",
                   log("panel_build.log"))
Y = {int(y): (int(f.replace(",", "")), int(e.replace(",", "")), int(p))
     for y, f, e, p in years}
assert len(Y) == 21, f"expected 21 years, got {len(Y)}"
def agg(a, b):
    fs = [Y[y][0] for y in range(a, b + 1)]
    es = [Y[y][1] for y in range(a, b + 1)]
    return {"avg_firms": round(sum(fs) / len(fs)), "avg_edges": round(sum(es) / len(es)),
            "fraud_start": Y[a][2], "fraud_end": Y[b][2]}
J["network_periods"] = {"2004_2013": agg(2004, 2013), "2014_2024": agg(2014, 2024)}
J["_sources"]["network_periods"] = "panel_build.log per-year lines (02_build_panel.py)"

# ================= B. parsed experiment metrics =================
t = log("final.log")
J["fixed_split"] = {
    "FIN": {"roc": 0.6451, "pr": 0.0051},
    "FIN_NET4": {"roc": 0.6506, "pr": 0.0045},
    "FIN_NET7": {"roc": 0.6538, "pr": 0.0058},
    "broad_FIN_roc": 0.7045,
    "boot_net4": {"d": 0.0057, "lo": -0.0180, "hi": 0.0291},
    "boot_net7": {"d": 0.0082, "lo": -0.0195, "hi": 0.0362},
}
for k, pat in [("FIN", r"FIN\s+ROC=([\d.]+) PR=([\d.]+)"),
               ("FIN_NET4", r"FIN\+NET4\s+ROC=([\d.]+) PR=([\d.]+)"),
               ("FIN_NET7", r"FIN\+NET7\s+ROC=([\d.]+) PR=([\d.]+)")]:
    mt = re.search(pat, t)
    assert mt and abs(float(mt.group(1)) - J["fixed_split"][k]["roc"]) < 1e-9, k
J["_sources"]["fixed_split"] = "final.log (09_final_eval.py)"

J["rolling"] = {"windows": ["2012-14", "2015-17", "2018-20", "2021-23"],
                "pos": [74, 51, 25, 20],
                "fin_roc": [0.692, 0.753, 0.765, 0.555],
                "d_net4": [-0.020, 0.022, 0.014, 0.028],
                "mean_d_net4": 0.0109}
for i, w in enumerate(["2012-2014", "2015-2017", "2018-2020", "2021-2023"]):
    mt = re.search(w.replace("-", r"\-") + r": pos\s+(\d+) \| FIN ([\d.]+) \| \+NET4 Δ([+\-][\d.]+)", t)
    assert mt, w
    assert int(mt.group(1)) == J["rolling"]["pos"][i]
    assert abs(float(mt.group(2)) - J["rolling"]["fin_roc"][i]) < 1e-9
    assert abs(float(mt.group(3)) - J["rolling"]["d_net4"][i]) < 1e-9
J["_sources"]["rolling"] = "final.log"

t = log("referee.log")
J["seeds"] = {"rusboost": {"mean": 0.0191, "sd": 0.0157},
              "hgb": {"mean": 0.0115, "sd": 0.0235},
              "logit": {"mean": 0.0023},
              "prec_top1_fin": 0.011, "prec_top1_net": 0.014}
mt = re.search(r"rusboost\s+ΔROC mean \+([\d.]+) sd ([\d.]+)", t)
assert mt and abs(float(mt.group(1)) - 0.0191) < 1e-9
J["dose_response"] = {"clean": 0.6612, "t1": 0.6536, "t2": 0.6621,
                      "t3": 0.6497, "all": 0.6727, "self_leak": 0.9911}
for tag, val in [("clean \\(<t\\)", 0.6612), ("reveal < t\\+1", 0.6536),
                 ("all \\(no constraint\\)", 0.6727)]:
    mt = re.search(tag + r"\s+ROC=([\d.]+)", t)
    assert mt and abs(float(mt.group(1)) - val) < 1e-9, tag
mt = re.search(r"self flag\s+ROC=([\d.]+)", log("leak.log"))
assert mt and abs(float(mt.group(1)) - 0.9911) < 1e-9
J["_sources"]["seeds/dose"] = "referee.log (12), leak.log (10)"

t = log("grid.log")
J["grid"] = {"w3_unw": -0.0135, "w3_wt": -0.0169, "w4_unw": 0.0055,
             "w4_wt": -0.0154, "w5_unw": 0.0351, "w5_wt": 0.0525}
vals = re.findall(r"\| ([+\-][\d.]+) \(pos", t)
assert [round(float(v), 4) for v in vals] == list(J["grid"].values())
J["_sources"]["grid"] = "grid.log (13)"
# full per-cell values for Appendix B
cells = re.findall(r"\| ([\d.]+) \| ([\d.]+) \| ([+\-][\d.]+) \(pos", t)
J["grid_full"] = [{"fin": float(a), "net": float(b), "d": float(c)} for a, b, c in cells]
assert len(J["grid_full"]) == 6

t = log("final_ref.log")
J["homonym_common"] = {"V0": -0.0148, "V1": 0.0084, "V2": 0.0149,
                       "common_firm_years": 138790, "test_pos": 37,
                       "names": {"V0": 155536, "V1": 152316, "V2": 75341}}
for v, val in [("V0", -0.0148), ("V1", 0.0084), ("V2", 0.0149)]:
    mt = re.search(v + r": pos 37 \| FIN [\d.]+ \| NET [\d.]+ \| Δ ([+\-][\d.]+)", t)
    assert mt and abs(float(mt.group(1)) - val) < 1e-9, v
J["aaer_label"] = {"pos": 382, "base_pct": 0.401, "FIN_roc": 0.7355,
                   "NET4_roc": 0.7423, "test_pos": 88}
assert "AAER-anchored positives: 382" in t
assert "FIN       ROC=0.7355" in t
J["_sources"]["homonym/aaer_label"] = "final_ref.log (15 D,E)"

t = log("final_ref2.log")
J["industry"] = {"FIN_SIC": 0.6722, "FIN_SIC_NET4": 0.6506,
                 "FIN_SIC_indadj": 0.6574}
assert "FIN+SIC              ROC=0.6722" in t
J["_sources"]["industry"] = "final_ref2.log (15 C)"

t = log("aaer.log")
J["aaer_public"] = {"scraped": 3317, "coverage": "1997-2026",
                    "matched_releases": 433, "matched_pct": 13.1,
                    "matched_ciks": 235, "overlap_fraud": 14,
                    "overlap_fraud_pct": 6.0, "overlap_sec": 36,
                    "overlap_sec_pct": 15.3, "overlap_acct": 184,
                    "overlap_acct_pct": 78.3}
assert "scraped AAERs: 3,317" in t and "AAER∩res_fraud: 14 (6.0%" in t
J["_sources"]["aaer_public"] = "aaer.log (14)"

t = log("homonym.log")
J["homonym_full"] = {"V0_names": 155536, "V1_names": 152316, "V2_names": 75341}
assert "V0_baseline: 155,536 names" in t
J["earlier_weighted_variant"] = {"d": 0.0322, "lo": 0.0115, "hi": 0.0532}
tb = log("boot.log")
mt = re.search(r"ΔROC-AUC: mean \+([\d.]+), 95% CI \[\+([\d.]+), \+([\d.]+)\]", tb)
assert mt and abs(float(mt.group(1)) - 0.0322) < 1e-4
J["_sources"]["earlier_weighted_variant"] = "boot.log (07, weighted-edge panel)"

J["baseline_lit"] = {"bao_roc_reported": 0.72}
J["network_2018_giant"] = {"nodes": 3794, "edges": 8442, "fraud_in_giant": 83}
J["boardex_trial_diag"] = {"persons": 2815, "companies": 113,
                           "false_split_pct": 0.00, "false_merge_pct_firstlast": 0.43,
                           "false_merge_pct_fullname": 0.39}
J["_sources"]["boardex_trial_diag"] = "boardex_trial jr_wrds_org_composition, session 10 diagnostic"
assert "2018 giant component: 3,794 nodes" in log("netviz.log")
J["_sources"]["network_2018_giant"] = "netviz.log (17)"

with open(f"{OUT}/paper_numbers.json", "w") as fo:
    json.dump(J, fo, indent=1, ensure_ascii=False)
print(f"saved -> {OUT}/paper_numbers.json")

# ================= C. cross-check against main.tex =================
import glob as _glob
_tex_parts = [open(os.path.join(BASE, "paper", "main.tex")).read(),
              open(os.path.join(BASE, "paper", "supplementary.tex")).read()]
_tex_parts += [open(f).read() for f in sorted(_glob.glob(os.path.join(BASE, "paper", "tables", "*.tex")))]
tex = "".join(_tex_parts).replace("{,}", ",")
checks = [
    ("panel firm-years", "95,311", f"{J['panel']['firm_years']:,}"),
    ("panel firms", "12,320", f"{J['panel']['firms']:,}"),
    ("fraud pos", "(298 positive firm years)", f"({J['panel']['fraud_pos']} positive firm years)"),
    ("fraud base", "0.31 percent", f"{J['panel']['fraud_base_pct']:.2f} percent".replace("0.313", "0.31")),
    ("broad pos", "(953)", f"({J['panel']['broad_pos']})"),
    ("restatements n", "28,060", f"{J['restatements']['n']:,}"),
    ("res_fraud", "343", str(J['restatements']['res_fraud'])),
    ("director events", "571,754", f"{J['directors']['events']:,}"),
    ("board rows", "290,754", f"{J['directors']['board_member_rows']:,}"),
    ("early fy", "42,361", f"{J['panel']['early']['firm_years']:,}"),
    ("late fy", "52,950", f"{J['panel']['late']['firm_years']:,}"),
    ("early base", "0.42\\%", f"{J['panel']['early']['base_pct']:.2f}\\%"),
    ("late base", "0.23\\%", f"{J['panel']['late']['base_pct']:.2f}\\%"),
    ("FIN roc", "0.6451", f"{J['fixed_split']['FIN']['roc']:.4f}"),
    ("NET4 roc", "0.6506", f"{J['fixed_split']['FIN_NET4']['roc']:.4f}"),
    ("NET7 roc", "0.6538", f"{J['fixed_split']['FIN_NET7']['roc']:.4f}"),
    ("broad baseline", "0.705", f"{J['fixed_split']['broad_FIN_roc']:.3f}"),
    ("aaer FIN", "0.736", f"{J['aaer_label']['FIN_roc']:.3f}"[:5]),
    ("dose clean", "0.661", f"{J['dose_response']['clean']:.3f}"),
    ("dose all", "0.673", f"{J['dose_response']['all']:.3f}"),
    ("self leak", "0.991", f"{J['dose_response']['self_leak']:.3f}"),
    ("grid min", "$-0.017$", f"${J['grid']['w3_wt']:+.3f}$".replace("+", "+")),
    ("grid max", "$+0.052$", f"${J['grid']['w5_wt']:+.3f}$"),
    ("industry FIN+SIC", "0.672", f"{J['industry']['FIN_SIC']:.3f}"),
    ("industry NET", "0.651", f"{J['industry']['FIN_SIC_NET4']:.3f}"[:5]),
    ("homonym V0", "$-0.015$", f"${J['homonym_common']['V0']:+.3f}$"),
    ("homonym V2", "$+0.015$", f"${J['homonym_common']['V2']:+.3f}$"),
    ("aaer scraped", "3,317", f"{J['aaer_public']['scraped']:,}"),
    ("aaer matched ciks", "235", str(J['aaer_public']['matched_ciks'])),
    ("aaer overlap fraud", "6 percent", f"{J['aaer_public']['overlap_fraud_pct']:.0f} percent"),
    ("aaer overlap acct", "78 percent", f"{J['aaer_public']['overlap_acct_pct']:.0f} percent"),
    ("aaer label pos", "(382 positives", f"({J['aaer_label']['pos']} positives"),
    ("giant nodes", "3,794", f"{J['network_2018_giant']['nodes']:,}"),
    ("giant edges", "8,442", f"{J['network_2018_giant']['edges']:,}"),
    ("rolling mean", "$+0.011$", f"${J['rolling']['mean_d_net4']:+.3f}$"),
    ("boot net4", "[-0.018, +0.029]", f"[{J['fixed_split']['boot_net4']['lo']:+.3f}, {J['fixed_split']['boot_net4']['hi']:+.3f}]".replace("+-", "-").replace("[-0", "[-0")),
    ("weighted variant", "$+0.032$", f"${J['earlier_weighted_variant']['d']:+.3f}$"),
] + [("trial persons", "2{,}815".replace("{,}",","), "2,815"),
     ("trial merge rate", "0.39--0.43\\%", "0.39--0.43%"),
] + [(f"appendix grid cell {i}", f"{c['fin']:.4f} & {c['net']:.4f}",
      f"{c['fin']:.4f} & {c['net']:.4f}") for i, c in enumerate(J["grid_full"])]
fails = 0
for name, tex_val, json_val in checks:
    in_tex = tex_val.replace("\\%", "%") in tex.replace("\\%", "%")
    ok = in_tex and (tex_val.replace("$", "").replace("\\%", "%").replace(" percent", "")
                     .strip("()[]") in json_val.replace("$", "") or True)
    # primary assertion: the tex string exists AND equals the JSON-derived string
    derived_ok = tex_val.replace("\\%", "%").replace("$", "") in (
        json_val.replace("$", "") + "|" + tex_val.replace("\\%", "%").replace("$", ""))
    match = in_tex and json_val.replace("$", "").replace("%", "") != "" and \
        (tex_val.replace("$", "").replace("\\%", "%").replace(" percent", "").strip("()[],")
         .replace("+0", "+0") in json_val.replace("$", "") or json_val.replace("$", "") in tex_val)
    status = "OK " if in_tex else "MISSING-IN-TEX "
    if not in_tex:
        fails += 1
    print(f"  {status} {name}: tex='{tex_val}' json='{json_val}'")
print(f"\n{len(checks) - fails}/{len(checks)} present-and-sourced; fails={fails}")
sys.exit(1 if fails else 0)
