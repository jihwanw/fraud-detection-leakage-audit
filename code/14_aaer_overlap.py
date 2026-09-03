#!/usr/bin/env python3
"""14_aaer_overlap.py — Validate res_fraud against the public SEC AAER list.

1. Scrape the SEC AAER listing (all pages) -> respondent text, date, AAER no.
2. Map company-like respondent names to CIK via SEC company_tickers.json
   (normalized name containment match; conservative).
3. Report: (a) AAER firm-CIKs found, (b) overlap with res_fraud CIKs,
   (c) overlap with broader restatement flags — the label-validity check
   requested by domain reviewers.
Honest caveats printed: name matching is conservative (undercounts), and the
Drupal listing covers recent releases only (coverage window reported).
"""
import re
import json
import time
import html
import urllib.request
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
UA = {"User-Agent": "Jihwan Woo academic research jihwan.woo@gmail.com"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")


rows = []
for page in range(0, 34):
    url = ("https://www.sec.gov/enforcement-litigation/"
           f"accounting-auditing-enforcement-releases?page={page}")
    try:
        t = get(url)
    except Exception as e:
        print(f"page {page}: fetch failed {e}")
        continue
    for m in re.finditer(
            r"<time datetime=\"(\d{4})-\d{2}-\d{2}T[^\"]*\"[^>]*>.*?"
            r"respondents'><a[^>]*>(.*?)</a>.*?"
            r"subfield_value\">(.*?)</span>", t, re.S):
        year, resp, relno = m.group(1), html.unescape(m.group(2)), m.group(3)
        aaer = re.search(r"AAER-(\d+)", relno)
        rows.append({"year": int(year), "respondent": resp.strip(),
                     "aaer": int(aaer.group(1)) if aaer else None})
    time.sleep(0.4)

df = pd.DataFrame(rows).dropna(subset=["aaer"]).drop_duplicates("aaer")
print(f"scraped AAERs: {len(df):,} | coverage {df.year.min()}-{df.year.max()} "
      f"| AAER no. {df.aaer.min():.0f}-{df.aaer.max():.0f}")
df.to_parquet(f"{DATA}/aaer_public.parquet")

# ---- name -> CIK map ----
tick = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
comp = pd.DataFrame(tick).T
comp["norm"] = (comp.title.str.upper()
                .str.replace(r"[^A-Z0-9 ]", "", regex=True)
                .str.replace(r"\b(INC|CORP|CO|LTD|LLC|PLC|THE|COMPANY|"
                             r"CORPORATION|INCORPORATED|HOLDINGS?|GROUP)\b",
                             "", regex=True).str.strip())
comp = comp[comp.norm.str.len() >= 5]
name2cik = dict(zip(comp.norm, comp.cik_str.astype(int)))

def norm(s):
    s = re.sub(r"\(.*?\)", "", s.upper())
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\b(INC|CORP|CO|LTD|LLC|PLC|THE|COMPANY|CORPORATION|"
               r"INCORPORATED|HOLDINGS?|GROUP)\b", "", s)
    return s.strip()

hits = {}
for r in df.itertuples():
    n = norm(r.respondent)
    for cand, cik in name2cik.items():
        if cand and (cand in n):
            hits.setdefault(r.aaer, set()).add(cik)
aaer_ciks = set().union(*hits.values()) if hits else set()
print(f"AAERs matched to at least one public-company CIK: {len(hits):,} "
      f"({len(hits) / len(df):.1%}) | distinct CIKs {len(aaer_ciks):,}")

res = pd.read_parquet(f"{DATA}/restatements.parquet")
res["cik"] = pd.to_numeric(res.company_fkey, errors="coerce")
res["file_year"] = pd.to_datetime(res.file_date).dt.year
lo, hi = df.year.min(), df.year.max()
win = res[(res.file_year >= lo - 2) & (res.file_year <= hi)]

fraud_ciks = set(win[win.res_fraud == 1].cik.dropna().astype(int))
sec_ciks = set(win[win.res_sec_investigation == 1].cik.dropna().astype(int))
acc_ciks = set(win[win.res_accounting == 1].cik.dropna().astype(int))
print(f"\nrestatement-flag CIKs in window {lo - 2}-{hi}: "
      f"fraud {len(fraud_ciks)}, sec_invest {len(sec_ciks)}, acct {len(acc_ciks)}")
print(f"AAER∩res_fraud: {len(aaer_ciks & fraud_ciks)} "
      f"({len(aaer_ciks & fraud_ciks) / max(len(aaer_ciks), 1):.1%} of AAER CIKs)")
print(f"AAER∩res_sec_investigation: {len(aaer_ciks & sec_ciks)} "
      f"({len(aaer_ciks & sec_ciks) / max(len(aaer_ciks), 1):.1%})")
print(f"AAER∩any accounting restatement: {len(aaer_ciks & acc_ciks)} "
      f"({len(aaer_ciks & acc_ciks) / max(len(aaer_ciks), 1):.1%})")
print(f"res_fraud NOT in AAER: {len(fraud_ciks - aaer_ciks)} "
      f"(expected: AAER list here covers {lo}-{hi} only + individuals excluded)")
