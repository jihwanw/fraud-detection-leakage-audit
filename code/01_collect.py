#!/usr/bin/env python3
"""01_collect.py — Fraud detection rebuild: real data collection from WRDS.

Collects three raw inputs (all saved to data/):
  1. restatements.parquet   — audit.feed39_financial_restatements
                              (labels kept SEPARATE: res_fraud, res_sec_investigation,
                               res_adverse; primary label = res_fraud only)
  2. directors.parquet      — audit.feed17_director_and_officer_chan
                              (board membership spells for interlock network)
  3. funda.parquet          — comp.funda annual fundamentals
                              (real Beneish M-Score components)

Honesty rules:
  - No synthetic values anywhere.
  - Label definitions documented in data/collection_log.txt with base rates.
"""
import os
import sys
import builtins
import getpass
import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

from wrds_auth import connect

log = open(os.path.join(DATA, "collection_log.txt"), "w")


def say(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


say(f"collection started {datetime.datetime.now()}")
db = connect()

# ---------- 1. Restatements with honest, separate labels ----------
if os.path.exists(f"{DATA}/restatements.parquet"):
    say("[1/3] restatements already collected — skip")
else:
    say("[1/3] restatements (audit.feed39)...")
    res = db.raw_sql("""
    SELECT company_fkey, file_date,
           res_begin_date, res_end_date,
           res_accounting, res_fraud, res_adverse, res_sec_investigation,
           res_clerical_errors, res_improves
    FROM audit.feed39_financial_restatements
    WHERE file_date BETWEEN '2000-01-01' AND '2024-12-31'
    """, date_cols=["file_date", "res_begin_date", "res_end_date"])
    say(f"    {len(res):,} restatement records, {res.company_fkey.nunique():,} firms")
    for col in ["res_fraud", "res_sec_investigation", "res_adverse", "res_accounting"]:
        rate = (res[col] == 1).mean()
        say(f"    base rate {col}: {(res[col] == 1).sum():,} ({rate:.2%} of restatements)")
    res.to_parquet(f"{DATA}/restatements.parquet")

# ---------- 2. Directors & officers (board network) ----------
say("[2/3] directors/officers (audit.feed17)...")
dirs = db.raw_sql("""
    SELECT company_fkey, do_off_pers_key,
           first_name, middle_name, last_name,
           is_bdmem_pers, is_ceo, is_cfo, is_chair,
           eff_date, file_date, do_off_remains
    FROM audit.feed17_director_and_officer_chan
""", date_cols=["eff_date"])
say(f"    {len(dirs):,} rows, {dirs.do_off_pers_key.nunique():,} persons, "
    f"{dirs.company_fkey.nunique():,} firms")
dirs.to_parquet(f"{DATA}/directors.parquet")

# ---------- 3. Compustat annual fundamentals (M-Score inputs) ----------
say("[3/3] compustat annual (comp.funda)...")
fund = db.raw_sql("""
    SELECT gvkey, datadate, fyear, cik, tic,
           at, act, lct, che, rect, invt, ppent, ppegt,
           revt, cogs, xsga, ni, sale, dp, dltt, oancf, capx
    FROM comp.funda
    WHERE datadate BETWEEN '1999-01-01' AND '2024-12-31'
      AND indfmt = 'INDL' AND datafmt = 'STD'
      AND popsrc = 'D' AND consol = 'C'
      AND at > 0
""", date_cols=["datadate"])
say(f"    {len(fund):,} firm-years, {fund.gvkey.nunique():,} firms")
fund.to_parquet(f"{DATA}/funda.parquet")

db.close()
say(f"collection finished {datetime.datetime.now()}")
log.close()
