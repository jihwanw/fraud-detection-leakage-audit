#!/usr/bin/env python3
"""05_collect_bao_items.py — Compustat raw items following Bao et al. (2020 JAR).

28 raw financial statement items (Cecchini et al. 2010 / Dechow et al. 2011
lineage) used as features without ratio construction.
"""
import os
import builtins
import getpass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

u = p = None
for line in open("/Users/jihwanw/PhD/.wrds_config"):
    if "USERNAME" in line:
        u = line.split("=")[1].strip()
    elif "PASSWORD" in line:
        p = line.split("=")[1].strip()
os.environ["PGPASSWORD"] = p
builtins.input = lambda x="": u
getpass.getpass = lambda x="": p

import wrds  # noqa: E402

db = wrds.Connection(wrds_username=u)
print("pulling comp.funda raw items (Bao 2020 list)...")
fund = db.raw_sql("""
    SELECT gvkey, datadate, fyear, cik, tic,
           act, ap, at, ceq, che, cogs, csho, dlc, dltis, dltt, dp,
           ib, invt, ivao, ivst, lct, lt, ni, ppegt, pstk, re, rect,
           sale, sstk, txp, txt, xint, prcc_f
    FROM comp.funda
    WHERE datadate BETWEEN '2002-01-01' AND '2024-12-31'
      AND indfmt = 'INDL' AND datafmt = 'STD'
      AND popsrc = 'D' AND consol = 'C'
      AND at > 0
""", date_cols=["datadate"])
print(f"{len(fund):,} firm-years, {fund.gvkey.nunique():,} firms")
fund.to_parquet(f"{DATA}/funda_bao.parquet")
db.close()
print("saved funda_bao.parquet")
