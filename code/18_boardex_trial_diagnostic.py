#!/usr/bin/env python3
"""18_boardex_trial_diagnostic.py — Reproduce aggregate name-matching diagnostics.

Queries the licensed WRDS BoardEx trial table in memory and writes aggregate
counts only. No licensed row-level data are saved or redistributed.
"""
import builtins
import configparser
import getpass
import json
import os
import re
from pathlib import Path

import pandas as pd
from wrds_auth import connect

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "boardex_trial_diag.json"


def tokens(value: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+", str(value).upper())


def full_name_key(value: str) -> str:
    return "|".join(tokens(value))


def first_last_key(value: str) -> str:
    parts = tokens(value)
    if not parts:
        return ""
    return f"{parts[0]}|{parts[-1]}" if len(parts) > 1 else parts[0]


def diagnostics(frame: pd.DataFrame, key_function) -> dict:
    pairs = frame[["directorid", "directorname"]].dropna().drop_duplicates().copy()
    pairs["name_key"] = pairs.directorname.map(key_function)
    pairs = pairs[pairs.name_key != ""]
    keys_by_person = pairs.groupby("directorid").name_key.nunique()
    persons_by_key = pairs.groupby("name_key").directorid.nunique()
    return {
        "name_keys": int(persons_by_key.size),
        "false_split_ids": int((keys_by_person > 1).sum()),
        "false_split_pct": round(float((keys_by_person > 1).mean()) * 100, 4),
        "false_merge_keys": int((persons_by_key > 1).sum()),
        "false_merge_pct": round(float((persons_by_key > 1).mean()) * 100, 4),
    }


def main() -> None:
    connection = connect()
    try:
        frame = connection.raw_sql(
            "SELECT companyid, directorid, directorname "
            "FROM boardex_trial.jr_wrds_org_composition"
        )
    finally:
        connection.close()

    result = {
        "source": "boardex_trial.jr_wrds_org_composition",
        "rows": int(len(frame)),
        "persons": int(frame.directorid.nunique()),
        "companies": int(frame.companyid.nunique()),
        "full_name": diagnostics(frame, full_name_key),
        "first_last": diagnostics(frame, first_last_key),
        "note": "Aggregate diagnostics only; licensed row-level trial data are not saved.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
