"""Portable WRDS authentication for the public replication package.

Supported methods, in order:
1. WRDS_USERNAME and WRDS_PASSWORD environment variables.
2. WRDS_CONFIG environment variable pointing to either an INI file with a
   [wrds] section or a simple USERNAME=... / PASSWORD=... file.
3. ~/.wrds_config in either format.

Credentials are never printed or written by this module.
"""
import builtins
import configparser
import getpass
import os
from pathlib import Path

import wrds


def _read_file(path: Path):
    parser = configparser.ConfigParser()
    try:
        if parser.read(path) and parser.has_section("wrds"):
            return parser.get("wrds", "username"), parser.get("wrds", "password")
    except (configparser.Error, KeyError):
        pass
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    username = values.get("username") or values.get("wrds_username")
    password = values.get("password") or values.get("wrds_password")
    return username, password


def credentials():
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")
    if username and password:
        return username, password
    candidates = []
    if os.getenv("WRDS_CONFIG"):
        candidates.append(Path(os.environ["WRDS_CONFIG"]).expanduser())
    candidates.append(Path.home() / ".wrds_config")
    for path in candidates:
        if path.exists():
            username, password = _read_file(path)
            if username and password:
                return username, password
    raise RuntimeError(
        "WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD, "
        "or set WRDS_CONFIG to a credential file."
    )


def connect():
    username, password = credentials()
    os.environ["PGPASSWORD"] = password
    builtins.input = lambda prompt="": username
    getpass.getpass = lambda prompt="": password
    return wrds.Connection(wrds_username=username)
