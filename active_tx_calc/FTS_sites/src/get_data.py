import fnmatch
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

def load_file(path, cfg):
    path = Path(path)
    df = pd.read_csv(path)
    log.info("Load %s: %d records", path, len(df))

    # match column name
    mapping = {file_col: name for name, file_col in cfg["columns"].items()}
    df = df.rename(columns=mapping)[list(mapping.values())]

    unit = cfg.get("timestamp_unit")
    for col in ("start", "complete"):
        df[col] = pd.to_datetime(df[col], unit=unit) if unit else pd.to_datetime(df[col])
    return df

def filter_period(df, cfg):
# by --start and --end to filter out which period of time we are analyzing.
# (delete if different weeks in different file)
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    df = df[(df["start"] >= pd.Timestamp(start)) & (df["start"] < pd.Timestamp(end))]
# check if the time period is selected
    if df.empty:
        raise SystemExit("there is no data in the time period --start / --end")
    log.info("time_range: %s ~ %s: %d records", start, end, len(df))
    return df

def select_sources(df, cfg):
    # filter out any src_site which is not CERN, T1, T2_US (if not done before)
    present = sorted(df["source"].dropna().unique())
    ordered = []
    for pat in cfg["source_sites"]:
        for s in present:
            if fnmatch.fnmatch(s, pat) and s not in ordered:
                ordered.append(s)
    return df[df["source"].isin(ordered)], ordered