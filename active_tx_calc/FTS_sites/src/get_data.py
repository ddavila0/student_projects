import fnmatch
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

def load_file(path, cfg):
    path = Path(path)
    file_cols = list(cfg["columns"].values())  # only read the columns we need

    if path.suffix == ".parquet":
        day_dirs = [path]  # also allow a single day as --input
    else:
        day_dirs = sorted(path.glob("*.parquet"))
    if not day_dirs:
        raise SystemExit(f"no *.parquet found in {path}")

    frames = []
    for d in day_dirs:
        day = pd.read_parquet(d, columns=file_cols)
        log.info("Load %s: %d records", d.name, len(day))
        frames.append(day)
    df = pd.concat(frames, ignore_index=True)
    log.info("Load %s: %d days, %d records in total", path.name, len(day_dirs), len(df))

    # match column name
    mapping = {file_col: name for name, file_col in cfg["columns"].items()}
    df = df.rename(columns=mapping)[list(mapping.values())]

    # timestamps: numbers (ms) -> datetime; if already datetime, leave as it is
    unit = cfg.get("timestamp_unit")
    for col in ("start", "complete"):
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], unit=unit)
        else:
            df[col] = pd.to_datetime(df[col])
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