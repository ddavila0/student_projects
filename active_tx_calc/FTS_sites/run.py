"""
    python run_analysis.py --input data/all.csv --site T2_US_UCSD --start 2025-03-03 --end 2025-03-10
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.get_data import filter_period, load_file, select_sources
from src.functions import build_model_data
from src.ml_model import fit_weights
from src.plot_graph import plot_throughput, plot_weights

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config" / "default.yaml"
log = logging.getLogger("FTS_sites")


def parse_args():
    p = argparse.ArgumentParser(
        description="weight for T2_US SITES",
        epilog=f"config file: {CONFIG_PATH}"
    )
    # Todo：1. 同一周的文件在一个文件夹里，不需要start time和end time，就是提取文件夹 2. 是parquet file
    p.add_argument("--input", required=True, help="datafile(.csv)")
    #p.add_argument("--start", required=True, help="start time: yyyy-mm-dd")
    #p.add_argument("--end", required=True, help="end time, same with start time")
    #p.add_argument("--site", required=True, help="dst_site, T2_US_*")
    p.add_argument("--output-dir", help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


if __name__ == "__main__":
    main()