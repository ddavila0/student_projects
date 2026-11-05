"""
    python run_analysis.py --input data/all.csv --site T2_US_UCSD --start 2025-03-03 --end 2025-03-10
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.get_data import load_file, select_sources
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
    # 1. 同一周的文件在一个文件夹里，不需要start time和end time，就是提取文件夹 2. 是parquet file
    p.add_argument("--input", required=True,
                   help="week folder with fts_YYYYMMDD.parquet inside, or one day .parquet")
    #p.add_argument("--start", required=True, help="start time: yyyy-mm-dd")
    #p.add_argument("--end", required=True, help="end time, same with start time")
    p.add_argument("--site", required=True, help="dst_site, T2_US_*")
    p.add_argument("--output-dir", help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(CONFIG_PATH))

    if args.output_dir:
        cfg["output"] = args.output_dir
    site = args.site

    # read all days in the week folder
    df = load_file(args.input, cfg)

    # check dst_site
    df = df[df["destination"] == site]
    if df.empty:
        raise SystemExit(f"no transfers to {site}, check the spelling of --site")

    # keep only CERN / T1 / T2_US as src_site
    df, sources = select_sources(df.dropna(subset=["source"]), cfg)
    log.info("dst_site %s: %d records, src_sites: %s", site, len(df), ", ".join(sources))
    min_n = cfg.get("min_transfers", 100)
    if len(df) < min_n:
        raise SystemExit(f"only {len(df)} records, less than min_transfers={min_n}")

    # active transfers & throughput per src_site, then K-fold weights
    model_data, buckets = build_model_data(df, sources, cfg)
    cv, weights = fit_weights(model_data, buckets, cfg)

    # save outputs
    # output： output/datafile_name/outputs
    out = Path(cfg["output"]) / Path(args.input).stem / site
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_info.json").write_text(json.dumps(
        {"input": args.input, "site": site, "config_file": str(CONFIG_PATH), "config": cfg},
        indent=2, ensure_ascii=False, default=str))

    cv.to_csv(out / "cv_results.csv")
    pd.DataFrame({
        "source_site": weights.index,
        "weight_mean": weights.mean(axis=1).values,
        "weight_std": weights.std(axis=1).values,
    }).to_csv(out / "summary.csv", index=False)
    if cfg["save_plots"]:
        plot_throughput(model_data, site, out / "throughput.png")
        plot_weights(weights, site, out / "weights.png")

    print(cv.round(3))
    log.info("done: R2 mean %.3f, WAPE mean %.3f, results in %s", cv["R2"].mean(), cv["WAPE"].mean(), out)

if __name__ == "__main__":
    main()