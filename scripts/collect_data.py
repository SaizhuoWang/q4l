import argparse
import os

import numpy as np
import pandas as pd

from scripts.collector import (
    collect_yahoo_1min_data,
    download_daily_data,
    get_tickers,
)


def main(args):
    regions = os.listdir(os.path.join(args.data_root, "meta"))
    for region in regions:
        # for region in ["jp"]:
        # if region != 'jp': continue
        print(f"Collecting data for {region}")
        target_dir = os.path.join(args.data_root, "main", region)
        download_daily_data(
            region=region,
            start=args.start_date,
            end=args.end_date,
            raw_data_dir=os.path.join(args.data_root, "meta", region),
            dump_data_dir=target_dir,
            num_workers=args.num_workers,
        )


def compute_change(args):
    regions = os.listdir(os.path.join(args.data_root, "meta"))
    for region in regions:
        print(f"Collecting data for {region}")
        region_dir = os.path.join(args.data_root, "main", region, "features")
        freq = os.listdir(region_dir)
        for f in freq:
            feature_dir = os.path.join(region_dir, f)
            open_price = pd.read_csv(
                os.path.join(feature_dir, "open.csv"), index_col=0
            )
            close_price = pd.read_csv(
                os.path.join(feature_dir, "close.csv"), index_col=0
            )
            change = (close_price - open_price) / open_price
            change.to_csv(os.path.join(feature_dir, "change.csv"))
            factor_np = np.ones_like(change, dtype=np.float32)
            factor = pd.DataFrame(
                factor_np,
                index=change.index.copy(),
                columns=change.columns.copy(),
            )
            factor.to_csv(os.path.join(feature_dir, "factor.csv"))


def crontab_routine(args):
    regions = os.listdir(os.path.join(args.data_root, "meta"))
    for region in regions:
        # end_time = datetime.now() - timedelta(days=1)
        # start_time = end_time - timedelta(days=7)
        tickers = get_tickers(
            region=region, data_dir=args.data_root, read_existing=True
        )
        collect_yahoo_1min_data(
            tickers=tickers,
            data_dir=os.path.join(args.data_root, "main", region),
            num_workers=args.num_workers,
            # dates=(start_time.strftime("%Y%m%d"), end_time.strftime("%Y%m%d")),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
    )
    parser.add_argument("--start_date", type=str, default="20060101")
    parser.add_argument("--end_date", type=str, default="20230428")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--func", type=str, default="main")
    args = parser.parse_args()

    globals()[args.func](args)
