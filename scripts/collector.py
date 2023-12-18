"""Collectors are used to collect data from different sources into storage with
unified format."""
import functools
import os
import shutil
import time
import typing as tp
import warnings
from datetime import datetime, timedelta

import joblib
import pandas as pd
from ideadata.stock import stock_data, trade_calendar
from tqdm import tqdm

region_index_symbol = {
    "hk": ["^HSI"],
    "uk": ["^FTSE"],
    "jp": ["^N225", "^N300"],
    "fr": ["^FCHI"],
    "us": ["^GSPC", "^DJI", "^NDX", "^SP400", "^SP600", "^SP1500"],
}

region_index_map = {
    "hk": ["hsi", "hsli", "hsmi", "hssi"],
    "uk": ["ftse250", "ftse350", "ftseAllShare"],
    "jp": ["nikkei225", "topix"],
    "fr": ["cac40"],
    "us": ["nasdaq100", "dji"],
}

exchange_map = {
    "hk": "hkex",
    "uk": "lse",
    "jp": "topix",
    "fr": "paris",
    "us": "amex",
}


fmt_converter = lambda x: datetime.strptime(x, "%Y/%m/%d").strftime("%Y-%m-%d")

# -------------------------- ideadata functions --------------------------
def reshape_intraday_df(df: pd.DataFrame, ticks_per_day: int, keys: tp.List):
    values_np = df[keys].values.copy()
    values_np = values_np.reshape(
        -1, ticks_per_day, len(keys)
    )  # (num_stocks, ticks_per_day, num_keys)
    values_np = values_np.transpose(
        2, 0, 1
    )  # (num_keys, num_stocks, ticks_per_day)
    day = df["date"].iloc[0].strftime("%Y%m%d")
    ticks = sorted(list(set(df["time"])))
    ticks = set([pd.Timestamp(day + " " + t) for t in ticks])
    assert len(ticks) == ticks_per_day
    ticks = sorted(list(ticks))
    tickers = sorted(list(set(df["sec_id"])))
    assert len(tickers) == values_np.shape[1]
    dfs = {}
    for i, key in enumerate(keys):
        df = pd.DataFrame(data=values_np[i], index=tickers, columns=ticks)
        dfs[key] = df.transpose()
    return dfs


def get_1min_kbar(start: str, end: str):
    """Get adjusted 1-min kbar data for all stocks in the universe (CN stock).

    Parameters
    ----------
    start : str
        Start date in the form of 'YYYYMMDD'
    end : str
        End date in the form of 'YYYYMMDD'

    """
    # Create trading calendar iterator
    cal = trade_calendar.TradeCalendar("XSHG")
    tcal = cal.get_trade_cal(begin_date=start, end_date=end)
    ticks_per_day = 242  # 0925, 0930, 0931, ..., 1459, 1500
    trading_dates = [d[1]["date"] for d in tcal.iterrows() if d[1]["is_open"]]
    intraday_dfs = {}
    print(f"Trading dates ({len(trading_dates)}):")
    print(trading_dates)

    for day in trading_dates:
        print(f"Processing {day} ...")
        kbar_df = stock_data.get_idea_stk_1min_data(trade_date=day)
        keys = [
            "adj_open_px",
            "adj_close_px",
            "adj_high_px",
            "adj_low_px",
            "adj_volume",
            "act_volume",
            "amount",
            "adj_vwap",
            "c_2_c",
            "log_c_2_c",
        ]
        dfs = reshape_intraday_df(kbar_df, ticks_per_day, keys)
        for i, key in enumerate(keys):
            if key not in intraday_dfs:
                intraday_dfs[key] = []
            intraday_dfs[key].append(dfs[key])

    for k, v in tqdm(intraday_dfs.items(), desc="Dumping to csv"):
        # Dump to csv
        v = pd.concat(v, axis=0)
        v.to_csv(f"./{k}.csv")
    return


# ------------------------------------------------------------------------

# -------------------------- Yahoo Finance functions --------------------------
def deco_retry(retry: int = 5, retry_sleep: int = 3):
    """Retry decorator for request functions that may fail.

    Parameters
    ----------
    retry : int, optional
        Maximum retry, by default 5
    retry_sleep : int, optional
        Time interval between trials in seconds, by default 3

    """

    def deco_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retry = 5 if callable(retry) else retry
            _result = None
            for _i in range(1, _retry + 1):
                try:
                    _result = func(*args, **kwargs)
                    break

                except Exception as e:
                    warnings.warn(f"{func.__name__}: {_i} :{e}")
                    if _i == _retry:
                        raise

                time.sleep(retry_sleep)
            return _result

        return wrapper

    return deco_func(retry) if callable(retry) else deco_func


def wind_to_yahoo_ticker(ticker):
    """A translation function to convert Wind ticker to Yahoo ticker."""
    if ticker.endswith(".N") or ticker.endswith(".O") or ticker.endswith(".A"):
        # US ticker
        symbol = ticker[:-2]
        if symbol.endswith("_U"):
            symbol = symbol[:-2]
        return symbol.replace("_", "-")
    elif ticker.endswith(".L") and ticker.startswith("0"):
        # UK ticker
        # print(f"Converting {ticker} => {ticker[:-2] + ".IL"}")
        return ticker[:-2] + ".IL"
    else:
        return ticker


@deco_retry(retry=5, retry_sleep=3)
def collect_single_ticker_data(
    ticker: str, start: str, end: str, freq: str
) -> tp.Optional[tp.Dict[str, pd.DataFrame]]:
    """Collect from Yahoo Finance the history quotes, number of shares, and
    industry&sector for a single ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol
    start : str
        Start date for quote data, in the form of 'YYYY-MM-DD'
    end : str
        End date for quote data, in the form of 'YYYY-MM-DD'

    """
    try:
        yh = yq.Ticker(ticker)
        df = yh.history(start=start, end=end, interval=freq, adj_ohlc=True)
    except:
        return None
    if not isinstance(df, pd.DataFrame):
        # If there's anything wrong, just return None. Errors will be handled by the caller.
        return None
    return {"quote": df}


def collect_tickers_history(
    tickers: tp.List[str],
    start: str,
    end: str,
    freq: str = "1d",
    num_workers: int = 32,
) -> tp.Dict[str, pd.DataFrame]:
    """Given a list of tickers, collect the history quotes for each ticker. Each
    ticker is processed by a separate worker. The underlying worker function is
    `collect_single_ticker_data`.

    Parameters
    ----------
    tickers : tp.List[str]
        List of tickers in arbitrary format, will be converted to Yahoo ticker internally.
    start : str
        Start date for quote data
    end : str
        End date for quote data
    num_workers : int, optional
        Number of workers, by default 32

    Returns
    -------
    tp.Dict[str, pd.DataFrame]
        A dictionary of ticker to DataFrame

    """
    # Convert tickers to Yahoo format
    new_tickers = sorted(
        set([wind_to_yahoo_ticker(ticker) for ticker in tickers])
    )
    feature_store = {}

    exector = joblib.Parallel(n_jobs=num_workers, backend="loky")

    def worker_fn(ticker: str) -> tp.Dict[str, pd.Series]:
        ticker = ticker.strip()
        data = collect_single_ticker_data(ticker, start, end, freq)
        if data is None:
            return {}
        # Process features
        qdf = data["quote"]
        ret = {}
        # Extract features by column
        for key in qdf.columns:
            sub_df = qdf[key]
            try:
                sub_series = pd.Series(
                    data=sub_df.values,
                    index=sub_df.index.get_level_values(1),
                    name=ticker,
                )
            except Exception as e:
                print("haha")
                raise e
            ret[key] = sub_series
        return ret

    data_dicts: tp.List[tp.Dict[str, pd.Series]] = exector(
        joblib.delayed(worker_fn)(ticker)
        for ticker in tqdm(new_tickers, desc="Collecting data")
    )

    num_valid = sum([len(x) > 0 for x in data_dicts])
    print(
        f"Total number of tickers: {len(data_dicts)}, valid tickers: "
        f"{num_valid}, valid rate {num_valid / len(data_dicts)}"
    )

    feature_keys = set.union(*[set(d.keys()) for d in data_dicts])
    for key in feature_keys:
        feature_store[key] = [d[key] for d in data_dicts if key in d]

    # Concat all tickers into a single dataframe
    ret = {}
    for feature_name, feature_series in feature_store.items():
        ret[feature_name] = pd.concat(feature_series, axis=1).reindex(
            new_tickers, axis=1
        )  # Reorder columns
    return ret


def parse_idx_history(idx_name: str, data_dir: str) -> tp.List[str]:
    """Build index constituent history from 2 files: a current constituent csv
    file and a in/out event history csv file.

    The current constituent file has 4 columns:
        - No.: row number
        - DATE: today date (which is useless)
        - WIND_CODE: ticker code
        - SEC_NAME: security name (which is useless)

    The in/out event history file has several columns:
        - No.: row number (which is useless)
        - TRADEDATE: date of the in/out event
        - TRADECODE: the related security code
        - TRADENAME: the related security name (which is useless)
        - MV: useless
        - WEIGHT: useless
        - TRADESTATUS: one of ("纳入", "剔除")
                       "纳入" for added in, "剔除" for kicked out

    The parsed result is a list of rows, where each row has the form of:
        ticker  in_date  out_date

    If ticker is in the index and never added, the in_date is 1990-01-01.
    If ticker is still in the index, out_date is 2099-12-31.

    Parameters
    ----------
    idx_name : str
        Index name, e.g. "ftse350"
    data_dir : str
        Data directory, it is [root]/meta/[region]

    Returns
    -------
    tp.List[str]
        A list of ticker records in the format of:
        [ticker, in_date, out_date], delimited by tab.

    """
    # Last row is "Wind", useless, drop it
    current_constituent = pd.read_csv(
        os.path.join(data_dir, "pools", f"{idx_name}_constituents.csv")
    ).iloc[:-1]
    inout_history = pd.read_csv(
        os.path.join(data_dir, "pools", f"{idx_name}_history.csv")
    ).iloc[:-1]

    records = {}
    epoch = "1990/01/01"

    for i in range(len(current_constituent)):
        row = current_constituent.iloc[i]
        ticker = row["WIND_CODE"]
        records[ticker] = [("in", epoch), ("out", "2099/12/31")]

    for i in range(len(inout_history)):
        row = inout_history.iloc[i]
        ticker = row["TRADECODE"]
        tick = row["TRADEDATE"]
        if row["TRADESTATUS"] == "纳入":
            record = records.setdefault(ticker, [])
            record.append(("in", tick))
        else:
            record = records.setdefault(ticker, [])
            record.append(("out", tick))

    records = {k: sorted(v, key=lambda x: x[1]) for k, v in records.items()}

    ret = []

    for k, v in records.items():
        if "!" in k:  # Skip delisted stocks
            continue
        history_events = sorted(v, key=lambda x: x[1])
        in_stamp = epoch
        for event in history_events:
            if event[0] == "in":
                in_stamp = event[1]
            elif event[0] == "out":
                stock = wind_to_yahoo_ticker(k)
                in_date = fmt_converter(in_stamp)
                out_date = fmt_converter(event[1])
                ret.append(f"{stock}\t{in_date}\t{out_date}")

    with open(os.path.join(data_dir, "pools", f"{idx_name}_cal.txt"), "w") as f:
        f.write("\n".join(ret))

    return ret


def get_tickers(
    region: str,
    data_dir: str,
    read_existing: bool = False,
) -> tp.List[str]:
    """Get all tickers in the given region. The ticker list comes from two sources:
    1. Today's all tickers ([region]_all_tickers.csv)
    2. Index constituent history ([index]_cal.txt)
    3. (If exists) all tickers csv of that exchange

    If '!' is in the ticker, it is a delisted ticker, and should be ignored.

    Parameters
    ----------
    region : str
        The region, e.g. "us", "uk", "hk"
    data_dir: str
        Root dataset directory that contains "main" and "meta"
    read_existing: bool
        If true, check if the 'all.txt' universe file exists, and read it if so.

    Returns
    -------
    tp.List[str]
        A list of total tickers in Yahoo format.

    Examples
    --------
    >>> get_tickers(region="us", data_dir="my_data_dir", read_existing=True)
    Will read from my_data_dir/main/us/instruments/all.txt
    >>> get_tickers(region="us", data_dir="my_data_dir/meta/us", read_existing=False)
    Will read all historical indices and all tickers in that exchange from meta dir
    """

    ticker_path = os.path.join(
        data_dir, "main", region, "instruments", "all.txt"
    )
    if os.path.exists(ticker_path) and read_existing:
        all_tickers = pd.read_csv(
            ticker_path, delimiter="\t", header=None, keep_default_na=False
        )
        return all_tickers[0].tolist()

    # Source 1: *cap_ind.csv in that region
    today_all_tickers = pd.read_csv(
        os.path.join(data_dir, f"{region}_cap_ind.csv"), keep_default_na=False
    ).iloc[:-1]
    tickers = set(today_all_tickers["WindCodes"].tolist())

    # Source 2: all historical index constituents
    for index in region_index_map[region]:
        index_constituent = pd.read_csv(
            os.path.join(data_dir, "pools", f"{index}_cal.txt"),
            delimiter="\t",
            header=None,
            keep_default_na=False,
        )
        tickers.update(index_constituent[0].tolist())

    # Source 3: all tickers in that exchange
    all_tickers_xchg = os.path.join(
        data_dir, "pools", f"{exchange_map[region]}_all_tickers.csv"
    )
    if os.path.exists(all_tickers_xchg):
        all_tickers_xchg = pd.read_csv(
            all_tickers_xchg, keep_default_na=False, index_col=0
        )[:-1]
        tickers.update(all_tickers_xchg["WIND_CODE"].tolist())

    stock_list = [wind_to_yahoo_ticker(x) for x in list(tickers)]
    index_list = region_index_symbol[region]
    return stock_list + index_list


def append_remaining_rows_to_csv(df_b: pd.DataFrame, csv_file: str):
    # Read df_a from the CSV file
    df_a = pd.read_csv(csv_file, index_col=0)

    # Align the order of columns in df_b to match df_a
    df_b = df_b[df_a.columns]

    # Find the indices in df_b that do not exist in df_a
    remaining_indices = df_b.index.difference(df_a.index)

    # Get the rows with the remaining indices
    df_remaining = df_b.loc[remaining_indices]

    # Print the number of rows to be appended
    print(f"Appending {len(df_remaining)} rows to {csv_file}")

    # Append the remaining rows to the CSV file without writing the header
    df_remaining.to_csv(csv_file, mode="a", header=False)


def download_daily_data(
    region: str,
    start: str,
    end: str,
    raw_data_dir: str,
    dump_data_dir: str,
    num_workers: int,
):
    """
    Prepare a data directory for a given region.
    Steps:
        - Get all tickers in the region
        - Get all tickers' fundamentals, put to [data_root]/fundamentals
        - Get all tickers' history data, put to [data_root]/features/day
        - Get all tickers' index history data, put to [data_root]/instruments

    Parameters
    ----------
    region : str
        The region, e.g. 'hk'
    start : str
        Tick in format of YYYYMMDD
    end : str
        Tick in format of YYYYMMDD
    raw_data_dir : str
        The directory containing meta data, with a subdir 'pools'
    dump_data_dir : str
        The target directory to dump data, consisting of 'calendars', 'features', 'fundamental', 'instruments'
    """
    # exchange = exchange_map[region]
    print(f"Meta data: {raw_data_dir}\nDump to: {dump_data_dir}")

    # Make dir structure
    for subdir in ["calendars", "features", "fundamental", "instruments"]:
        os.makedirs(os.path.join(dump_data_dir, subdir), exist_ok=True)
    print(f"Processing {region.upper()} from {start} to {end}")

    # Parse index history and translate them into qlib format
    region_index_list = region_index_map[region]
    print(f"Processing index history. Available index: {region_index_list}")
    for index in region_index_list:
        print(f"Processing {index}")
        idx_events = parse_idx_history(idx_name=index, data_dir=raw_data_dir)
        instrument_fpath = os.path.join(
            dump_data_dir, "instruments", f"{index}.txt"
        )
        with open(instrument_fpath, "w") as f:
            f.write("\n".join(idx_events))
    if region == "us":
        # US indices can be retrieved with the help of qlib
        print(
            "For US stock market, SP400, SP500, SP600 can be retrieved "
            "with qlib. Please do that manually."
        )

    # Dump all tickers
    start = datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d")
    end = datetime.strptime(end, "%Y%m%d").strftime("%Y-%m-%d")
    all_tickers = get_tickers(
        region, data_dir=raw_data_dir, read_existing=False
    )
    print(f"Got {len(all_tickers)} tickers from exchange {region}")
    print("Dumping all tickers to all.txt")
    with open(os.path.join(dump_data_dir, "instruments", "all.txt"), "w") as f:
        for ticker in sorted(set(all_tickers)):  # De-duplication
            # TODO: There's a caveat here. We assume all tickers that ever
            # existed are valid across our time interval.
            # This is not true for some tickers. Now we leave the processing
            # to downstream missing value processing routines.
            f.write(
                "\t".join(
                    [
                        wind_to_yahoo_ticker(ticker),
                        start,
                        end,
                    ]
                )
            )
            f.write("\n")
    print(f"Stock pools constructed at {dump_data_dir}/instruments")

    # Get historical quote data (daily)
    historical_quotes = collect_tickers_history(
        tickers=all_tickers,
        start=start,
        end=end,
        freq="1d",
        num_workers=num_workers,
    )
    for feat_name, feat_data in historical_quotes.items():
        os.makedirs(os.path.join(dump_data_dir, "features/day"), exist_ok=True)
        feat_data.to_csv(
            os.path.join(dump_data_dir, "features/day", f"{feat_name}.csv")
        )

    # Find all ticks and dump to [data_root]/calendars
    all_ticks = set()
    for feat_name, feat_data in historical_quotes.items():
        all_ticks.update(feat_data.index.tolist())
    all_ticks = sorted(all_ticks)
    with open(os.path.join(dump_data_dir, "calendars", "day.txt"), "w") as f:
        for tick in all_ticks:
            f.write(f"{tick}\n")

    # Get fundamentals
    region_fundemental_fpath = os.path.join(
        raw_data_dir, f"{region}_cap_ind.csv"
    )
    pd.read_csv(region_fundemental_fpath)
    shutil.copy(
        src=region_fundemental_fpath,
        dst=os.path.join(dump_data_dir, "fundamental", "ind_cap.csv"),
    )

    # Finished, print a message
    print(f"Finished processing {region.upper()} from {start} to {end}")


def collect_yahoo_1min_data(
    tickers: tp.List[str],
    data_dir: str,
    dates: tp.Optional[tp.Tuple[str, str]] = None,
    num_workers: int = 100,
):
    # Check if dates are valid
    if dates is None:
        today = datetime.now()
        end_date = today.strftime("%Y%m%d")
        start_date = (today - timedelta(days=29)).strftime("%Y%m%d")
        dates = (start_date, end_date)

    # Check if history exceeds 30 days
    start_date = datetime.strptime(dates[0], "%Y%m%d")
    end_date = datetime.strptime(dates[1], "%Y%m%d")
    if (datetime.now() - start_date).days > 30:
        raise ValueError(
            "History exceeds 30 days. Please adjust the date range."
        )

    # Collect data in chunks of 7 days
    date_ranges = []
    while start_date < end_date:
        next_week = start_date + timedelta(days=7)
        date_ranges.append((start_date, min(end_date, next_week)))
        start_date = next_week

    # Collect data using yahooquery
    all_data = None
    for date_range in date_ranges:
        start = date_range[0].strftime("%Y-%m-%d")
        end = date_range[1].strftime("%Y-%m-%d")
        print(f"Fetching data from {start} to {end}...")
        data = collect_tickers_history(
            tickers, start, end, "1m", num_workers=num_workers
        )
        if all_data is None:
            all_data = data
        else:
            # Beware of boundary conditions!
            for key in data.keys():
                try:
                    current_df = all_data[key]
                    updated_df = data[key]
                    new_df = pd.concat([current_df, updated_df], axis=0)
                    all_data[key] = new_df.drop_duplicates()
                except KeyError:
                    all_data[key] = data[key]

    # Align the ticks in all_data
    all_ticks = set()
    for feat_name, feat_data in all_data.items():
        all_ticks.update(feat_data.index.tolist())
    all_ticks = sorted(all_ticks)
    for feat_name, feat_data in all_data.items():
        all_data[feat_name] = feat_data.reindex(all_ticks)

    # Dump ticks
    with open(os.path.join(data_dir, "calendars", "1min.txt"), "w") as f:
        for tick in all_ticks:
            f.write(f"{tick}\n")

    print("Finished collecting 1min data.")

    # Update data in the directories
    print("Updating data directories...")
    calendars_dir = os.path.join(data_dir, "calendars")
    features_dir = os.path.join(data_dir, "features", "1min")
    instruments_dir = os.path.join(data_dir, "instruments")

    # Create data directories if they don't exist
    os.makedirs(calendars_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(instruments_dir, exist_ok=True)

    # Update calendars/1min.txt
    calendar_file = os.path.join(calendars_dir, "1min.txt")
    if not os.path.isfile(calendar_file):
        open(calendar_file, "w").close()

    with open(calendar_file, "a+") as f:
        f.seek(0)
        existing_ticks = set(f.read().splitlines())
        current_ticks = set(
            [
                x.strftime("%Y%m%dT%H%M%S")
                for x in all_data["open"].index.get_level_values("date")
            ]
        )
        new_ticks = sorted(current_ticks - existing_ticks)
        for tick in new_ticks:
            f.write(f"{tick}\n")

    # Update features/*.csv
    # for ticker, data in all_data.items():
    for key, v in all_data.items():
        feature_df = v

        feature_file = os.path.join(features_dir, f"{key}.csv")
        if not os.path.isfile(feature_file):
            feature_df.to_csv(feature_file)
        else:
            append_remaining_rows_to_csv(df_b=feature_df, csv_path=feature_file)

    print("Data update complete.")


