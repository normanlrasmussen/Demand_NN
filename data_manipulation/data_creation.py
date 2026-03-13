import pandas as pd
import numpy as np
from typing import Tuple

ROLLING_MEAN_SPECS = ("7_day_rolling_mean", "30_day_rolling_mean", "90_day_rolling_mean", "180_day_rolling_mean", "365_day_rolling_mean")
ROLLING_VOLATILITY_SPECS = ("7_day_rolling_volatility", "30_day_rolling_volatility", "90_day_rolling_volatility", "180_day_rolling_volatility", "365_day_rolling_volatility")
ROLLING_MIN_SPECS = ("7_day_rolling_min", "30_day_rolling_min", "90_day_rolling_min", "180_day_rolling_min", "365_day_rolling_min")
ROLLING_EMA_SPECS = ("7_day_rolling_ema", "30_day_rolling_ema", "90_day_rolling_ema", "180_day_rolling_ema", "365_day_rolling_ema")
LAG_SPECS = {"1_day_lag": 1, "2_day_lag": 2, "3_day_lag": 3, "4_day_lag": 4, "5_day_lag": 5, "6_day_lag": 6, "7_day_lag": 7,
             "14_day_lag": 14, "28_day_lag": 28, "365_day_lag": 365}

def create_data(input_file:str="../data/demand_data.csv", output_file:str=None, specs: Tuple[str, ...]=()) -> None:
    """
    Create the data for a given schema of inputs

    Here is a list of all possible specs:
        "one_hot_month",
        "one_hot_week",
        "one_hot_day_of_week",
        "one_hot_weekend",
        "circular_sin_cos_month",
        "circular_sin_cos_week",
        "circular_sin_cos_day_of_week",
        "7_day_rolling_mean",
        "30_day_rolling_mean",
        "90_day_rolling_mean",
        "180_day_rolling_mean",
        "365_day_rolling_mean",
        "7_day_rolling_volatility",
        "30_day_rolling_volatility",
        "90_day_rolling_volatility",
        "180_day_rolling_volatility",
        "365_day_rolling_volatility",
        "7_day_rolling_min",
        "30_day_rolling_min",
        "90_day_rolling_min",
        "180_day_rolling_min",
        "365_day_rolling_min",
        "7_day_rolling_ema",
        "30_day_rolling_ema",
        "90_day_rolling_ema",
        "180_day_rolling_ema",
        "365_day_rolling_ema",
        "1_day_lag",
        "2_day_lag",
        "3_day_lag",
        "4_day_lag",
        "5_day_lag",
        "6_day_lag",
        "7_day_lag",
        "14_day_lag",
        "28_day_lag",
        "365_day_lag",
        "diff_1_day",
        "diff_7_day",
        "diff_30_day",
        "diff_90_day",
        "diff_180_day",
        "diff_365_day",
    """

    # Load the data 
    # NOTE the colums are date,store,item,sales
    df = pd.read_csv(input_file)

    # Add various features related to time
    date_dt = pd.to_datetime(df["date"])
    if "one_hot_month" in specs:
        # One hot encode the month
        df['month'] = date_dt.dt.month
        month_dummies = pd.get_dummies(df['month'], prefix='month')
        df = pd.concat([df, month_dummies], axis=1)
        df.drop('month', axis=1, inplace=True)
    if "one_hot_week" in specs:
        # One hot encode the week
        df["week"] = date_dt.dt.isocalendar().week
        week_dummies = pd.get_dummies(df["week"], prefix="week")
        df = pd.concat([df, week_dummies], axis=1)
        df.drop("week", axis=1, inplace=True)
    if "one_hot_day_of_week" in specs:
        # One hot encode the day of the week
        df["day_of_week"] = date_dt.dt.dayofweek
        day_of_week_dummies = pd.get_dummies(df["day_of_week"], prefix="day_of_week")
        df = pd.concat([df, day_of_week_dummies], axis=1)
        df.drop("day_of_week", axis=1, inplace=True)
    if "circular_sin_cos_month" in specs:
        # Add the circular sin and cos versions of the month
        df["month_sin"] = np.sin(2 * np.pi * date_dt.dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * date_dt.dt.month / 12)
    if "circular_sin_cos_week" in specs:
        # Add the circular sin and cos versions of the week
        df["week_sin"] = np.sin(2 * np.pi * date_dt.dt.isocalendar().week / 52)
        df["week_cos"] = np.cos(2 * np.pi * date_dt.dt.isocalendar().week / 52)
    if "circular_sin_cos_day_of_week" in specs:
        # Add the circular sin and cos versions of the day of the week
        df["day_of_week_sin"] = np.sin(2 * np.pi * date_dt.dt.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * date_dt.dt.dayofweek / 7)
    if "one_hot_weekend" in specs:
        # One hot encode weekend (1 = Saturday or Sunday, 0 = weekday)
        df["weekend"] = (date_dt.dt.dayofweek >= 5).astype(int)
        df['weekday'] = (date_dt.dt.dayofweek < 5).astype(int)

    # Add rolling statistics (sort when any rolling spec is requested)
    any_rolling = any(r in specs for r in ROLLING_MEAN_SPECS + ROLLING_VOLATILITY_SPECS + ROLLING_MIN_SPECS + ROLLING_EMA_SPECS)
    if any_rolling:
        # Sort the data by store, item, and date for any rolling mean
        df = df.sort_values(["store", "item", "date"])
    if "7_day_rolling_mean" in specs:
        df["7_day_rolling_mean"] = (
            df.groupby(["store", "item"])["sales"]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    if "30_day_rolling_mean" in specs:
        df["30_day_rolling_mean"] = (
            df.groupby(["store", "item"])["sales"]
            .rolling(window=30, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    if "90_day_rolling_mean" in specs:
        df["90_day_rolling_mean"] = (
            df.groupby(["store", "item"])["sales"]
            .rolling(window=90, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    if "180_day_rolling_mean" in specs:
        df["180_day_rolling_mean"] = (
            df.groupby(["store", "item"])["sales"]
            .rolling(window=180, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    if "365_day_rolling_mean" in specs:
        df["365_day_rolling_mean"] = (
            df.groupby(["store", "item"])["sales"]
            .rolling(window=365, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    # Add rolling volatility (std), then backfill initial NaNs within each series
    for spec, window in [("7_day_rolling_volatility", 7), ("30_day_rolling_volatility", 30),
                         ("90_day_rolling_volatility", 90), ("180_day_rolling_volatility", 180),
                         ("365_day_rolling_volatility", 365)]:
        if spec in specs:
            df[spec] = (
                df.groupby(["store", "item"])["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).std())
            )
            # Backfill NaNs within each (store, item) time series
            df[spec] = df.groupby(["store", "item"])[spec].transform(lambda x: x.bfill())

    # Add rolling minimum
    for spec, window in [("7_day_rolling_min", 7), ("30_day_rolling_min", 30),
                         ("90_day_rolling_min", 90), ("180_day_rolling_min", 180),
                         ("365_day_rolling_min", 365)]:
        if spec in specs:
            df[spec] = (
                df.groupby(["store", "item"])["sales"]
                .transform(lambda x: x.rolling(window=window, min_periods=1).min())
            )

    # Add rolling EMA
    for spec, span in [("7_day_rolling_ema", 7), ("30_day_rolling_ema", 30),
                       ("90_day_rolling_ema", 90), ("180_day_rolling_ema", 180),
                       ("365_day_rolling_ema", 365)]:
        if spec in specs:
            df[spec] = (
                df.groupby(["store", "item"])["sales"]
                .transform(lambda x: x.ewm(span=span, adjust=False).mean())
            )
    if any_rolling:
        df = df.sort_values(["date", "store", "item"])

    # Add lagged variables (1-7, 14, 28, 365 days); fill NaN with forward-looking values (bfill)
    lag_cols_added = []
    for spec, period in LAG_SPECS.items():
        if spec in specs:
            df[spec] = df.groupby(["store", "item"])["sales"].shift(period)
            lag_cols_added.append(spec)
    if lag_cols_added:
        for col in lag_cols_added:
            df[col] = df.groupby(["store", "item"])[col].transform(lambda x: x.bfill())

    # Add diff features (sales - X day lag); fill NaN with bfill
    DIFF_SPECS = {"diff_1_day": 1, "diff_7_day": 7, "diff_30_day": 30,
                  "diff_90_day": 90, "diff_180_day": 180, "diff_365_day": 365}
    diff_cols_added = []
    for spec, period in DIFF_SPECS.items():
        if spec in specs:
            shifted = df.groupby(["store", "item"])["sales"].shift(period)
            df[spec] = df["sales"] - shifted
            diff_cols_added.append(spec)
    if diff_cols_added:
        for col in diff_cols_added:
            df[col] = df.groupby(["store", "item"])[col].transform(lambda x: x.bfill())


    # Check to see if outfile
    if output_file is not None:
        df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    df = create_data()
    print(df.head())
    print(df[(df["store"] == 1) & (df["item"] == 1)].head())
