import pandas as pd
import numpy as np
from typing import Tuple

def create_data(input_file:str="../data/demand_data.csv", output_file:str=None, specs: Tuple[str, str, str, str]=tuple()) -> None:
    """
    Create the data for a given schema of inputs
    """

    # Load the data 
    # NOTE the colums are date,store,item,sales
    df = pd.read_csv(input_file)

    # Perfrom the specified operations
    if "one_hot_month" in specs:
        df['month'] = pd.to_datetime(df['date']).dt.month
        month_dummies = pd.get_dummies(df['month'], prefix='month')
        df = pd.concat([df, month_dummies], axis=1)
        df.drop('month', axis=1, inplace=True)
    if "one_hot_week" in specs:
        df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week
        week_dummies = pd.get_dummies(df["week"], prefix="week")
        df = pd.concat([df, week_dummies], axis=1)
        df.drop("week", axis=1, inplace=True)
    if "one_hot_day_of_week" in specs:
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        day_of_week_dummies = pd.get_dummies(df["day_of_week"], prefix="day_of_week")
        df = pd.concat([df, day_of_week_dummies], axis=1)
        df.drop("day_of_week", axis=1, inplace=True)


    # TODO:Add the ciruclar sin and cos verisions of these
    # TODO: Add rolling means of various times
    # TODO: add 

    # Check to see if outfile
    if output_file is not None:
        df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    df = create_data(output_file="test.csv", specs=("one_hot_week", "one_hot_day_of_week"))
    print(df.head())