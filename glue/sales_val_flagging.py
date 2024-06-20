import awswrangler as wr
import boto3
import datetime
import numpy as np
import os
import pandas as pd
import pytz
import re
import sys
from pyathena import connect
from pyathena.pandas.util import as_pandas
from dateutil.relativedelta import relativedelta


def sql_type_to_pd_type(sql_type):
    """
    This function translates SQL data types to equivalent
    pandas dtypes, using athena parquet metadata,
    used within a dictionary comprehension.
    """

    # This is used to fix dtype so there is not error thrown in
    # deviation_dollars() in flagging script on line 375
    if sql_type in ["decimal"]:
        return "float64"


# -----------------------------------------------------------------------------
# Helpers for writing tables
# -----------------------------------------------------------------------------


def modify_dtypes(df):
    """
    Helper function for resolving Athena parquet errors. Made for the parameter table.

    Sometimes, when writing data of pandas dtypes to S3/athena, there
    are errors with consistent metadata between the parquet files, even though
    in pandas the dtypes are consistent. This script removes all object types
    and strandardizes int values. This function has proved to be a fix for
    problems of this nature.

    Inputs:
       df: df ready to write to parquet in S3
    Outputs:
        df: df of standardized dtypes
    """

    # Convert Int64 columns to int64
    for col in df.select_dtypes("Int64").columns:
        df[col] = df[col].astype("int64")

    # Function to ensure all numeric values are converted to float
    def to_float(value):
        if isinstance(value, list):
            return [float(i) for i in value]
        elif isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        elif isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        else:
            return value

    # Apply conversion function to specified columns
    conversion_columns = ["dev_bounds", "ptax_sd"]
    for col in conversion_columns:
        if col in df.columns:
            df[col] = df[col].apply(to_float)

    # Adjustments for specific string columns, ensuring compatibility
    string_columns = [
        "run_id",
        "run_filter",
        "iso_forest_cols",
        "stat_groups",
        "time_frame",
        "sales_to_write_filter",
        "housing_market_class_codes",
    ]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df
