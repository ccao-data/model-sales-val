import pandas as pd
import os
import awswrangler as wr
import subprocess as sp
import numpy as np
import pytz
from random_word import RandomWords
import datetime

"""
Helper functions used by this script and those in manual_flagging/
"""

def add_rolling_window(df):
    """
    This function implements a rolling window logic such that
    the data is formatted for the flagging script to correctly
    run the year-long window grouping for each obs.
    Inputs:
        df: dataframe of sales that we need to flag, data should be 11 months back 
            from the earliest unflagged sale in order for the rolling window logic to work
    Outputs:
        df: dataframe that has exploded each observation into 12 observations with a 12 distinct
            rolling window columns
    """
    max_date = df["meta_sale_date"].max()
    df = (
        # Creates dt column with 12 month dates
        df.assign(
            rolling_window=df["meta_sale_date"].apply(
                lambda x: pd.date_range(start=x, periods=12, freq="M")
            )
        )
        # Expand rolling_windows dates to individual rows
        .explode("rolling_window")
        # Tag original observations
        .assign(
            original_observation=lambda df: df["meta_sale_date"].dt.month
            == df["rolling_window"].dt.month
        )
        # Simplify to month level
        .assign(rolling_window=lambda df: df["rolling_window"].dt.to_period("M"))
        # Filter such that rolling_window isn't extrapolated into future, we are concerned with historic and present-month data
        .loc[lambda df: df["rolling_window"] <= max_date.to_period("M")]
        # Back to float for flagging script
        .assign(
            rolling_window=lambda df: df["rolling_window"]
            .apply(lambda x: x.strftime("%Y%m"))
            .astype(int)
        )
    )

    return df


def finish_flags(df, start_date, exempt_data, manual_update):
    """
    This functions 
        -takes the flagged data from the mansueto code
        -removes the unneeded observations used for the rolling window calculation
        -finishes adding sales val cols for flag table upload
        - 
    """
    # Remove duplicate rows
    df = df[df["original_observation"]]
    # Discard pre-2014 data
    df = df[df["meta_sale_date"] >= start_date]

    # Utilize PTAX-203, complete binary columns
    df = (
        df.rename(columns={"sv_is_outlier": "sv_is_autoval_outlier"})
        .assign(
            sv_is_autoval_outlier=lambda df: df["sv_is_autoval_outlier"] == "Outlier",
            sv_is_outlier=lambda df: df["sv_is_autoval_outlier"] | df["sale_filter_is_outlier"],
            # Incorporate PTAX in sv_outlier_type
            sv_outlier_type=lambda df: np.where(
                (df["sv_outlier_type"] == "Not outlier") & df["sale_filter_is_outlier"],
                "PTAX-203 flag",
                df["sv_outlier_type"],
            ),
        )
        .assign(
            # Change sv_is_outlier to binary
            sv_is_outlier=lambda df: (df["sv_outlier_type"] != "Not outlier").astype(int),
            # PTAX-203 binary
            sv_is_ptax_outlier=lambda df: np.where(df["sv_outlier_type"] == "PTAX-203 flag", 1, 0),
            # Heuristics flagging binary column
            sv_is_heuristic_outlier=lambda df: np.where(
                (df["sv_outlier_type"] != "PTAX-203 flag") & (df["sv_is_outlier"] == 1), 1, 0
            ),
        )
    )

    # Manually impute ex values as non-outliers
    exempt_to_append = exempt_data.meta_sale_document_num.reset_index().drop(columns="index")
    exempt_to_append["sv_is_outlier"] = 0
    exempt_to_append["sv_is_ptax_outlier"] = 0
    exempt_to_append["sv_is_heuristic_outlier"] = 0
    exempt_to_append["sv_outlier_type"] = "Not Outlier"

    cols_to_write = [
        "meta_sale_document_num",
        "rolling_window",
        "sv_is_outlier",
        "sv_is_ptax_outlier",
        "sv_is_heuristic_outlier",
        "sv_outlier_type",
    ]

    # Create run_id
    r = RandomWords()
    random_word_id = r.get_random_word()
    timestamp = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime("%Y-%m-%d_%H:%M")
    run_id = timestamp + "-" + random_word_id


    # Control flow for incorporating versioning
    dynamic_assignment = {
        "run_id": run_id,
        "rolling_window": lambda df: pd.to_datetime(df["rolling_window"], format="%Y%m").dt.date,
    }

    if not manual_update:
        dynamic_assignment["version"] = 1

    # Incorporate exempt values and finalize to write to flag table
    df = (
        # TODO: exempt will have an NA for rolling_window - make sure that is okay
        pd.concat([df[cols_to_write], exempt_to_append])
        .reset_index(drop=True)
        .assign(**dynamic_assignment)
    )

    return df, run_id, timestamp


def sql_type_to_pd_type(sql_type):
    """
    This function translates SQL data types to equivalent
    pandas dtypes, using athena parquet metadata
    """

    # This is used to fix dtype so there is not error thrown in
    # deviation_dollars() in flagging script on line 375
    if sql_type in ["decimal"]:
        return "float64"

# - - - - - - - - - - - - - -
# Helpers for writing tables
# - - - - - - - - - - - - - -

def get_group_mean_df(df, stat_groups, run_id):
    """
    This function creates group_mean table to write to athena
    Inputs: 
        df: data frame 
        stat_groups: list of stat_groups
    """
    unique_groups = (
            df.drop_duplicates(subset=stat_groups, keep="first")
            .reset_index(drop=True)
            .assign(
                rolling_window=lambda df: pd.to_datetime(df["rolling_window"], format="%Y%m").dt.date
            )
        )

    groups_string_col = "_".join(map(str, stat_groups))
    suffixes = ["mean_price", "mean_price_per_sqft"]

    cols_to_write_means = stat_groups + [
        f"sv_{suffix}_{groups_string_col}" for suffix in suffixes
    ]
    rename_dict = {f"sv_{suffix}_{groups_string_col}": f"{suffix}" for suffix in suffixes}

    df_means = (
        unique_groups[cols_to_write_means]
        .rename(columns=rename_dict)
        .assign(
            run_id=run_id, group=lambda df: df[stat_groups].astype(str).apply("_".join, axis=1)
        )
        .drop(columns=stat_groups)
    )

    return df_means


def get_parameter_df(df_to_write, df_ingest, iso_forest_cols, 
                     stat_groups, dev_bounds, short_term_thresh, run_id):
    sales_flagged = df_to_write.shape[0]
    earliest_sale_ingest = df_ingest.meta_sale_date.min()
    latest_sale_ingest = df_ingest.meta_sale_date.max()
    short_term_owner_threshold = short_term_thresh
    iso_forest_cols = iso_forest_cols
    stat_groups = stat_groups
    dev_bounds = dev_bounds

    parameter_dict_to_df = {
        "run_id": [run_id],
        "sales_flagged": [sales_flagged],
        "earliest_data_ingest": [earliest_sale_ingest],
        "latest_data_ingest": [latest_sale_ingest],
        "short_term_owner_threshold": [short_term_owner_threshold],
        "iso_forest_cols": [iso_forest_cols],
        "stat_groups": [stat_groups],
        "dev_bounds": [dev_bounds],
    }

    df_parameters = pd.DataFrame(parameter_dict_to_df)
    
    return df_parameters


def get_metadata_df(run_id, timestamp):
    commit_sha = sp.getoutput("git rev-parse HEAD")

    metadata_dict_to_df = {
        "run_id": [run_id],
        "long_commit_sha": commit_sha,
        "short_commit_sha": commit_sha[0:8],
        "run_timestamp": timestamp,
        "run_type": "initial_flagging",
        "flagging_hash": "",
    }

    df_metadata = pd.DataFrame(metadata_dict_to_df)

    return df_metadata



# These four could potentially be combined into one function
def write_to_flag_table(df, s3_warehouse_bucket_path, run_id):
    file_name = run_id + ".parquet"
    s3_file_path = os.path.join(s3_warehouse_bucket_path, "sale", "flag", file_name)
    wr.s3.to_parquet(df=df, path=s3_file_path)

def write_to_metadata_table(df, s3_warehouse_bucket_path, run_id):
    file_name = run_id + ".parquet"
    s3_file_path = os.path.join(s3_warehouse_bucket_path, "sale", "metadata", file_name)
    wr.s3.to_parquet(df=df, path=s3_file_path)

def write_to_parameter_table(df, s3_warehouse_bucket_path, run_id):
    file_name = run_id + ".parquet"
    s3_file_path = os.path.join(s3_warehouse_bucket_path, "sale", "parameter", file_name)
    wr.s3.to_parquet(df=df, path=s3_file_path)

def write_to_group_mean_table(df, s3_warehouse_bucket_path, run_id):
    file_name = run_id + ".parquet"
    s3_file_path = os.path.join(s3_warehouse_bucket_path, "sale", "group_mean", file_name)
    wr.s3.to_parquet(df=df, path=s3_file_path)