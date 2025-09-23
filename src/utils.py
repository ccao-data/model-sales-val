import datetime
import os

import awswrangler as wr
import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta


def months_back(date_str, num_months):
    """
    This function returns data from the earliest date needed to be pulled
    in order to calculate all of the flagging operations with the rolling
    window operation.

    Inputs:
        date_str: string that represents earliest date to flag.
        num_months: number that indicates how many months back
            data will be pulled for rolling window
    Outputs:
        str: a date in string format set to the first day of the month
            that is num_months months back from date_str
    """
    # Parse the date string to a datetime object and subtract the months
    result_date = datetime.datetime.strptime(
        date_str, "%Y-%m-%d"
    ) - relativedelta(months=num_months)

    # Set the day to 1
    result_date = result_date.replace(day=1)

    return result_date.strftime("%Y-%m-%d")


def add_rolling_window(df, num_months):
    """
    This function implements a rolling window logic such that
    the data is formatted for the flagging script to correctly
    run the year-long window grouping for each obs.

    Inputs:
        df: dataframe of sales that we need to flag, data should be N months back
            from the earliest unflagged sale in order for the rolling window logic to work
    Outputs:
        df: dataframe that has exploded each observation into N observations with a N months distinct
            rolling window columns
    """
    max_date = df["meta_sale_date"].max()
    df = (
        # Creates dt column with 12 month dates
        df.assign(
            rolling_window=df["meta_sale_date"].apply(
                lambda x: pd.date_range(start=x, periods=num_months, freq="M")
            )
        )
        # Expand rolling_windows dates to individual rows
        .explode("rolling_window")
        # Tag original observations
        .assign(
            original_observation=lambda df: (
                df["meta_sale_date"].dt.month == df["rolling_window"].dt.month
            )
            & (df["meta_sale_date"].dt.year == df["rolling_window"].dt.year)
        )
        # Simplify to month level
        .assign(
            rolling_window=lambda df: df["rolling_window"].dt.to_period("M")
        )
        # Filter such that rolling_window isn't extrapolated into future, we are concerned with historic and present-month data
        .loc[lambda df: df["rolling_window"] <= max_date.to_period("M")]
        # Back to float for flagging script
        .assign(
            rolling_window=lambda df: df["rolling_window"]
            .apply(lambda x: x.strftime("%Y%m"))
            .astype(int),
            meta_sale_price_original=lambda df: df["meta_sale_price"],
        )
    )

    return df


def ptax_adjustment(df, groups, ptax_sd, condos: bool):
    """
    This function manually applies a ptax adjustment, keeping only
    ptax flags that are outside of a certain standard deviation
    range in terms of raw price or price per sqft. It creates the
    new column and preserves the old ptax column.

    Inputs:
        df: dataframe after flagging has been done
        groups: stat groups used for outlier classification
        ptax_sd: a list that look like this - [low sd, high sd]
            - both values should be positive
    Outputs:
        df: ptax adjusted dataframe
    """

    if not condos:
        df["sv_ind_ptax_flag_w_high_price"] = df["ptax_flag_original"] & (
            df["sv_price_deviation"] >= ptax_sd[1]
        )

        df["sv_ind_ptax_flag_w_high_price_sqft"] = df["ptax_flag_original"] & (
            df["sv_price_per_sqft_deviation"] >= ptax_sd[1]
        )

        df["sv_ind_ptax_flag_w_low_price"] = df["ptax_flag_original"] & (
            df["sv_price_deviation"] <= -ptax_sd[0]
        )

        df["sv_ind_ptax_flag_w_low_price_sqft"] = df["ptax_flag_original"] & (
            df["sv_price_per_sqft_deviation"] <= -ptax_sd[0]
        )

    else:
        df["sv_ind_ptax_flag_w_high_price"] = df["ptax_flag_original"] & (
            df["sv_price_deviation"] >= ptax_sd[1]
        )

        df["sv_ind_ptax_flag_w_low_price"] = df["ptax_flag_original"] & (
            df["sv_price_deviation"] <= -ptax_sd[0]
        )

    df["sv_ind_ptax_flag"] = df["ptax_flag_original"].astype(int)

    return df


def classify_outliers(df, stat_groups: list, min_threshold):
    """
    This function does the following:

    1. We use all of the indicator columns created by outlier_type() in the
    Mansueto flagging script to populate our sv_outlier_reason1, sv_outlier_reason2,
    and sv_outlier_reason3 fields. We populate them first with ptax, then price, then char
    reasons.

    2. Implement our group threshold requirement. In the statistical flagging process, if
    the group a sale belongs too is below N=30 then we want to manually set these flags to
    non-outlier status, even if they were flagged in the mansueto script. This requirement
    is bypasses for ptax outliers and raw price threshold outliers - we don't care about
    group threshold in this case.

    Inputs:
        df: The data right after we perform the flagging script (go()), when the exploded
            rolling window hasn't been reduced.
        stat_groups: stat groups we are using for the groups within which we flag outliers
        min_threshold: at which group size we want to manually set values to 'Not Outlier'
        condos: boolean that tells the function to work with condos or res
    Outputs:
        df: dataframe with newly manually adjusted outlier values.

    """

    group_counts = df.groupby(stat_groups).size().reset_index(name="count")
    filtered_groups = group_counts[group_counts["count"] <= min_threshold]

    # Merge df_flagged with filtered_groups on the columns to get the matching rows
    df = pd.merge(
        df,
        filtered_groups[stat_groups],
        on=stat_groups,
        how="left",
        indicator=True,
    )

    # Assign sv_outlier_reasons
    for idx in range(1, 4):
        df[f"sv_outlier_reason{idx}"] = np.nan

    outlier_type_crosswalk = {
        "sv_ind_price_high_price": "High price",
        "sv_ind_ptax_flag_w_high_price": "High price",
        "sv_ind_price_low_price": "Low price",
        "sv_ind_ptax_flag_w_low_price": "Low price",
        "sv_ind_price_high_price_sqft": "High price per square foot",
        "sv_ind_ptax_flag_w_high_price_sqft": "High price per square foot",
        "sv_ind_price_low_price_sqft": "Low price per square foot",
        "sv_ind_ptax_flag_w_low_price_sqft": "Low price per square foot",
        "sv_ind_raw_price_threshold": "Raw price threshold",
        "sv_ind_ptax_flag": "PTAX-203 Exclusion",
        "sv_ind_char_short_term_owner": "Short-term owner",
        "sv_ind_char_family_sale": "Family Sale",
        "sv_ind_char_non_person_sale": "Non-person sale",
        "sv_ind_char_statistical_anomaly": "Statistical Anomaly",
        "sv_ind_char_price_swing_homeflip": "Price swing / Home flip",
    }

    """
    During our statistical flagging process, we automatically discard
    a sale's eligibility for outlier status if the number of sales in 
    the statistical grouping is below a certain threshold. The list - 
    `group_thresh_price_fix` along with the ['_merge'] column will allow
    us to exclude these sales for the sv_is_outlier status.

    Since the `sv_is_outlier` column requires a price value, we simply
    do not assign these price outlier flags if the group number is below a certain
    threshold

    Note: This doesn't apply for sales that also have a ptax outlier status.
          In this case, we still assign the price outlier status.

          We also don't apply this threshold with sv_raw_price_threshold,
          since this is designed to be a safeguard that catches very high price
          sales that may have slipped through the cracks due to the group
          threshold requirement
    """
    group_thresh_price_fix = [
        "sv_ind_price_high_price",
        "sv_ind_price_low_price",
        "sv_ind_price_high_price_sqft",
        "sv_ind_price_low_price_sqft",
    ]

    def fill_outlier_reasons(row):
        reasons_added = set()  # Set to track reasons already added

        for reason_ind_col in outlier_type_crosswalk:
            current_reason = outlier_type_crosswalk[reason_ind_col]
            # Add a check to ensure that only specific reasons are added if _merge is not 'both'
            if (
                reason_ind_col in row
                and row[reason_ind_col]
                and current_reason
                not in reasons_added  # Check if the reason is already added
                # Apply group threshold logic: `row["_merge"]` will be `both` when the group threshold
                # is not met, but only price indicators (`group_thresh_price_fix`) should use this threshold,
                # since ptax indicators don't currently utilize group threshold logic
                and not (
                    row["_merge"] == "both"
                    and reason_ind_col in group_thresh_price_fix
                )
            ):
                row[f"sv_outlier_reason{len(reasons_added) + 1}"] = (
                    current_reason
                )
                reasons_added.add(
                    current_reason  # Add current reason to the set
                )
                if len(reasons_added) >= 3:
                    break

        return row

    df = df.apply(fill_outlier_reasons, axis=1)

    # Drop the _merge column
    df = df.drop(columns=["_merge"])

    # Assign outlier status, these are the outlier types
    # that assign a sale as an outlier
    values_to_check = {
        "High price",
        "Low price",
        "High price per square foot",
        "Low price per square foot",
        "Raw price threshold",
    }

    df["sv_is_outlier"] = np.where(
        df[[f"sv_outlier_reason{idx}" for idx in range(1, 4)]]
        .isin(values_to_check)
        .any(axis=1),
        True,
        False,
    )

    # Add group column to eventually write to athena sale.flag table. Picked up in finish_flags()
    df["group"] = df.apply(
        lambda row: "_".join([str(row[col]) for col in stat_groups]), axis=1
    )

    df = df.assign(
        # PTAX-203 binary
        sv_is_ptax_outlier=lambda df: (df["sv_is_outlier"] is True)
        & (df["sv_ind_ptax_flag"] == 1),
        sv_is_heuristic_outlier=lambda df: (~df["sv_ind_ptax_flag"] == 1)
        & (df["sv_is_outlier"] is True),
    )

    return df


def finish_flags(df, start_date, manual_update, sales_to_write_filter):
    """
    This functions
        -takes the flagged data from the mansueto code
        -removes the unneeded observations used for the rolling window calculation
        -finishes adding sales val cols for flag table upload
    Inputs:
        df: df flagged with mansueto flagging methodology
        start_date: a limit on how early we flag sales from
        manual_update: whether or not manual_update.py is using this script,
                       if True, adds a versioning capability.
        sales_to_write_filter: this param specifies a specific set of sales that we
            want to write. This input works with our sales_to_write_filter object in
            the yaml config file
    Outputs:
        df: reduced data frame in format of sales.flag table
        run_id: unique run_id used for metadata. etc.
        timestamp: unique timestamp for metadata
    """

    # Remove duplicate rows
    df = df[df["original_observation"]]
    # Discard pre-2014 data
    df = df[df["meta_sale_date"] >= start_date]

    if sales_to_write_filter["column"]:
        df = df[
            df[sales_to_write_filter["column"]].isin(
                sales_to_write_filter["values"]
            )
        ]

    cols_to_write = [
        "meta_sale_document_num",
        "meta_sale_price_original",
        "rolling_window",
        "sv_is_outlier",
        "sv_outlier_reason1",
        "sv_outlier_reason2",
        "sv_outlier_reason3",
        "sv_is_ptax_outlier",
        "ptax_flag_original",
        "sv_is_heuristic_outlier",
        "sv_price_deviation",
        "sv_price_per_sqft_deviation",
        "group",
    ]

    # Create run_id
    left = pd.read_csv(
        "https://raw.githubusercontent.com/ccao-data/data-architecture/master/dbt/seeds/ccao/ccao.adjective.csv"
    )
    right = pd.read_csv(
        "https://raw.githubusercontent.com/ccao-data/data-architecture/master/dbt/seeds/ccao/ccao.person.csv"
    )

    adj_name_combo = (
        np.random.choice(left["adjective"])
        + "-"
        + np.random.choice(right["person"])
    )
    timestamp = datetime.datetime.now(
        pytz.timezone("America/Chicago")
    ).strftime("%Y-%m-%d_%H:%M")
    run_id = timestamp + "-" + adj_name_combo

    # Control flow for incorporating versioning
    dynamic_assignment = {
        "run_id": run_id,
        "rolling_window": lambda df: pd.to_datetime(
            df["rolling_window"], format="%Y%m"
        ).dt.date,
    }

    if not manual_update:
        dynamic_assignment["version"] = 1

    df = df[cols_to_write].assign(**dynamic_assignment).reset_index(drop=True)

    return df, run_id, timestamp


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


def get_group_mean_df(df, stat_groups, run_id, condos):
    """
    This function creates group_mean table to write to athena. This allows
    us to trace back why some sales may have been flagged within our flagging model.

    It calculates the relevant group means and standard deviations.
    Inputs:
        df: data frame
        stat_groups: list of stat_groups used in flagging model
        run_id: unique run_id of script
    Outputs:
        df: dataframe that is ready to be written to athena as a parquet
    """

    # Calculate group sizes
    group_sizes = df.groupby(stat_groups).size().reset_index(name="group_size")
    df = df.merge(group_sizes, on=stat_groups, how="left")

    df["group"] = df.apply(
        lambda row: "_".join([str(row[col]) for col in stat_groups]), axis=1
    )

    if condos:
        df = df.drop_duplicates(subset=["group"])[
            ["group", "group_mean", "group_std", "group_size"]
        ].assign(run_id=run_id)
    else:
        df = df.drop_duplicates(subset=["group"])[
            [
                "group",
                "group_mean",
                "group_std",
                "group_sqft_std",
                "group_sqft_mean",
                "group_size",
            ]
        ].assign(run_id=run_id)

    return df


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


def get_parameter_df(
    df_to_write,
    df_ingest,
    run_filter,
    iso_forest_cols,
    stat_groups,
    sales_to_write_filter,
    housing_market_class_codes,
    dev_bounds,
    ptax_sd,
    rolling_window,
    time_frame,
    short_term_threshold,
    min_group_threshold,
    raw_price_threshold,
    run_id,
):
    """
    This functions extracts relevant data to write a parameter table,
    which tracks important information about the flagging run.
    Inputs:
        df_to_write: The final table used to write data to the sale.flag table
        df_ingest: raw data read in to perform flagging
        iso_forest_cols: columns used in iso_forest model in Mansueto's flagging model
        res_stat_groups: which groups were used for mansueto's flagging model
        condo_stat_groups: which groups were used for condos
        dev_bounds: standard deviation bounds to catch outliers
        ptax_sd: list of standard deviations used for ptax flagging
        rolling_window: how many months used in rolling window methodology
        date_floor: parameter specification that limits earliest flagging write
        short_term_threshold: short-term threshold for Mansueto's flagging model
        min_group_thresh: minimum group size threshold needed to flag as outlier
        raw_price_threshold: raw price threshold at which we unconditionally classify sales as outliers
        run_id: unique run_id to flagging program run
    Outputs:
        df_parameters: parameters table associated with flagging run
    """
    sales_flagged = df_to_write.shape[0]
    earliest_sale_ingest = df_ingest.meta_sale_date.min()
    latest_sale_ingest = df_ingest.meta_sale_date.max()

    parameter_dict_to_df = {
        "run_id": [run_id],
        "sales_flagged": [sales_flagged],
        "earliest_data_ingest": [earliest_sale_ingest],
        "latest_data_ingest": [latest_sale_ingest],
        "run_filter": [run_filter],
        "iso_forest_cols": [iso_forest_cols],
        "stat_groups": [stat_groups],
        "sales_to_write_filter": [sales_to_write_filter],
        "housing_market_class_codes": [housing_market_class_codes],
        "dev_bounds": [dev_bounds],
        "ptax_sd": [ptax_sd],
        "rolling_window": [rolling_window],
        "time_frame": [time_frame],
        "short_term_owner_threshold": [short_term_threshold],
        "min_group_thresh": [min_group_threshold],
        "raw_price_threshold": [raw_price_threshold],
    }

    df_parameters = pd.DataFrame(parameter_dict_to_df)

    return df_parameters


def get_metadata_df(run_id, timestamp, run_type, commit_sha, run_note):
    """
    Function creates a table to be written to s3 with a unique set of
    metadata for the flagging run
    Inputs:
        run_id: unique run_id for flagging run
        timestamp: unique timestamp for program run
        run_type: initial, manual, or Glue/recurring
        commit_sha: SHA1 hash of the commit used to run the flagging script
        flagging_hash: MD5 hash of the flagging script
    Outputs:
        df_metadata: table to be written to s3
    """

    metadata_dict_to_df = {
        "run_id": [run_id],
        "long_commit_sha": commit_sha,
        "short_commit_sha": commit_sha[0:8],
        "run_timestamp": timestamp,
        "run_type": run_type,
        "run_note": run_note,
    }

    df_metadata = pd.DataFrame(metadata_dict_to_df)

    return df_metadata


def write_to_table(df, table_name, run_id, output_environment):
    """
    This function writes a parquet file to s3 which will either create or append
    to a table in athena.
    Inputs:
        df: dataframe ready to be written
        table_name: which table the parquet will be written to
        run_id: unique run_id of the script
    """
    if output_environment == "prod":
        base_path = "s3://ccao-data-warehouse-us-east-1/sale"
    else:
        USER = os.getenv("USER")
        if not USER:
            raise ValueError(
                "$USER environment variable is unset but is required when "
                "output_environment == 'dev'"
            )
        base_path = f"s3://ccao-data-warehouse-dev-us-east-1/z_dev_{USER}_sale"

    file_name = run_id + ".parquet"
    s3_file_path = f"{base_path}/{table_name}/{file_name}"
    wr.s3.to_parquet(df=df, path=s3_file_path)
