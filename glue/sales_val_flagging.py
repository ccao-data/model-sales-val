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
    result_date = datetime.datetime.strptime(date_str, "%Y-%m-%d") - relativedelta(
        months=num_months
    )

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
        .assign(rolling_window=lambda df: df["rolling_window"].dt.to_period("M"))
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

    group_string = "_".join(groups)

    if not condos:
        df["sv_ind_ptax_flag_w_high_price"] = df["ptax_flag_original"] & (
            (df[f"sv_price_deviation_{group_string}"] >= ptax_sd[1])
        )

        df["sv_ind_ptax_flag_w_high_price_sqft"] = df["ptax_flag_original"] & (
            (df[f"sv_price_per_sqft_deviation_{group_string}"] >= ptax_sd[1])
        )

        df["sv_ind_ptax_flag_w_low_price"] = df["ptax_flag_original"] & (
            (df[f"sv_price_per_sqft_deviation_{group_string}"] <= -ptax_sd[0])
        )

        df["sv_ind_ptax_flag_w_low_price_sqft"] = df["ptax_flag_original"] & (
            (df[f"sv_price_per_sqft_deviation_{group_string}"] <= -ptax_sd[0])
        )

    else:
        df["sv_ind_ptax_flag_w_high_price"] = df["ptax_flag_original"] & (
            (df[f"sv_price_deviation_{group_string}"] >= ptax_sd[1])
        )

        df["sv_ind_ptax_flag_w_low_price"] = df["ptax_flag_original"] & (
            (df[f"sv_price_deviation_{group_string}"] <= -ptax_sd[0])
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
        df, filtered_groups[stat_groups], on=stat_groups, how="left", indicator=True
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
                    row["_merge"] == "both" and reason_ind_col in group_thresh_price_fix
                )
            ):
                row[f"sv_outlier_reason{len(reasons_added) + 1}"] = current_reason
                reasons_added.add(current_reason)  # Add current reason to the set
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
        sv_is_ptax_outlier=lambda df: (df["sv_is_outlier"] == True)
        & (df["sv_ind_ptax_flag"] == 1),
        sv_is_heuristic_outlier=lambda df: (~df["sv_ind_ptax_flag"] == 1)
        & (df["sv_is_outlier"] == True),
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
            df[sales_to_write_filter["column"]].isin(sales_to_write_filter["values"])
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
        np.random.choice(left["adjective"]) + "-" + np.random.choice(right["person"])
    )
    timestamp = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime(
        "%Y-%m-%d_%H:%M"
    )
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

    # Finalize to write to sale.flag table
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


def write_to_table(df, table_name, s3_warehouse_bucket_path, run_id):
    """
    This function writes a parquet file to s3 which will either create or append
    to a table in athena.
    Inputs:
        df: dataframe ready to be written
        table_name: which table the parquet will be written to
        s3_warehouse_bucket_path: s3 bucket
        run_id: unique run_id of the script
    """
    file_name = run_id + ".parquet"
    s3_file_path = os.path.join(s3_warehouse_bucket_path, "sale", table_name, file_name)
    wr.s3.to_parquet(df=df, path=s3_file_path)


if __name__ == "__main__":
    from awsglue.utils import getResolvedOptions

    # Create clients
    s3 = boto3.client("s3")
    glue = boto3.client("glue")

    # Set timezone for run_id
    chicago_tz = pytz.timezone("America/Chicago")

    # Load in Glue job parameters
    args = getResolvedOptions(
        sys.argv,
        [
            "region_name",
            "s3_staging_dir",
            "aws_s3_warehouse_bucket",
            "s3_glue_bucket",
            "s3_prefix",
            "stat_groups",
            "rolling_window_num",
            "time_frame_start",
            "iso_forest",
            "min_groups_threshold",
            "dev_bounds",
            "commit_sha",
            "ptax_sd",
            "sale_flag_table",
        ],
    )

    # Load the python flagging script.
    # We should refactor this to use a wheel that Glue can install. See:
    # https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python-libraries.html#addl-python-modules-support
    local_path = f"/tmp/flagging.py"
    s3.download_file(
        args["s3_glue_bucket"],
        os.path.join(args["s3_prefix"], "flagging.py"),
        local_path,
    )
    exec(open(local_path).read())

    # Connect to Athena
    conn = connect(
        s3_staging_dir=args["s3_staging_dir"], region_name=args["region_name"]
    )

    """
    This query grabs all data needed to flag unflagged values.
    It takes 11 months of data prior to the earliest unflagged sale up
    to the monthly data of the latest unflagged sale
    """

    rolling_window_num_sql = str(int(args["rolling_window_num"]) - 1)

    SQL_QUERY = f"""
    WITH CombinedData AS (
        SELECT
            'res_char' AS source_table,
            'res' AS indicator, -- Indicator column for 'res'
            res.class AS class,
            res.township_code AS township_code,
            res.year AS year,
            res.pin AS pin,
            res.char_bldg_sf AS char_bldg_sf,
            res.pin_is_multicard
        FROM default.vw_card_res_char res
        WHERE res.class IN (
            '202', '203', '204', '205', '206', '207', '208', '209',
            '210', '211', '212', '218', '219', '234', '278', '295'
        )

        UNION ALL

        SELECT
            'condo_char' AS source_table,
            'condo' AS indicator, -- Indicator column for 'condo'
            condo.class AS class,
            condo.township_code AS township_code,
            condo.year AS year,
            condo.pin AS pin,
            NULL AS char_bldg_sf,
            FALSE AS pin_is_multicard
        FROM default.vw_pin_condo_char condo
        WHERE condo.class IN ('297', '299', '399')
        AND NOT condo.is_parking_space
        AND NOT condo.is_common_area
        AND NOT condo.is_question_garage_unit
    ),
    NA_Dates AS (
        SELECT
            MIN(DATE_TRUNC('MONTH', sale.sale_date)) - INTERVAL '{rolling_window_num_sql}' MONTH AS StartDate,
            MAX(date_add('day', -1, date_add('month', 1, DATE_TRUNC('MONTH', sale.sale_date)))) AS EndDate
        FROM CombinedData data
        INNER JOIN default.vw_pin_sale sale
            ON sale.pin = data.pin
            AND sale.year = data.year
        LEFT JOIN {args["sale_flag_table"]} flag
            ON flag.meta_sale_document_num = sale.doc_no
        WHERE flag.sv_is_outlier IS NULL
        AND sale.sale_date >= DATE '{args["time_frame_start"]}'
        AND NOT sale.is_multisale
        AND (NOT data.pin_is_multicard OR data.source_table = 'condo_char')
    )
    SELECT
        sale.sale_price AS meta_sale_price,
        sale.sale_date AS meta_sale_date,
        sale.doc_no AS meta_sale_document_num,
        sale.seller_name AS meta_sale_seller_name,
        sale.buyer_name AS meta_sale_buyer_name,
        sale.sale_filter_ptax_flag AS ptax_flag_original,
        data.class,
        data.township_code,
        data.year,
        data.pin,
        data.char_bldg_sf,
        data.indicator, -- Selecting the indicator column
        flag.run_id,
        flag.sv_is_outlier,
        flag.sv_is_ptax_outlier,
        flag.sv_is_heuristic_outlier,
        flag.sv_outlier_type
    FROM CombinedData data
    INNER JOIN default.vw_pin_sale sale
        ON sale.pin = data.pin
        AND sale.year = data.year
    LEFT JOIN {args["sale_flag_table"]} flag
        ON flag.meta_sale_document_num = sale.doc_no
    INNER JOIN NA_Dates
        ON sale.sale_date BETWEEN NA_Dates.StartDate AND NA_Dates.EndDate
    WHERE NOT sale.is_multisale
    AND NOT sale.sale_filter_same_sale_within_365
    AND NOT sale.sale_filter_less_than_10k
    AND NOT sale.sale_filter_deed_type
    AND (NOT data.pin_is_multicard OR data.source_table = 'condo_char')
    AND data.class IN (
        '202', '203', '204', '205', '206', '207', '208', '209',
        '210', '211', '212', '218', '219', '234', '278', '295',
        '297', '299', '399'
    )
    """

    # -------------------------------------------------------------------------
    # Execute queries and return as pandas df
    # -------------------------------------------------------------------------

    # Instantiate cursor
    cursor = conn.cursor()

    # Get data needed to flag non-flagged data
    cursor.execute(SQL_QUERY)
    metadata = cursor.description
    df_ingest_full = as_pandas(cursor)
    df = df_ingest_full

    # Filter the dataframe to look at sales we are interested in flagging,
    # not prior rolling window data
    filtered_df = df_ingest_full[
        df_ingest_full["meta_sale_date"] >= args["time_frame_start"]
    ]

    # Skip rest of script if no new unflagged sales
    if filtered_df.sv_outlier_type.isna().sum() == 0:
        print("WARNING: No new sales to flag")
    else:
        # Make sure None types aren't utilized in type conversion
        conversion_dict = {
            col[0]: sql_type_to_pd_type(col[1])
            for col in metadata
            if sql_type_to_pd_type(col[1]) is not None
        }
        df = df.astype(conversion_dict)

        df["ptax_flag_original"].fillna(False, inplace=True)

        # Separate res and condo sales based on the indicator column
        df_res = df[df["indicator"] == "res"].reset_index(drop=True)
        df_condo = df[df["indicator"] == "condo"].reset_index(drop=True)

        # Create rolling windows
        df_res_to_flag = add_rolling_window(
            df_res, num_months=int(args["rolling_window_num"])
        )
        df_condo_to_flag = add_rolling_window(
            df_condo, num_months=int(args["rolling_window_num"])
        )

        # Parse glue args
        stat_groups_list = args["stat_groups"].split(",")
        iso_forest_list = args["iso_forest"].split(",")
        dev_bounds_list = list(map(int, args["dev_bounds"].split(",")))
        dev_bounds_tuple = tuple(map(int, args["dev_bounds"].split(",")))
        ptax_sd_list = list(map(int, args["ptax_sd"].split(",")))

        # Create condo stat groups. Condos are all collapsed into a single
        # class, since there are very few 297s or 399s
        condo_stat_groups = stat_groups_list.copy()
        condo_stat_groups.remove("class")

        # Flag outliers using the main flagging model
        df_res_flagged = go(
            df=df_res_to_flag,
            groups=tuple(stat_groups_list),
            iso_forest_cols=iso_forest_list,
            dev_bounds=dev_bounds_tuple,
            condos=False,
        )

        # Discard any flags with a group size under the threshold
        df_res_flagged_updated = group_size_adjustment(
            df=df_res_flagged,
            stat_groups=stat_groups_list,
            min_threshold=int(args["min_groups_threshold"]),
            condos=False,
        )

        # Flag condo outliers, here we remove price per sqft as an input
        # for the isolation forest model since condos don't have a unit sqft
        condo_iso_forest = iso_forest_list.copy()
        condo_iso_forest.remove("sv_price_per_sqft")

        df_condo_flagged = go(
            df=df_condo_to_flag,
            groups=tuple(condo_stat_groups),
            iso_forest_cols=condo_iso_forest,
            dev_bounds=dev_bounds_tuple,
            condos=True,
        )

        # Discard any flags with a group size under the threshold
        df_condo_flagged_updated = group_size_adjustment(
            df=df_condo_flagged,
            stat_groups=condo_stat_groups,
            min_threshold=int(args["min_groups_threshold"]),
            condos=True,
        )

        # Update the PTAX flag column with an additional std dev conditional
        df_res_flagged_updated_ptax = ptax_adjustment(
            df=df_res_flagged_updated,
            groups=stat_groups_list,
            ptax_sd=ptax_sd_list,
            condos=False,
        )

        # Update the PTAX flag column with an additional std dev conditional
        df_condo_flagged_updated_ptax = ptax_adjustment(
            df=df_condo_flagged_updated,
            groups=condo_stat_groups,
            ptax_sd=ptax_sd_list,
            condos=True,
        )

        df_flagged_ptax_merged = pd.concat(
            [df_res_flagged_updated_ptax, df_condo_flagged_updated_ptax]
        ).reset_index(drop=True)

        # Finish flagging and subset to write to flag table
        df_flagged_final, run_id, timestamp = finish_flags(
            df=df_flagged_ptax_merged,
            start_date=args["time_frame_start"],
            manual_update=False,
        )

        # Find rows in df_ingest_full with sv_is_outlier having a value
        existing_flags = df_ingest_full.dropna(subset=["sv_is_outlier"])[
            "meta_sale_document_num"
        ]

        # Filter out rows from df_flagged_final that are in the above subset
        rows_to_append = df_flagged_final[
            ~df_flagged_final["meta_sale_document_num"].isin(existing_flags)
        ].reset_index(drop=True)

        # Write to sale.flag table
        write_to_table(
            df=rows_to_append,
            table_name="flag",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )

        # Write to sale.parameter table
        df_parameters = get_parameter_df(
            df_to_write=rows_to_append,
            df_ingest=df_ingest_full,
            iso_forest_cols=iso_forest_list,
            res_stat_groups=stat_groups_list,
            condo_stat_groups=condo_stat_groups,
            dev_bounds=dev_bounds_list,
            ptax_sd=ptax_sd_list,
            rolling_window=int(args["rolling_window_num"]),
            date_floor=args["time_frame_start"],
            short_term_thresh=SHORT_TERM_OWNER_THRESHOLD,
            min_group_thresh=int(args["min_groups_threshold"]),
            run_id=run_id,
        )

        # Standardize dtypes to prevent Athena errors
        df_parameters = modify_dtypes(df_parameters)

        write_to_table(
            df=df_parameters,
            table_name="parameter",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )

        # Write to sale.group_mean table
        df_res_group_mean = get_group_mean_df(
            df=df_res_flagged, stat_groups=stat_groups_list, run_id=run_id, condos=False
        )

        # Write to sale.group_mean table
        df_condo_group_mean = get_group_mean_df(
            df=df_condo_flagged,
            stat_groups=condo_stat_groups,
            run_id=run_id,
            condos=True,
        )

        df_group_mean_merged = pd.concat(
            [df_res_group_mean, df_condo_group_mean]
        ).reset_index(drop=True)

        write_to_table(
            df=df_group_mean_merged,
            table_name="group_mean",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )

        # Write to sale.metadata table
        job_name = "sales_val_flagging"
        response = glue.get_job(JobName=job_name)
        commit_sha = args["commit_sha"]

        # Write to sale.metadata table
        df_metadata = get_metadata_df(
            run_id=run_id,
            timestamp=timestamp,
            run_type="recurring",
            commit_sha=commit_sha,
        )

        write_to_table(
            df=df_metadata,
            table_name="metadata",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )
