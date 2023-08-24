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
from random_word import RandomWords


def months_back(date_str, num_months):
    """
    This function returns data from the earliest date needed to be pulled
    in order to calculate all of the flagging operations with the rolling
    window operation.

    Inputs:
        date_str: string that represents earliest date to flag.
    Outputs:
        outputs the earliest date to pull from sql for rolling window
        operation
    """
    # Parse the date string to a datetime object
    given_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

    # Handle the month subtraction
    new_month = given_date.month - num_months
    if new_month < 1:
        new_month += 12
        new_year = given_date.year - 1
    else:
        new_year = given_date.year

    # Create the new date with the first day of the month
    result_date = given_date.replace(year=new_year, month=new_month, day=1)
    return result_date.strftime("%Y-%m-%d")


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
    Inputs:
        df: df flagged with manuesto flagging methodology
        start_date: a limit on how early we flag sales from
        exempt_data: data of class exempt
        manual_update: whether or not manual_update.py is using this script,
                       if True, adds a versioning capability.
    Outputs:
        df: reduced data frame in format of sales.flag table
        run_id: unique run_id used for metadata. etc.
        timestamp: unique timestamp for metadata
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
            sv_is_outlier=lambda df: df["sv_is_autoval_outlier"]
            | df["sale_filter_is_outlier"],
            # Incorporate PTAX in sv_outlier_type
            sv_outlier_type=lambda df: np.where(
                (df["sv_outlier_type"] == "Not outlier") & df["sale_filter_is_outlier"],
                "PTAX-203 flag",
                df["sv_outlier_type"],
            ),
        )
        .assign(
            # Change sv_is_outlier to binary
            sv_is_outlier=lambda df: (df["sv_outlier_type"] != "Not outlier").astype(
                int
            ),
            # PTAX-203 binary
            sv_is_ptax_outlier=lambda df: np.where(
                df["sv_outlier_type"] == "PTAX-203 flag", 1, 0
            ),
            # Heuristics flagging binary column
            sv_is_heuristic_outlier=lambda df: np.where(
                (df["sv_outlier_type"] != "PTAX-203 flag") & (df["sv_is_outlier"] == 1),
                1,
                0,
            ),
        )
    )

    # Manually impute ex values as non-outliers
    exempt_to_append = exempt_data.meta_sale_document_num.reset_index().drop(
        columns="index"
    )
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
    timestamp = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime(
        "%Y-%m-%d_%H:%M"
    )
    run_id = timestamp + "-" + random_word_id

    # Control flow for incorporating versioning
    dynamic_assignment = {
        "run_id": run_id,
        "rolling_window": lambda df: pd.to_datetime(
            df["rolling_window"], format="%Y%m"
        ).dt.date,
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
    pandas dtypes, using athena parquet metadata,
    used within a dictionary comprehension.
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
    This function creates group_mean table to write to athena. This allows
    us to trace back why some sales may have been flagged within our flagging model
    Inputs:
        df: data frame
        stat_groups: list of stat_groups used in flagging model
        run_id: unique run_id of script
    Outputs:
        df: dataframe that is ready to be written to athena as a parquet
    """
    unique_groups = (
        df.drop_duplicates(subset=stat_groups, keep="first")
        .reset_index(drop=True)
        .assign(
            rolling_window=lambda df: pd.to_datetime(
                df["rolling_window"], format="%Y%m"
            ).dt.date
        )
    )

    groups_string_col = "_".join(map(str, stat_groups))
    suffixes = ["mean_price", "mean_price_per_sqft"]

    cols_to_write_means = stat_groups + [
        f"sv_{suffix}_{groups_string_col}" for suffix in suffixes
    ]
    rename_dict = {
        f"sv_{suffix}_{groups_string_col}": f"{suffix}" for suffix in suffixes
    }

    df_means = (
        unique_groups[cols_to_write_means]
        .rename(columns=rename_dict)
        .assign(
            run_id=run_id,
            group=lambda df: df[stat_groups].astype(str).apply("_".join, axis=1),
        )
        .drop(columns=stat_groups)
    )

    return df_means


def get_parameter_df(
    df_to_write,
    df_ingest,
    iso_forest_cols,
    stat_groups,
    dev_bounds,
    short_term_thresh,
    run_id,
):
    """
    This functions extracts relevant data to write a parameter table,
    which tracks important information about the flagging run.
    Inputs:
        df_to_write: The final table used to write data to the sales.flag table
        df_ingest: raw data read in to perform flagging
        iso_forest_cols: columns used in iso_forest model in mansueto's flagging model
        stat_groups: which groups were used for mansueto's flagging model
        dev_bounds: standard devation bounds to catch outliers
        short_term_thresh: short-term threshold for mansueto's flagging model
        run_id: unique run_id to flagging program run
    Outputs:
        df_parameters: parameters table associated with flagging run
    """
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


def get_metadata_df(run_id, timestamp, run_type, commit_sha, flagging_hash=""):
    """
    Function creates a table to be written to s3 with a unique set of
    metadata for the flagging run
    Inputs:
        run_id: unique run_id for flagging run
        timestamp: unique timestamp for program run
    Outputs:
        df_metadata: table to be written to s3
    """

    metadata_dict_to_df = {
        "run_id": [run_id],
        "long_commit_sha": commit_sha,
        "short_commit_sha": commit_sha[0:8],
        "run_timestamp": timestamp,
        "run_type": run_type,
        "flagging_hash": flagging_hash,
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

    # Load in glue job parameters
    args = getResolvedOptions(
        sys.argv,
        [
            "region_name",
            "s3_staging_dir",
            "aws_s3_warehouse_bucket",
            "s3_glue_bucket",
            "stat_groups",
            "iso_forest",
            "dev_bounds",
        ],
    )

    # Define pattern to match flagging script in s3
    pattern = "^flagging_([0-9a-z]{6})\.py$"

    # List objects in the S3 bucket and prefix
    objects = s3.list_objects(
        Bucket=args["s3_glue_bucket"], Prefix="scripts/sales-val/"
    )

    # Read in flagging script
    for obj in objects["Contents"]:
        key = obj["Key"]
        filename = os.path.basename(key)
        local_path = f"/tmp/{key.split('/')[-1]}"
        if re.match(pattern, filename):
            # If a match is found, download the file
            s3.download_file(args["s3_glue_bucket"], key, local_path)
            hash_to_save = re.search(pattern, filename).group(1)

            # Load the python flagging script
            exec(open(local_path).read())
            break

    # Connect to athena
    conn = connect(
        s3_staging_dir=args["s3_staging_dir"], region_name=args["region_name"]
    )

    """
    This query grabs all data needed to flag unflagged values.
    It takes 11 months of data prior to the earliest unflagged sale up
    to the monthly data of the latest unflagged sale
    """
    SQL_QUERY = """
    WITH NA_Dates AS (
        SELECT
            MIN(DATE_TRUNC('MONTH', sale.sale_date)) - INTERVAL '11' MONTH AS StartDate,
            MAX(DATE_TRUNC('MONTH', sale.sale_date)) AS EndDate
        FROM default.vw_card_res_char res
        INNER JOIN default.vw_pin_sale sale
            ON sale.pin = res.pin
            AND sale.year = res.year
        LEFT JOIN sale.flag flag
            ON flag.meta_sale_document_num = sale.doc_no
        WHERE flag.sv_is_outlier IS NULL
            AND sale.sale_date >= DATE '2021-01-01'
            AND NOT sale.is_multisale
            AND NOT res.pin_is_multicard
    )
    SELECT 
        sale.sale_price AS meta_sale_price,
        sale.sale_date AS meta_sale_date,
        sale.doc_no AS meta_sale_document_num,
        sale.seller_name AS meta_sale_seller_name,
        sale.buyer_name AS meta_sale_buyer_name,
        sale.sale_filter_is_outlier,
        res.class AS class,
        res.township_code AS township_code,
        res.year AS year,
        res.pin AS pin,
        res.char_bldg_sf AS char_bldg_sf,
        flag.run_id,
        flag.sv_is_outlier,
        flag.sv_is_ptax_outlier,
        flag.sv_is_heuristic_outlier,
        flag.sv_outlier_type
    FROM default.vw_card_res_char res
    INNER JOIN default.vw_pin_sale sale
        ON sale.pin = res.pin
        AND sale.year = res.year
    LEFT JOIN sale.flag flag
        ON flag.meta_sale_document_num = sale.doc_no
    INNER JOIN NA_Dates
        ON sale.sale_date BETWEEN NA_Dates.StartDate AND NA_Dates.EndDate
    WHERE NOT sale.is_multisale
    AND NOT res.pin_is_multicard
    """

    SQL_QUERY_SALES_VAL = """
    SELECT *
    FROM sale.flag
    """

    # ----
    # Execute queries and return as pandas df
    # ----

    # Instantiate cursor
    cursor = conn.cursor()

    # Get data needed to flag non-flagged data
    cursor.execute(SQL_QUERY)
    metadata = cursor.description
    df_ingest_full = as_pandas(cursor)
    df = df_ingest_full

    # Skip rest of script if no new unflagged sales
    if df_ingest_full.sv_outlier_type.isna().sum() == 0:
        print("WARNING: No new sales to flag")
    else:
        # Grab existing sales val table for later join
        cursor.execute(SQL_QUERY_SALES_VAL)
        df_ingest_sales_val = as_pandas(cursor)
        df_sales_val = df_ingest_sales_val

        # Data cleaning
        df = df.astype({col[0]: sql_type_to_pd_type(col[1]) for col in metadata})

        # Exempt sale handling
        exempt_data = df[df["class"] == "EX"]
        df = df[df["class"] != "EX"]

        # Create rolling window
        df_to_flag = add_rolling_window(df)

        # Parse glue args
        stat_groups_list = args["stat_groups"].split(",")
        iso_forest_list = args["iso_forest"].split(",")
        dev_bounds_list = args["dev_bounds"].split(",")
        dev_bounds_tuple = tuple(map(int, args["dev_bounds"].split(",")))

        # Flag Outliers
        df_flagged = go(
            df=df_to_flag,
            groups=tuple(stat_groups_list),
            iso_forest_cols=iso_forest_list,
            dev_bounds=dev_bounds_tuple,
        )

        # Finish flagging
        df_flagged_final, run_id, timestamp = finish_flags(
            df=df_flagged,
            start_date="2021-01-01",
            exempt_data=exempt_data,
            manual_update=False,
        )

        # Filter to keep only flags not already present in the flag table
        rows_to_append = df_flagged_final[
            ~df_flagged_final["meta_sale_document_num"].isin(
                df_sales_val["meta_sale_document_num"]
            )
        ].reset_index(drop=True)

        # Write to flag table
        write_to_table(
            df=rows_to_append,
            table_name="flag",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )

        # Write to parameter table
        df_parameters = get_parameter_df(
            df_to_write=rows_to_append,
            df_ingest=df_ingest_full,
            iso_forest_cols=iso_forest_list,
            stat_groups=stat_groups_list,
            dev_bounds=dev_bounds_list,
            short_term_thresh=SHORT_TERM_OWNER_THRESHOLD,
            run_id=run_id,
        )

        write_to_table(
            df=df_parameters,
            table_name="parameter",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )

        # Write to group_mean table
        df_write_group_mean = get_group_mean_df(
            df=df_flagged, stat_groups=stat_groups_list, run_id=run_id
        )

        write_to_table(
            df=df_write_group_mean,
            table_name="group_mean",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )

        # Write to metadata table
        job_name = "sales_val_flagging"
        response = glue.get_job(JobName=job_name)
        commit_sha = response["Job"]["SourceControlDetails"]["LastCommitId"]

        # Write to metadata table
        df_metadata = get_metadata_df(
            run_id=run_id,
            timestamp=timestamp,
            run_type="recurring",
            commit_sha=commit_sha,
            flagging_hash=hash_to_save,
        )

        write_to_table(
            df=df_metadata,
            table_name="metadata",
            s3_warehouse_bucket_path=args["aws_s3_warehouse_bucket"],
            run_id=run_id,
        )
