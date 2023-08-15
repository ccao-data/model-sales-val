from manual_flagging.src import flagging_rolling as flg_model
from manual_flagging.src.flagging_rolling import SHORT_TERM_OWNER_THRESHOLD
from manual_flagging.src import flagging_utils as flg
import awswrangler as wr
import datetime
import numpy as np
import os
import pandas as pd
import pytz
from pyathena import connect
from pyathena.pandas.util import as_pandas
from random_word import RandomWords
import subprocess as sp
import yaml

# Set working to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "manual_flagging"))

# Set timezone for run_id
chicago_tz = pytz.timezone("America/Chicago")

# Inputs yaml as inputs
with open(os.path.join("yaml", "inputs_recurring.yaml"), "r") as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Connect to athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION")
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
    df_ingest_flag = as_pandas(cursor)
    df_flag_table = df_ingest_flag

    # Data cleaning
    df = df.astype({col[0]: flg.sql_type_to_pd_type(col[1]) for col in metadata})

    # Exempt sale handling
    exempt_data = df[df["class"] == "EX"]
    df = df[df["class"] != "EX"]

    # Create rolling window
    df_to_flag = flg.add_rolling_window(df)

    # Flag Outliers
    df_flagged = flg_model.go(
        df=df_to_flag,
        groups=tuple(inputs["stat_groups"]),
        iso_forest_cols=inputs["iso_forest"],
        dev_bounds=tuple(inputs["dev_bounds"]),
    )

    # Finish flagging
    df_flagged_final, run_id, timestamp = flg.finish_flags(
        df=df_flagged, start_date="2021-01-01", exempt_data=exempt_data,
        manual_update=False
    )

    # Filter to keep only flags not already present in the flag table
    rows_to_append = df_flagged_final[
        ~df_flagged_final["meta_sale_document_num"].isin(df_flag_table["meta_sale_document_num"])
    ].reset_index(drop=True)

    # Write to flag table
    flg.write_to_flag_table(
        df=rows_to_append,
        s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        run_id=run_id,
    )
    # Write to parameter table
    df_parameters = flg.get_parameter_df(
        df_to_write=rows_to_append,
        df_ingest=df_ingest_full,
        iso_forest_cols=inputs["iso_forest"],
        stat_groups=inputs["stat_groups"],
        dev_bounds=inputs["dev_bounds"],
        short_term_thresh=SHORT_TERM_OWNER_THRESHOLD,
        run_id=run_id,
    )

    flg.write_to_parameter_table(
        df=df_parameters,
        s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        run_id=run_id,
    )

    # Write to group_mean table
    df_write_group_mean = flg.get_group_mean_df(
        df=df_flagged, stat_groups=inputs["stat_groups"], run_id=run_id
    )

    flg.write_to_group_mean_table(
        df=df_write_group_mean,
        s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        run_id=run_id,
    )

    # Write to metadata table
    df_metadata = flg.get_metadata_df(run_id=run_id, timestamp=timestamp, run_type='recurring')

    flg.write_to_metadata_table(
        df=df_metadata,
        s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        run_id=run_id,
    )