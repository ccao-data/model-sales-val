from manual_flagging.src import flagging_rolling as flg_model
from manual_flagging.src.flagging_rolling import SHORT_TERM_OWNER_THRESHOLD
from glue import sales_val_flagging as flg
import awswrangler as wr
import os
import datetime
import numpy as np
import pandas as pd
import pytz
import subprocess as sp
import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas
from random_word import RandomWords

# Set working to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "manual_flagging"))

# Set time for run_id
chicago_tz = pytz.timezone("America/Chicago")

# Inputs yaml as inputs
with open(os.path.join("yaml", "inputs_update.yaml"), "r") as stream:
    inputs = yaml.safe_load(stream)

# Connect to athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
)

date_floor = flg.months_back(date_str=inputs["time_frame"]["start"], num_months=11)

# Parse yaml to get which sales to flag
if inputs["time_frame"]["end"] == None:
    sql_time_frame = f"sale.sale_date >= DATE '{date_floor}'"
else:
    sql_time_frame = f"""(sale.sale_date 
        BETWEEN DATE '{date_floor}'
        AND DATE '{inputs['time_frame']['end']}')"""

SQL_QUERY = f"""
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
    res.char_bldg_sf AS char_bldg_sf
FROM default.vw_card_res_char res
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = res.pin
    AND sale.year = res.year
WHERE {sql_time_frame}
AND NOT sale.is_multisale
AND NOT res.pin_is_multicard
AND res.class IN (
    '202', '203', '204', '205', '206', '207', '208', '209',
    '210', '211', '212', '218', '219', '234', '278', '295'
)
"""

SQL_QUERY_SALES_VAL = """
SELECT *
FROM sale.flag
"""

# Execute queries and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description
df_ingest = as_pandas(cursor)
df = df_ingest

cursor.execute(SQL_QUERY_SALES_VAL)
df_ingest_flag = as_pandas(cursor)
df_flag_table = df_ingest_flag

# Data cleaning
df = df.astype({col[0]: flg.sql_type_to_pd_type(col[1]) for col in metadata})

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
    df=df_flagged,
    start_date=inputs["time_frame"]["start"],
    manual_update=True,
)

# - - - - - -
# Update version of re-flagged entries
# - - - - - -

# Group the existing data by 'ID' and find the maximum 'version' for each sale
existing_max_version = (
    df_flag_table.groupby("meta_sale_document_num")["version"]
    .max()
    .reset_index()
    .rename(columns={"version": "existing_version"})
)

# Merge, compute new version, and drop unnecessary columns
df_to_write = (
    df_flagged_final.merge(
        existing_max_version, on="meta_sale_document_num", how="left"
    )
    .assign(
        version=lambda x: x["existing_version"]
        .apply(lambda y: y + 1 if pd.notnull(y) else 1)
        .astype(int)
    )
    .drop(columns=["existing_version"])
)

# Write to flag table
flg.write_to_table(
    df=df_to_write,
    table_name="flag",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to parameter table
df_parameters = flg.get_parameter_df(
    df_to_write=df_to_write,
    df_ingest=df_ingest,
    iso_forest_cols=inputs["iso_forest"],
    stat_groups=inputs["stat_groups"],
    dev_bounds=inputs["dev_bounds"],
    short_term_thresh=SHORT_TERM_OWNER_THRESHOLD,
    run_id=run_id,
)

flg.write_to_table(
    df=df_parameters,
    table_name="parameter",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to group_mean table
df_write_group_mean = flg.get_group_mean_df(
    df=df_flagged, stat_groups=inputs["stat_groups"], run_id=run_id
)

flg.write_to_table(
    df=df_write_group_mean,
    table_name="group_mean",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to metadata table
commit_sha = sp.getoutput("git rev-parse HEAD")
df_metadata = flg.get_metadata_df(
    run_id=run_id, timestamp=timestamp, run_type="manual_update", commit_sha=commit_sha
)

flg.write_to_table(
    df=df_metadata,
    table_name="metadata",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)
