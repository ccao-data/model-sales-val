from glue.flagging_script_glue import flagging as flg_model
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

# Set working dir to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "manual_flagging"))

# Set time for run_id
chicago_tz = pytz.timezone("America/Chicago")

# Use yaml as inputs
with open(os.path.join("yaml", "inputs_update.yaml"), "r") as stream:
    inputs = yaml.safe_load(stream)

# Connect to Athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
)

date_floor = flg.months_back(
    date_str=inputs["time_frame"]["start"],
    num_months=inputs["rolling_window_months"] - 1,
)

# Parse yaml to get which sales to flag
if inputs["time_frame"]["end"] == None:
    sql_time_frame = f"sale.sale_date >= DATE '{date_floor}'"
else:
    sql_time_frame = f"""(sale.sale_date
        BETWEEN DATE '{date_floor}'
        AND DATE '{inputs['time_frame']['end']}')"""

# Fetch sales and characteristics from Athena
SQL_QUERY = f"""
WITH CombinedData AS (
    -- Select data from vw_card_res_char
    SELECT
        'res_char' AS source_table,
        'res' AS indicator,  -- Indicator column for 'res'
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

    -- Selecting data from vw_pin_condo_char
    SELECT
        'condo_char' AS source_table,
        'condo' AS indicator,  -- Indicator column for 'condo'
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
)

-- Now, join with sale table and filters
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
    data.indicator  -- Selecting the indicator column
FROM CombinedData data
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = data.pin
    AND sale.year = data.year
WHERE {sql_time_frame}
AND NOT sale.sale_filter_same_sale_within_365
AND NOT sale.sale_filter_less_than_10k
AND NOT sale.sale_filter_deed_type
AND NOT sale.is_multisale
AND (
    NOT data.pin_is_multicard
    OR data.source_table = 'condo_char'
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

conversion_dict = {
    col[0]: flg.sql_type_to_pd_type(col[1])
    for col in metadata
    if flg.sql_type_to_pd_type(col[1]) is not None
}
df = df.astype(conversion_dict)
df["ptax_flag_original"].fillna(False, inplace=True)

# Separate res and condo sales based on the indicator column
df_res = df[df["indicator"] == "res"].reset_index(drop=True)
df_condo = df[df["indicator"] == "condo"].reset_index(drop=True)

# Create rolling windows
df_res_to_flag = flg.add_rolling_window(
    df_res, num_months=inputs["rolling_window_months"]
)
df_condo_to_flag = flg.add_rolling_window(
    df_condo, num_months=inputs["rolling_window_months"]
)

# Create condo stat groups. Condos are all collapsed into a single class,
# since there are very few 297s or 399s
condo_stat_groups = inputs["stat_groups"].copy()
condo_stat_groups.remove("class")

# Flag outliers using the main flagging model
df_res_flagged = flg_model.go(
    df=df_res_to_flag,
    groups=tuple(inputs["stat_groups"]),
    iso_forest_cols=inputs["iso_forest"],
    dev_bounds=tuple(inputs["dev_bounds"]),
    condos=False,
)

# Discard any flags with a group size under the threshold
df_res_flagged_updated = flg.group_size_adjustment(
    df=df_res_flagged,
    stat_groups=inputs["stat_groups"],
    min_threshold=inputs["min_groups_threshold"],
    condos=False,
)

# Flag condo outliers, here we remove price per sqft as an input
# for the isolation forest model since condos don't have a unit sqft
condo_iso_forest = inputs["iso_forest"].copy()
condo_iso_forest.remove("sv_price_per_sqft")

df_condo_flagged = flg_model.go(
    df=df_condo_to_flag,
    groups=tuple(condo_stat_groups),
    iso_forest_cols=condo_iso_forest,
    dev_bounds=tuple(inputs["dev_bounds"]),
    condos=True,
)

df_condo_flagged_updated = flg.group_size_adjustment(
    df=df_condo_flagged,
    stat_groups=condo_stat_groups,
    min_threshold=inputs["min_groups_threshold"],
    condos=True,
)

# Update the PTAX flag column with an additional std dev conditional w/ res groups
df_res_flagged_updated_ptax = flg.ptax_adjustment(
    df=df_res_flagged_updated,
    groups=inputs["stat_groups"],
    ptax_sd=inputs["ptax_sd"],
    condos=False,
)

# Update the PTAX flag column with an additional std dev conditional w/ condo groups
df_condo_flagged_updated_ptax = flg.ptax_adjustment(
    df=df_condo_flagged_updated,
    groups=condo_stat_groups,
    ptax_sd=inputs["ptax_sd"],
    condos=True,
)

df_flagged_ptax_merged = pd.concat(
    [df_res_flagged_updated_ptax, df_condo_flagged_updated_ptax]
).reset_index(drop=True)

# Finish flagging and subset to write to flag table
df_flagged_final, run_id, timestamp = flg.finish_flags(
    df=df_flagged_ptax_merged,
    start_date=inputs["time_frame"]["start"],
    manual_update=False,
)

# -----------------------------------------------------------------------------
# Update version of re-flagged sales
# -----------------------------------------------------------------------------

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

# Write to sale.flag table
flg.write_to_table(
    df=df_to_write,
    table_name="flag",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to sale.parameter table
df_parameters = flg.get_parameter_df(
    df_to_write=df_to_write,
    df_ingest=df_ingest,
    iso_forest_cols=inputs["iso_forest"],
    res_stat_groups=inputs["stat_groups"],
    condo_stat_groups=condo_stat_groups,
    dev_bounds=inputs["dev_bounds"],
    ptax_sd=inputs["ptax_sd"],
    rolling_window=inputs["rolling_window_months"],
    date_floor=inputs["time_frame"]["start"],
    short_term_thresh=flg_model.SHORT_TERM_OWNER_THRESHOLD,
    min_group_thresh=inputs["min_groups_threshold"],
    run_id=run_id,
)

# Standardize dtypes to prevent Athena errors
df_parameters = flg.modify_dtypes(df_parameters)

flg.write_to_table(
    df=df_parameters,
    table_name="parameter",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to sale.group_mean table
df_res_group_mean = flg.get_group_mean_df(
    df=df_res_flagged, stat_groups=inputs["stat_groups"], run_id=run_id, condos=False
)

df_condo_group_mean = flg.get_group_mean_df(
    df=df_condo_flagged, stat_groups=condo_stat_groups, run_id=run_id, condos=True
)

df_group_mean_merged = pd.concat([df_res_group_mean, df_condo_group_mean]).reset_index(
    drop=True
)

flg.write_to_table(
    df=df_group_mean_merged,
    table_name="group_mean",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to sale.metadata table
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
