import datetime
import os
import subprocess as sp
import time

import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas

import constants
import utils

root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "src"))

# Use yaml as inputs
with open("inputs.yaml", "r") as stream:
    inputs = yaml.safe_load(stream)


# Connect to Athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
)

date_floor = utils.months_back(
    date_str=inputs["time_frame"]["start"],
    num_months=inputs["rolling_window_months"] - 1,
)

# Parse yaml to get which sales to flag
if inputs["time_frame"]["end"] is None:
    sql_time_frame = f"sale.sale_date >= DATE '{date_floor}'"
else:
    sql_time_frame = f"""(sale.sale_date
        BETWEEN DATE '{date_floor}'
        AND DATE '{inputs["time_frame"]["end"]}')"""

start_time = time.time()
print("Starting ingest query...")
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
        res.char_yrblt as yrblt,
        res.pin AS pin,
        res.char_bldg_sf AS char_bldg_sf,
        res.pin_is_multicard
    FROM {constants.DEFAULT_CARD_RES_CHAR_TABLE} res
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
        NULL AS yrblt,
        condo.pin AS pin,
        NULL AS char_bldg_sf,
        FALSE AS pin_is_multicard
    FROM {constants.DEFAULT_VW_PIN_CONDO_CHAR_TABLE} condo
    WHERE condo.class IN ('297', '299', '399')
    AND NOT condo.is_parking_space
    AND NOT condo.is_common_area
),

-- Select neighborhood groups and filter for most recent versions
neighborhood_group AS (
    SELECT nbhd_group.nbhd, nbhd_group.group_name
    FROM {constants.LOCATION_NEIGHBORHOOD_GROUP_TABLE} AS nbhd_group
    INNER JOIN (
        SELECT nbhd, MAX(version) AS version
        FROM {constants.LOCATION_NEIGHBORHOOD_GROUP_TABLE}
        GROUP BY nbhd
    ) AS latest_group_version
        ON nbhd_group.nbhd = latest_group_version.nbhd
        AND nbhd_group.version = latest_group_version.version
)

-- Now, join with sale table and filters
SELECT
    sale.sale_price AS meta_sale_price,
    sale.sale_date AS meta_sale_date,
    sale.doc_no AS meta_sale_document_num,
    sale.seller_name AS meta_sale_seller_name,
    sale.buyer_name AS meta_sale_buyer_name,
    sale.nbhd as nbhd,
    nbhd_group.group_name as geography_split,
    sale.sale_filter_ptax_flag AS ptax_flag_original,
    data.class,
    data.township_code,
    data.yrblt,
    data.year,
    data.pin,
    data.char_bldg_sf,
    data.indicator,
    universe.triad_code 
FROM CombinedData data
INNER JOIN {constants.DEFAULT_VW_PIN_SALE_TABLE} sale
    ON sale.pin = data.pin
    AND sale.year = data.year
INNER JOIN {constants.DEFAULT_VW_PIN_UNIVERSE_TABLE} universe 
    ON universe.pin = data.pin
    AND universe.year = data.year
LEFT JOIN neighborhood_group nbhd_group
    ON sale.nbhd = nbhd_group.nbhd
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

# Execute query and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description

df_ingest = as_pandas(cursor)
end_time = time.time()
elapsed = end_time - start_time
print(f"Ingest query completed in {elapsed:.2f} seconds")
df = df_ingest

conversion_dict = {
    col[0]: utils.sql_type_to_pd_type(col[1])
    for col in metadata
    if utils.sql_type_to_pd_type(col[1]) is not None
}
df = df.astype(conversion_dict)
df["ptax_flag_original"].fillna(False, inplace=True)

# Calculate the building's age for feature creation
current_year = datetime.datetime.now().year
df["char_bldg_age"] = current_year - df["yrblt"]

df.to_parquet(os.path.join(root, "input", "sales_ingest.parquet"), index=False)
