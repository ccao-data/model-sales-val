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

# Use yaml as inputs
with open(os.path.join("yaml", "inputs_initial.yaml"), "r") as stream:
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
        res.char_yrblt as yrblt,
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
        NULL AS yrblt,
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
    sale.nbhd as nbhd,
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
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = data.pin
    AND sale.year = data.year
INNER JOIN "default"."vw_pin_universe" universe 
    ON universe.pin = data.pin
    AND universe.year = data.year
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
df = df_ingest

conversion_dict = {
    col[0]: flg.sql_type_to_pd_type(col[1])
    for col in metadata
    if flg.sql_type_to_pd_type(col[1]) is not None
}
df = df.astype(conversion_dict)
df["ptax_flag_original"].fillna(False, inplace=True)


# - - - - - - - - -
# Development
# - - - - - - - - -

# Handle current methodology data manipulation if needed
if "current" in tri_stat_groups.values():
    # Calculate the building's age
    current_year = datetime.datetime.now().year
    df["bldg_age"] = current_year - df["yrblt"]

    # Ingest new geographic groups
    df_new_groups = pd.read_excel(
        os.path.join(root, "QC_salesval_nbhds_round2.xlsx"),
        usecols=["Town Nbhd", "Town Grp 1"],
    ).rename(columns={"Town Nbhd": "nbhd", "Town Grp 1": "geography_split"})
    df["nbhd"] = df["nbhd"].astype(int)
    df = pd.merge(df, df_new_groups, on="nbhd", how="left")

#
# Make correct filters and set up dictionary structure
#

# Filter to correct tris
tri_stat_groups = {
    tri: method
    for tri, method in inputs["tri_stat_groups"].items()
    if tri in inputs["run_tri"]
}


dfs_to_feature_creation = {}  # Dictionary to store DataFrames

for tri, method in tri_stat_groups.items():
    print(tri, method)

    # Iterate over markets
    for market in inputs["housing_run_type"]:
        if method == "current":
            key = f"df_tri{tri}_{market}_current"
            # Filter by triad code and market type
            triad_code_filter = df["triad_code"] == str(tri)
            market_filter = df["class"].isin(inputs["housing_run_type_filters"][market])

            dfs_to_feature_creation[key] = {
                "df": df[triad_code_filter & market_filter],
                "columns": inputs["stat_groups"]["current"][market]["columns"],
                "iso_forest_cols": inputs["iso_forest"][
                    "res" if "res" in market else "condos"
                ],
                "condos_boolean": market == "condos",
            }

        elif method == "og_mansueto":
            # Collect all mansueto tris
            mansueto_tris_to_flag = [
                str(key)
                for key, value in tri_stat_groups.items()
                if value == "og_mansueto"
            ]
            # Filter by triad code and market type
            triad_code_filter = ~df["triad_code"].isin(mansueto_tris_to_flag)

            df_res_og_mansueto = df[(df["indicator"] == "res") & triad_code_filter]
            df_condo_og_mansueto = df[(df["indicator"] == "condo") & triad_code_filter]

            # Append these DataFrames to the dictionary
            key_res = f"df_res_og_mansueto"
            key_condo = f"df_condos_og_mansueto"

            dfs_to_feature_creation[key_res] = {
                "df": df_res_og_mansueto,
                "columns": inputs["stat_groups"]["og_mansueto"]["res_single_family"][
                    "columns"
                ],
                "iso_forest_cols": inputs["iso_forest"]["res"],
                "condos_boolean": False,
            }
            dfs_to_feature_creation[key_condo] = {
                "df": df_condo_og_mansueto,
                "columns": inputs["stat_groups"]["og_mansueto"]["condos"]["columns"],
                "iso_forest_cols": inputs["iso_forest"]["condos"],
                "condos_boolean": True,
            }

# check names
for key in dfs_to_feature_creation:
    print(key)

# - - - - - - -
# Make features
# - - - - - - -


def create_bins_and_labels(input_list):
    # Initialize bins with 0 and float("inf")
    bins = [0] + input_list + [float("inf")]

    # Generate labels based on bins
    labels = []
    for i in range(len(bins) - 1):
        if i == 0:
            labels.append(f"below_{bins[i+1]}")
        elif i == len(bins) - 2:
            labels.append(f"above_{bins[i]}")
        else:
            labels.append(f"{bins[i]}_to_{bins[i+1]}")

    return bins, labels


# Iterate through loop and correctly create columns that are used for statistical grouping
dfs_to_rolling_window = dfs_to_feature_creation.copy()

for df_name, df in dfs_to_feature_creation.items():
    # Make a copy of the data frame to edit
    df_copy = df["df"].copy()

    # print(f"{df_name}")
    if "current" in df_name:
        print("current")

        if "res_single_family" in df_name:
            # Bin sf
            print("Start making bins")
            char_bldg_sf_bins, char_bldg_sf_labels = create_bins_and_labels(
                inputs["stat_groups"]["current"]["res_single_family"][
                    "sf_bin_specification"
                ]
            )

            df_copy["char_bldg_sf_bin"] = pd.cut(
                df_copy["char_bldg_sf"],
                bins=char_bldg_sf_bins,
                labels=char_bldg_sf_labels,
            )
            # Define bins for building age
            bldg_age_bins, bldg_age_labels = create_bins_and_labels(
                inputs["stat_groups"]["current"]["res_single_family"][
                    "age_bin_specification"
                ]
            )
            df_copy["bldg_age_bin"] = pd.cut(
                df_copy["bldg_age"],
                bins=char_bldg_sf_bins,
                labels=char_bldg_sf_labels,
            )
            print("Finish bins")
        if "res_multi_family" in df_name:
            # Define bins for building age
            bldg_age_bins, bldg_age_labels = create_bins_and_labels(
                inputs["stat_groups"]["current"]["res_multi_family"][
                    "age_bin_specification"
                ]
            )
            df_copy["bldg_age_bin"] = pd.cut(
                df_copy["bldg_age"],
                bins=char_bldg_sf_bins,
                labels=char_bldg_sf_labels,
            )
            pass

        if "condos" in df_name:
            """
            Currently no feature engineering needed in
            the current condos method
            """
            pass

    elif "og_mansueto" in df_name:
        print("og_mansueto")
        if "res" in df_name:
            """
            Currently no feature engineering needed in
            the og_mansueto method
            """
            pass

        if "condos" in df_name:
            """
            Currently no feature engineering needed in
            the current og_mansueto method
            """
            pass

        # Add the edited or unedited dataframe to the new dictionary
    dfs_to_rolling_window[df_name]["df"] = df_copy


# check names and columns
for key, value in dfs_to_rolling_window.items():
    print(key)
    print(value["df"].columns)

# - - - - - -
# Make rolling window
# - - - - - -

dfs_to_flag = dfs_to_rolling_window.copy()

for df_name, df in dfs_to_rolling_window.items():
    print(f"Assigning rolling window for {df_name}")
    df_copy = df["df"].copy()

    df_copy = flg.add_rolling_window(
        df_copy, num_months=inputs["rolling_window_months"]
    )
    dfs_to_flag[df_name]["df"] = df_copy


# - - - - -
# Flag Sales
# - - - - -

dfs_flagged = dfs_to_flag.copy()

for df_name, df in dfs_to_flag.items():
    # Make a copy of the data frame to edit
    print(f"\nFlagging sales for {df_name}")
    df_copy = df["df"].copy()
    df_copy = flg_model.go(
        df=df_copy,
        groups=tuple(df["columns"]),
        iso_forest_cols=df["iso_forest_cols"],
        dev_bounds=tuple(inputs["dev_bounds"]),
        condos=df["condos_boolean"],
    )

    # Add the edited or unedited dataframe to the new dictionary
    dfs_flagged[df_name]["df"] = df_copy

# check names
for key in dfs_flagged:
    print(key)

#
# Adjust outliers based on group sizes and incorporate ptax information
#


# NEW
dfs_to_finalize = dfs_flagged.copy()

for df_name, df in dfs_flagged.items():
    # Make a copy of the data frame to edit
    print(f"\n Enacting group threshold and creating ptax data for {df_name}")
    df_copy = df["df"].copy()
    df_copy = flg.group_size_adjustment(
        df=df_copy,
        stat_groups=df["columns"],
        min_threshold=inputs["min_groups_threshold"],
        condos=df["condos_boolean"],
    )
    df_copy = flg.ptax_adjustment(
        df=df_copy,
        groups=df["columns"],
        ptax_sd=inputs["ptax_sd"],
        condos=df["condos_boolean"],
    )

    # Add the edited or unedited dataframe to the new dictionary
    dfs_to_finalize[df_name]["df"] = df_copy


df_flagged_ptax_merged = pd.concat(
    [df_res_flagged_updated_ptax, df_condo_flagged_updated_ptax]
).reset_index(drop=True)

# Finish flagging and subset to write to flag table
df_to_write, run_id, timestamp = flg.finish_flags(
    df=df_flagged_ptax_merged,
    start_date=inputs["time_frame"]["start"],
    manual_update=False,
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
    run_id=run_id,
    timestamp=timestamp,
    run_type="initial_flagging",
    commit_sha=commit_sha,
)

flg.write_to_table(
    df=df_metadata,
    table_name="metadata",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)
