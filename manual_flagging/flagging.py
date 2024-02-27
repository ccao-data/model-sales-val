from glue.flagging_script_glue import flagging as flg_model
from glue import sales_val_flagging as flg
import awswrangler as wr
import copy
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
with open(os.path.join("yaml", "inputs.yaml"), "r") as stream:
    inputs = yaml.safe_load(stream)

# Validate the input specification
# Check housing_market_type
assert "housing_market_type" in inputs, "Missing key: 'housing_market_type'"
assert set(inputs["housing_market_type"]).issubset(
    {"res_single_family", "res_multi_family", "condos"}
), "housing_market_type can only contain 'res_single_family', 'res_multi_family', 'condos'"
assert len(inputs["housing_market_type"]) == len(
    set(inputs["housing_market_type"])
), "Duplicate values in 'housing_market_type'"

# Check run_tri
assert "run_tri" in inputs, "Missing key: 'run_tri'"
assert set(inputs["run_tri"]).issubset({1, 2, 3}), "run_tri can only contain 1, 2, 3"
assert len(inputs["run_tri"]) == len(
    set(inputs["run_tri"])
), "Duplicate values in 'run_tri'"

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
INNER JOIN default.vw_pin_universe universe 
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

if inputs["manual_update"] == True:
    # TODO: sub this out for prod table before merging
    SQL_QUERY_SALES_VAL = """
    SELECT *
    FROM ci_model_sales_val_89_architecture_change_for_variable_methodology_sale.flag
    """
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

# Filter to correct tris
tri_stat_groups = {
    tri: method
    for tri, method in inputs["tri_stat_groups"].items()
    if tri in inputs["run_tri"]
}

# Calculate the building's age for feature creation
current_year = datetime.datetime.now().year
df["bldg_age"] = current_year - df["yrblt"]

"""
Ingest and join new geographic groups for current methodology.

To update our methodology with new geographic classifications, we currently
utilize the 'geography_split' column, which is effective for uniform groupings
across all market types, as observed in the city tri(1). For
subsequent tris, if new classifications are consistent across markets,
they can be appended to the 'geo_geography_split' column. However, for
market-specific variations (e.g., condos vs. single-family homes),
we should introduce an additional column or use a conditional join to
ensure accurate integration of these diverse groupings.
"""

df_new_groups_tri1 = pd.read_excel(
    os.path.join(root, "data", "res_condos_nbhd_groups_2024.xlsx"),
    usecols=["Town Nbhd", "Town Grp 1"],
).rename(columns={"Town Nbhd": "nbhd", "Town Grp 1": "geography_split"})

df["nbhd"] = df["nbhd"].astype(int)
df = pd.merge(df, df_new_groups_tri1, on="nbhd", how="left")


def create_bins_and_labels(input_list):
    """
    Some of the groups used for flagging are partitions of
    building size or age, this helper function returns the
    bins and labels for the column creation based on input data
    from our config file.
    """

    # Initialize bins with 0 and float("inf")
    bins = [0] + input_list + [float("inf")]

    # Generate labels based on bins
    labels = []
    for i in range(len(bins) - 1):
        if i == 0:
            labels.append(f"below-{bins[i+1]}")
        elif i == len(bins) - 2:
            labels.append(f"above-{bins[i]}")
        else:
            labels.append(f"{bins[i]}-to-{bins[i+1]}")

    return bins, labels


# - - - - - - -
# Make correct filters and set up dictionary structure
# Split tris into groups according to their flagging methods
# - - - - - - -

dfs_to_rolling_window = {}  # Dictionary to store DataFrames

# Collect tris where our contemporary flagging methods are used
current_tris = [tri for tri, method in tri_stat_groups.items() if method == "current"]

for tri in current_tris:
    # Iterate over housing types defined in yaml
    for housing_type in inputs["housing_market_type"]:
        # Assign the df a name
        key = f"df_tri{tri}_{housing_type}_current"

        # Perform filtering based on tri and housing market class codes
        triad_code_filter = df["triad_code"] == str(tri)
        market_filter = df["class"].isin(
            inputs["housing_market_class_codes"][housing_type]
        )

        # Initialize the DataFrame for the current key
        df_filtered = df[triad_code_filter & market_filter].copy()
        dfs_to_rolling_window[key] = {"df": df_filtered}  # Store the filtered DataFrame

        # Extract the specific housing type configuration
        housing_type_config = inputs["stat_groups"]["current"][f"tri{tri}"][
            housing_type
        ]

        # Update market/tri configurations
        dfs_to_rolling_window[key]["columns"] = housing_type_config["columns"]
        dfs_to_rolling_window[key]["iso_forest_cols"] = inputs["iso_forest"][
            "res" if "res" in housing_type else "condos"
        ]
        dfs_to_rolling_window[key]["condos_boolean"] = housing_type == "condos"
        dfs_to_rolling_window[key]["market"] = housing_type

        # Apply binning configurations if they exist
        if "sf_bin_specification" in housing_type_config:
            # Create and apply bins for square footage
            sf_bins, sf_labels = create_bins_and_labels(
                housing_type_config["sf_bin_specification"]
            )
            dfs_to_rolling_window[key]["df"]["char_bldg_sf_bin"] = pd.cut(
                dfs_to_rolling_window[key]["df"]["char_bldg_sf"],
                bins=sf_bins,
                labels=sf_labels,
            )

        if "age_bin_specification" in housing_type_config:
            # Create and apply bins for age
            age_bins, age_labels = create_bins_and_labels(
                housing_type_config["age_bin_specification"]
            )
            dfs_to_rolling_window[key]["df"]["bldg_age_bin"] = pd.cut(
                dfs_to_rolling_window[key]["df"]["bldg_age"],
                bins=age_bins,
                labels=age_labels,
            )

# Collect tris that will be flagged for OG method
og_mansueto_tris = [
    tri for tri, method in tri_stat_groups.items() if method == "og_mansueto"
]

if og_mansueto_tris:
    # Filter by triad code and market type
    og_mansueto_filter = ~df["triad_code"].isin(og_mansueto_tris)

    df_res_og_mansueto = df[(df["indicator"] == "res") & triad_code_filter]
    df_condo_og_mansueto = df[(df["indicator"] == "condo") & triad_code_filter]

    # Append these DataFrames to the dictionary
    key_res = (
        f"df_tri{'&'.join(str(number) for number in og_mansueto_tris)}_res_og_mansueto"
    )
    key_condo = f"df_tri{'&'.join(str(number) for number in og_mansueto_tris)}_condos_og_mansueto"

    dfs_to_rolling_window[key_res] = {
        "df": df_res_og_mansueto,
        "columns": inputs["stat_groups"]["og_mansueto"]["res_single_family"]["columns"],
        "iso_forest_cols": inputs["iso_forest"]["res"],
        "condos_boolean": False,
        "market": "res_og_mansueto",
    }
    dfs_to_rolling_window[key_condo] = {
        "df": df_condo_og_mansueto,
        "columns": inputs["stat_groups"]["og_mansueto"]["condos"]["columns"],
        "iso_forest_cols": inputs["iso_forest"]["condos"],
        "condos_boolean": True,
        "market": "condos_og_mansueto",
    }

# - - - - - -
# Make rolling window
# - - - - - -

dfs_to_flag = copy.deepcopy(dfs_to_rolling_window)

for df_name, df_info in dfs_to_rolling_window.items():
    print(f"Assigning rolling window for {df_name}")
    df_copy = df_info["df"].copy()

    df_copy = flg.add_rolling_window(
        df_copy, num_months=inputs["rolling_window_months"]
    )
    dfs_to_flag[df_name]["df"] = df_copy

# - - - - -
# Flag Sales
# - - - - -

dfs_flagged = copy.deepcopy(dfs_to_flag)

for df_name, df_info in dfs_to_flag.items():
    print(f"\nFlagging sales for {df_name}")
    df_copy = df_info["df"].copy()
    df_copy = flg_model.go(
        df=df_copy,
        groups=tuple(df_info["columns"]),
        iso_forest_cols=df_info["iso_forest_cols"],
        dev_bounds=tuple(inputs["dev_bounds"]),
        condos=df_info["condos_boolean"],
    )

    # Add the edited or unedited dataframe to the new dictionary
    dfs_flagged[df_name]["df"] = df_copy

# - - - - - - - - - - -
# Adjust outliers based on group sizes and incorporate ptax information
# - - - - - - - - - - -

dfs_to_finalize = copy.deepcopy(dfs_flagged)

for df_name, df_info in dfs_flagged.items():
    # Make a copy of the data frame to edit
    print(f"\n Enacting group threshold and creating ptax data for {df_name}")
    df_copy = df_info["df"].copy()
    df_copy = flg.group_size_adjustment(
        df=df_copy,
        stat_groups=df_info["columns"],
        min_threshold=inputs["min_groups_threshold"],
        condos=df_info["condos_boolean"],
    )
    df_copy = flg.ptax_adjustment(
        df=df_copy,
        groups=df_info["columns"],
        ptax_sd=inputs["ptax_sd"],
        condos=df_info["condos_boolean"],
    )
    """
    Modify the 'group' column by appending '-market_value', this is done
    to make sure that a two different groups with the same run_id won't
    be returned with the same value. For example, if res and condos have the 
    same column groupings, joining the group column from sale.flag to sale.group_mean
    by 'group' and 'run_id' could potentially return two groups. This market type
    append fixes that. This is also added in the group_mean data.
    """
    market_value = df_info["market"]
    df_copy["group"] = df_copy["group"].astype(str) + "-" + market_value

    # Add the edited or unedited dataframe to the new dictionary
    dfs_to_finalize[df_name]["df"] = df_copy

# - - - - - - -
# Finalize data to write and create data for all metadata tables
# - - - - - - - -

if inputs["manual_update"] == True:
    # Group the existing data by 'ID' and find the maximum 'version' for each sale
    existing_max_version = (
        df_flag_table.groupby("meta_sale_document_num")["version"]
        .max()
        .reset_index()
        .rename(columns={"version": "existing_version"})
    )


dfs_to_finalize_list = [details["df"] for details in dfs_to_finalize.values()]
df_to_finalize = pd.concat(dfs_to_finalize_list, axis=0)

df_to_write, run_id, timestamp = flg.finish_flags(
    df=df_to_finalize, start_date=inputs["time_frame"]["start"]
)

if inputs["manual_update"] == True:
    # Merge, compute new version, and drop unnecessary columns
    df_to_write = (
        df_to_write.merge(existing_max_version, on="meta_sale_document_num", how="left")
        .assign(
            version=lambda x: x["existing_version"]
            .apply(lambda y: y + 1 if pd.notnull(y) else 1)
            .astype(int)
        )
        .drop(columns=["existing_version"])
    )

run_filter = str(
    {"housing_market_type": inputs["housing_market_type"], "run_tri": inputs["run_tri"]}
)

# Get sale.parameter data
df_parameter = flg.get_parameter_df(
    df_to_write=df_to_write,
    df_ingest=df_ingest,
    run_filter=run_filter,
    iso_forest_cols=inputs["iso_forest"],
    stat_groups=inputs["stat_groups"],
    tri_stat_groups=inputs["tri_stat_groups"],
    housing_market_class_codes=inputs["housing_market_class_codes"],
    dev_bounds=inputs["dev_bounds"],
    ptax_sd=inputs["ptax_sd"],
    rolling_window=inputs["rolling_window_months"],
    date_floor=inputs["time_frame"]["start"],
    short_term_thresh=flg_model.SHORT_TERM_OWNER_THRESHOLD,
    min_group_thresh=inputs["min_groups_threshold"],
    run_id=run_id,
)

# Standardize dtypes to prevent Athena errors
df_parameter = flg.modify_dtypes(df_parameter)

# Get sale.group_mean data
df_group_means = []  # List to store the transformed DataFrames

for df_name, df_info in dfs_to_finalize.items():
    df_group_mean = flg.get_group_mean_df(
        df=dfs_flagged[df_name]["df"],
        stat_groups=df_info["columns"],
        run_id=run_id,
        condos=df_info["condos_boolean"],
    )
    market_value = df_info["market"]
    df_group_mean["group"] = df_group_mean["group"].astype(str) + "-" + market_value
    df_group_means.append(df_group_mean)

df_group_mean_to_write = pd.concat(df_group_means, ignore_index=True)

# Get sale.metadata table
commit_sha = sp.getoutput("git rev-parse HEAD")

# Write to sale.group_mean table
df_metadata = flg.get_metadata_df(
    run_id=run_id,
    timestamp=timestamp,
    run_type="initial_flagging"
    if inputs["manual_update"] == False
    else "manual_update",
    commit_sha=commit_sha,
)

# - - - -
# Write tables
# - - - -

tables_to_write = {
    "flag": df_to_write,
    "parameter": df_parameter,
    "group_mean": df_group_mean_to_write,
    "metadata": df_metadata,
}

for table, df in tables_to_write.items():
    flg.write_to_table(
        df=df,
        table_name=table,
        s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        run_id=run_id,
    )
    print(f"{table} table successfully written")
