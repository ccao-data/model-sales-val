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

# Separate res and condo sales based on the indicator column
df_res = df[df["indicator"] == "res"].reset_index(drop=True)
df_condo = df[df["indicator"] == "condo"].reset_index(drop=True)

# Create condo stat groups. Condos are all collapsed into a single class,
# since there are very few 297s or 399s
condo_stat_groups = inputs["stat_groups"].copy()
condo_stat_groups.remove("class")

# Create rolling windows
df_res_to_flag = flg.add_rolling_window(
    df_res, num_months=inputs["rolling_window_months"]
)
df_condo_to_flag = flg.add_rolling_window(
    df_condo, num_months=inputs["rolling_window_months"]
)

# Flag outliers using the main flagging model
df_res_flagged = flg_model.go(
    df=df_res_to_flag,
    groups=tuple(inputs["stat_groups"]),
    iso_forest_cols=inputs["iso_forest"],
    dev_bounds=tuple(inputs["dev_bounds"]),
    condos=False,
)

# - - - - - - -
## test sd columns
# - - - - - - -

df_res_flagged["original_meta_sale_price"] = np.power(
    10, df_res_flagged["meta_sale_price"]
)

df_res_flagged["group"] = (
    df_res_flagged["rolling_window"].astype(str)
    + "_"
    + df_res_flagged["township_code"].astype(str)
    + "_"
    + df_res_flagged["class"].astype(str)
)

df_res_flagged["group_mean"] = df_res_flagged.groupby("group")[
    "meta_sale_price"
].transform("mean")

df_res_flagged_original = df_res_flagged[df_res_flagged["original_observation"] == True]


df_res_flagged_original["std"] = (
    df_res_flagged_original["meta_sale_price"] - df_res_flagged_original["group_mean"]
) / df_res_flagged_original["sv_price_deviation_rolling_window_township_code_class"]


# - - - - - - -
# Refactor test
# - - - - - - -


def get_group_mean_df(df, stat_groups, run_id, condos):
    """
    Creates a DataFrame for group mean data. This function is primarily used
    for tracing the flagging decisions in a property sales model. It calculates the standard deviation and
    mean for specified groups within the data, and merges these calculations with group size information.

    The output of this function 'std' and 'std_sqft' are subject to whatever transformations
    we make on the data. As of 12/1/2023, they are log10 transformed. In order to back out
    how many standard deviations away from the mean a sale was, we just need to log transform
    the original sale_price.

    The function performs the following operations:
    1. Calculates the mean and standard deviation for each group defined by the 'stat_groups' parameters.
    2. Conditions the calculation of square footage data based on the 'condos' flag.
    3. Calculates the size of each group.
    4. Merges the calculated mean, standard deviation, and group size into a single DataFrame.

    Inputs:
        df (pd.DataFrame): The input DataFrame containing property sales data.
        stat_groups (list of str): A list of column names used for statistical grouping in the flagging model.
                                Typically includes 'rolling_window', 'township_code', and 'class'.
        run_id (str): A unique identifier for the script's run. Used for tracking and logging purposes.
        condos (bool): A boolean flag that determines whether square footage data (sqft) should be excluded
                    from the standard deviation calculation. If True, sqft data is excluded.

    Output:
        pd.DataFrame: A DataFrame ready to be written to Athena, containing the group mean, standard deviation,
                    and size data, along with a unique run identifier.
    """
    # Process standard deviation
    df_std = (
        df.assign(
            rolling_window=lambda x: pd.to_datetime(
                x["rolling_window"], format="%Y%m"
            ).dt.date
        )
        .assign(
            group=lambda x: x["rolling_window"].astype(str)
            + "_"
            + x["township_code"].astype(str)
            + "_"
            + x["class"].astype(str)
        )
        .assign(
            group_mean=lambda x: x.groupby("group")["meta_sale_price"].transform("mean")
        )
        .assign(
            std=lambda x: (x["meta_sale_price"] - x["group_mean"])
            / x["sv_price_deviation_rolling_window_township_code_class"]
        )
    )

    if not condos:
        df_std = df_std.assign(
            group_mean_sqft=lambda x: x.groupby("group")["sv_price_per_sqft"].transform(
                "mean"
            ),
            std_sqft=lambda x: (x["sv_price_per_sqft"] - x["group_mean_sqft"])
            / x["sv_price_per_sqft_deviation_rolling_window_township_code_class"],
        )

    df_std = df_std[["group", "std", "std_sqft"]].drop_duplicates(
        subset="group", keep="first"
    )

    # Process group sizes and means
    group_sizes = df.groupby(stat_groups).size().reset_index(name="group_size")
    df = df.merge(group_sizes, on=stat_groups, how="left")

    unique_groups = (
        df.drop_duplicates(subset=stat_groups, keep="first")
        .reset_index(drop=True)
        .assign(
            rolling_window=lambda x: pd.to_datetime(
                x["rolling_window"], format="%Y%m"
            ).dt.date
        )
    )

    groups_string_col = "_".join(map(str, stat_groups))
    suffixes = ["mean_price"] if condos else ["mean_price", "mean_price_per_sqft"]
    cols_to_write_means = (
        stat_groups
        + ["group_size"]
        + [f"sv_{suffix}_{groups_string_col}" for suffix in suffixes]
    )

    rename_dict = {
        f"sv_{suffix}_{groups_string_col}": f"{suffix}" for suffix in suffixes
    }
    df_means = (
        unique_groups[cols_to_write_means]
        .rename(columns=rename_dict)
        .assign(
            run_id=run_id,
            group=lambda x: x[stat_groups].astype(str).apply("_".join, axis=1),
        )
        .drop(columns=stat_groups)
    )

    # Merging the dataframes
    merged_df = df_means.merge(
        df_std[["group", "std", "std_sqft"]], on="group", how="left"
    )

    return merged_df


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

df_flagged_merged = pd.concat(
    [df_res_flagged_updated, df_condo_flagged_updated]
).reset_index(drop=True)

# Update the PTAX flag column with an additional std dev conditional
df_flagged_ptax = flg.ptax_adjustment(
    df=df_flagged_merged, groups=inputs["stat_groups"], ptax_sd=inputs["ptax_sd"]
)

# Finish flagging and subset to write to flag table
df_to_write, run_id, timestamp = flg.finish_flags(
    df=df_flagged_ptax,
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
