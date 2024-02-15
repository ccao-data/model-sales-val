import os
import awswrangler as wr
import pandas as pd
import numpy as np
from glue import sales_val_flagging as flg
import subprocess as sp

# Set working dir to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root))


def write_dfs_to_s3(dfs, bucket, table):
    """
    Writes dicctionary of dfs to bucket
    """

    for df_name, df in dfs.items():
        file_path = f"{bucket}/sale/{table}/{df_name}.parquet"

        wr.s3.to_parquet(df=df, path=file_path, index=False)


def read_parquets_to_dfs(prefix, table):
    """
    Reads all parquet files from a specified S3 path into separate pandas DataFrames,
    names them with a given prefix, and returns them in a dictionary.

    Parameters:
    - prefix: The prefix to prepend to each DataFrame name.
    - s3_path: The S3 path where the parquet files are stored. Should be in the format "s3://bucket-name/path/to/files".

    Returns:
    A dictionary of DataFrames keyed by their names.
    """
    # List all parquet files in the specified S3 path
    parquet_files = wr.s3.list_objects(
        os.path.join(os.getenv("AWS_S3_WAREHOUSE_BUCKET"), "sale", table),
        suffix=".parquet",
    )

    # Initialize a dictionary to hold the dataframes
    dfs = {}

    # Loop through the parquet files, read each into a DataFrame, and store it in the dictionary
    for file in parquet_files:
        # Extract a meaningful name part from the file path for use in the DataFrame variable name
        name_part = file.split("/")[-1].split(".")[
            0
        ]  # Adjust this as necessary based on your file naming convention
        df_name = f"{prefix}{name_part}"

        # Read the parquet file into a DataFrame
        df = wr.s3.read_parquet(file)

        # Store the DataFrame in the dictionary with the constructed name
        dfs[df_name] = df

    return dfs


# - - - - - -
# Adjust flag tables
# - - - - - -

# Ingest sale.flag data
dfs_sale_flag = read_parquets_to_dfs("", "flag")

# OG initial run
# 'res_og_mansueto', 'condos_og_mansueto'
dfs_sale_flag["2024-01-19_18:46-clever-boni"]["group"] = dfs_sale_flag[
    "2024-01-19_18:46-clever-boni"
]["group"].apply(
    lambda value: value + "-res_og_mansueto"
    if value.count("_") == 2
    else (value + "-condos_og_mansueto" if value.count("_") == 1 else value)
)

# Update res run
dfs_sale_flag["2024-01-29_14:40-pensive-rina"]["group"] = dfs_sale_flag[
    "2024-01-29_14:40-pensive-rina"
]["group"].apply(
    lambda value: value + "-res_single_family"
    if "40_years" in value
    else (value + "-res_multi_family" if "20_years" in value else value)
)

# Update new condos run
dfs_sale_flag["2024-02-01_12:24-nifty-tayun"]["group"] = (
    dfs_sale_flag["2024-02-01_12:24-nifty-tayun"]["group"] + "-condos"
)

write_dfs_to_s3(dfs_sale_flag, os.getenv("AWS_TEST_ARCH_CHANGE_BUCKET"), "flag")

# - - - - - -
# Adjust parameter tables
# - - - - - -

# Change existing prod param table
dfs_sale_parameter = read_parquets_to_dfs("", "parameter")

dfs_sale_parameter["2024-01-19_18:46-clever-boni"].columns

dfs_sale_parameter["2024-01-19_18:46-clever-boni"]["stat_groups"] = str(
    {
        "og_mansueto": {
            "res": {"columns": ["rolling_window", "township_code", "class"]},
            "condos": {"columns": ["township_code", "rolling_window"]},
        }
    }
)

dfs_sale_parameter["2024-01-19_18:46-clever-boni"]["iso_forest_cols"] = str(
    {
        "res": {
            "columns": [
                "meta_sale_price",
                "sv_price_per_sqft",
                "sv_days_since_last_transaction",
                "sv_cgdr",
                "sv_sale_dup_counts",
            ]
        },
        "condos": {
            "columns": [
                "meta_sale_price",
                "sv_days_since_last_transaction",
                "sv_cgdr",
                "sv_sale_dup_counts",
            ]
        },
    }
)

dfs_sale_parameter["2024-01-19_18:46-clever-boni"]["tri_stat_groups"] = str(
    {1: "og_mansueto", 2: "og_mansueto", 3: "og_mansueto"}
)

dfs_sale_parameter["2024-01-19_18:46-clever-boni"]["run_filter"] = str(
    {
        "housing_market_type": ["res_single_family", "res_multi_family", "condos"],
        "run_tri": [1, 2, 3],
    }
)

dfs_sale_parameter["2024-01-19_18:46-clever-boni"] = dfs_sale_parameter[
    "2024-01-19_18:46-clever-boni"
][
    [
        "run_id",
        "sales_flagged",
        "earliest_data_ingest",
        "latest_data_ingest",
        "run_filter",
        "iso_forest_cols",
        "stat_groups",
        "tri_stat_groups",
        "dev_bounds",
        "ptax_sd",
        "rolling_window",
        "date_floor",
        "short_term_owner_threshold",
        "min_group_thresh",
    ]
]

# Add/Adjust condos city tri updates:
dfs_sale_parameter["2024-02-01_12:24-nifty-tayun"] = pd.read_parquet(
    os.path.join(root, "manual_flagging/new_condo_metadata/df_parameters.parquet")
)

dfs_sale_parameter["2024-02-01_12:24-nifty-tayun"]["stat_groups"] = str(
    {"condos": {"columns": ["rolling_window", "geography_split"]}}
)

dfs_sale_parameter["2024-02-01_12:24-nifty-tayun"]["iso_forest_cols"] = str(
    {
        "condos": {
            "columns": [
                "meta_sale_price",
                "sv_days_since_last_transaction",
                "sv_cgdr",
                "sv_sale_dup_counts",
            ]
        }
    }
)

dfs_sale_parameter["2024-02-01_12:24-nifty-tayun"]["tri_stat_groups"] = str(
    {1: "current", 2: "og_mansueto", 3: "og_mansueto"}
)

dfs_sale_parameter["2024-02-01_12:24-nifty-tayun"]["run_filter"] = str(
    {"housing_market_type": ["condos"], "run_tri": [1]}
)

dfs_sale_parameter["2024-02-01_12:24-nifty-tayun"] = dfs_sale_parameter[
    "2024-02-01_12:24-nifty-tayun"
][
    [
        "run_id",
        "sales_flagged",
        "earliest_data_ingest",
        "latest_data_ingest",
        "run_filter",
        "iso_forest_cols",
        "stat_groups",
        "tri_stat_groups",
        "dev_bounds",
        "ptax_sd",
        "rolling_window",
        "date_floor",
        "short_term_owner_threshold",
        "min_group_thresh",
    ]
]

# Add/Adjust res city tri updates:
dfs_sale_parameter["2024-01-29_14:40-pensive-rina"] = pd.read_parquet(
    os.path.join(root, "manual_flagging/new_res_metadata/df_parameters.parquet")
)

dfs_sale_parameter["2024-01-29_14:40-pensive-rina"]["stat_groups"] = str(
    {
        "res": {
            "single_family": {
                "columns": [
                    "rolling_window",
                    "geography_split",
                    "bldg_age_bin",
                    "char_bldg_sf_bin",
                ]
            },
            "multi_family": {
                "columns": ["rolling_window", "geography_split", "bldg_age_bin"]
            },
        }
    }
)

dfs_sale_parameter["2024-01-29_14:40-pensive-rina"]["iso_forest_cols"] = str(
    {
        "res": {
            "columns": [
                "meta_sale_price",
                "sv_price_per_sqft",
                "sv_days_since_last_transaction",
                "sv_cgdr",
                "sv_sale_dup_counts",
            ]
        }
    }
)

dfs_sale_parameter["2024-01-29_14:40-pensive-rina"]["tri_stat_groups"] = str(
    {1: "current", 2: "og_mansueto", 3: "og_mansueto"}
)

dfs_sale_parameter["2024-01-29_14:40-pensive-rina"]["run_filter"] = str(
    {"housing_market_type": ["res"], "run_tri": [1]}
)

dfs_sale_parameter["2024-01-29_14:40-pensive-rina"] = dfs_sale_parameter[
    "2024-01-29_14:40-pensive-rina"
][
    [
        "run_id",
        "sales_flagged",
        "earliest_data_ingest",
        "latest_data_ingest",
        "run_filter",
        "iso_forest_cols",
        "stat_groups",
        "tri_stat_groups",
        "dev_bounds",
        "ptax_sd",
        "rolling_window",
        "date_floor",
        "short_term_owner_threshold",
        "min_group_thresh",
    ]
]

for key, value in dfs_sale_parameter.items():
    dfs_sale_parameter[key] = flg.modify_dtypes(dfs_sale_parameter[key])

write_dfs_to_s3(
    dfs_sale_parameter, os.getenv("AWS_TEST_ARCH_CHANGE_BUCKET"), "parameter"
)

# - - -
# Update sale.group_mean tables
# - - -

# Ingest sale.flag data
dfs_sale_group_mean = read_parquets_to_dfs("", "group_mean")

# Read in other data here

# Add/Adjust condos city tri updates:
dfs_sale_group_mean["2024-02-01_12:24-nifty-tayun"] = pd.read_parquet(
    os.path.join(root, "manual_flagging/new_condo_metadata/df_condo_group_mean.parquet")
)
# Add/Adjust res city tri updates:
dfs_sale_group_mean["2024-01-29_14:40-pensive-rina"] = pd.read_parquet(
    os.path.join(root, "manual_flagging/new_res_metadata/df_group_mean.parquet")
)

# OG initial run
# 'res_og_mansueto', 'condos_og_mansueto'
dfs_sale_group_mean["2024-01-19_18:46-clever-boni"]["group"] = dfs_sale_group_mean[
    "2024-01-19_18:46-clever-boni"
]["group"].apply(
    lambda value: value + "-res_og_mansueto"
    if value.count("_") == 2
    else (value + "-condos_og_mansueto" if value.count("_") == 1 else value)
)

# Update res run
dfs_sale_group_mean["2024-01-29_14:40-pensive-rina"]["group"] = dfs_sale_group_mean[
    "2024-01-29_14:40-pensive-rina"
]["group"].apply(
    lambda value: value + "-res_single_family"
    if "40_years" in value
    else (value + "-res_multi_family" if "20_years" in value else value)
)

# Update new condos run
dfs_sale_group_mean["2024-02-01_12:24-nifty-tayun"]["group"] = (
    dfs_sale_group_mean["2024-02-01_12:24-nifty-tayun"]["group"] + "-condos"
)

write_dfs_to_s3(
    dfs_sale_group_mean, os.getenv("AWS_TEST_ARCH_CHANGE_BUCKET"), "group_mean"
)

# - - -
# Update sale.metadata tables
# - - -

# get from prod bucket
dfs_sale_metadata = read_parquets_to_dfs("", "metadata")

# Add/Adjust condos city tri updates:
dfs_sale_metadata["2024-02-01_12:24-nifty-tayun"] = pd.read_parquet(
    os.path.join(root, "manual_flagging/new_condo_metadata/df_metadata.parquet")
)
# Add/Adjust res city tri updates:
dfs_sale_metadata["2024-01-29_14:40-pensive-rina"] = pd.read_parquet(
    os.path.join(root, "manual_flagging/new_res_metadata/df_metadata.parquet")
)

write_dfs_to_s3(dfs_sale_metadata, os.getenv("AWS_TEST_ARCH_CHANGE_BUCKET"), "metadata")
