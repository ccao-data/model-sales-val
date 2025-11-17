"""
This migration updates sale.parameter to account for our standard deviation
bounds configuration change. With the change we can assign differing bounds
for condos and res in a single run.

The required change to `sale.parameter` is to remove `dev_bounds` and `ptax_sd`
and replace them with `standard_deviation_bounds`, which is a json/dict structure
stored as a string type.
"""

import json
import os

import awswrangler as wr

# Note: The directories in this script have a flag_* subdirectory even though
# the purpose is to migrate parameter data (not flag data). This is just a minor
# inconsistency in the script, the `sale.flag` table was not involved
parquet_files_prod_prior = wr.s3.list_objects(
    os.path.join(
        "s3://ccao-data-backup-us-east-1/0004_update_sd_bounds_config_structure/flag_prior/"
    ),
    suffix=".parquet",
)

dfs_prod_prior = {}
dfs_prod_updated = {}

# Read existing prod data (copied into backup bucket from prod)
for file in parquet_files_prod_prior:
    name_part = file.split("/")[-1].split(".")[0]
    df_name = f"{name_part}"

    df = wr.s3.read_parquet(file)
    dfs_prod_prior[df_name] = df

# Iterate through existing data to grab and transform sd bounds
for key, df in dfs_prod_prior.items():
    dev = list(df["dev_bounds"].iloc[0])
    ptax = list(df["ptax_sd"].iloc[0])

    json_data = json.dumps(
        {
            "standard_deviation_bounds": {
                "standard_bounds": {"res": dev, "condos": dev},
                "ptax_bounds": {"res": ptax, "condos": ptax},
            }
        }
    )

    # Output to new dict
    df_updated = df.copy(deep=True)
    df_updated["standard_deviation_bounds"] = [json_data] * len(df_updated)
    df_updated.drop(columns=["dev_bounds", "ptax_sd"], inplace=True)

    dfs_prod_updated[key] = df_updated

output_prefix = "s3://ccao-data-backup-us-east-1/0004_update_sd_bounds_config_structure/flag_updated/"

for name, df in dfs_prod_updated.items():
    output_path = os.path.join(output_prefix, f"{name}.parquet")
    wr.s3.to_parquet(
        df=df,
        path=output_path,
        index=False,
    )
