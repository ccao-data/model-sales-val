# TODO: Explain the script
import json
import os

import awswrangler as wr

parquet_files_prod_prior = wr.s3.list_objects(
    os.path.join(
        "s3://ccao-data-backup-us-east-1/0004_update_sd_bounds_config_structure/paramter_prod_prior"
    ),
    suffix=".parquet",
)

dfs_prod_prior = {}

# Loop through the parquet files, read each into a DataFrame, and store it in the dictionary
for file in parquet_files_prod_prior:
    name_part = file.split("/")[-1].split(".")[0]
    df_name = f"{name_part}"

    df = wr.s3.read_parquet(file)

    dfs_prod_prior[df_name] = df

for key, df in dfs_prod_prior.items():
    dev = list(df["dev_bounds"].iloc[0])
    ptax = list(df["ptax_sd"].iloc[0])
    s = json.dumps(
        {
            "standard_deviation_bounds": {
                "standard_bounds": {"res": dev, "condos": dev},
                "ptax_bounds": {"res": ptax, "condos": ptax},
            }
        }
    )
    df["standard_deviation_bounds"] = [s] * len(df)
    df.drop(columns=["dev_bounds", "ptax_sd"], inplace=True)
    dfs_prod_prior[key] = df
