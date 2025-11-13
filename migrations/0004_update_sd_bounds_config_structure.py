# TODO: Explain the script
import json
import os

import awswrangler as wr

parquet_files_prod_prior = wr.s3.list_objects(
    os.path.join(
        "s3://ccao-data-backup-us-east-1/0004_update_sd_bounds_config_structure/flag_prior/parameter_prod_prior/"
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

# Edit existing
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

    # Create a new dataframe instead of modifying the original
    df_updated = df.copy(deep=True)
    df_updated["standard_deviation_bounds"] = [s] * len(df_updated)
    df_updated.drop(columns=["dev_bounds", "ptax_sd"], inplace=True)

    dfs_prod_updated[key] = df_updated

output_prefix = "s3://ccao-data-backup-us-east-1/0004_update_sd_bounds_config_structure/flag_updated/parameter_prod_updated/"

for name, df in dfs_prod_updated.items():
    output_path = os.path.join(output_prefix, f"{name}.parquet")
    wr.s3.to_parquet(
        df=df,
        path=output_path,
        index=False,
    )
