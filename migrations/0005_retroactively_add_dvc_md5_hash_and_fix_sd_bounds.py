"""
This PR makes two small migrations.

sale.parameter migration is to correct an erroneous inclusion of the top level
keystandard_deviation_bounds from src/inputs.yaml

sale.metadata migration is to include an md5 hash for input data for a sales val
run after we implemented dvc into the pipeline
"""

import json
import os

import awswrangler as wr

# sale.parameter migration
# important: we are not touching the 2025-11-17_16:15-blissful-billy.parquet file in sale.parameter
parquet_files_prod_prior = wr.s3.list_objects(
    os.path.join(
        "s3://ccao-data-backup-us-east-1/0005_retroactively_add_dvc_md5_hash_and_fix_sd_bounds/parameter_prior/"
    ),
    suffix=".parquet",
)

dfs_parameter_prod_prior = {}
dfs_parameter_prod_updated = {}

# Read existing prod data (copied into backup bucket from prod)
for file in parquet_files_prod_prior:
    name_part = file.split("/")[-1].split(".")[0]
    df_name = f"{name_part}"

    df = wr.s3.read_parquet(file)
    dfs_parameter_prod_prior[df_name] = df

# Iterate through existing data to remove top level
for name, df in dfs_parameter_prod_prior.items():
    updated_df_param = df.copy()

    def strip_top_level(row):
        data = json.loads(row)
        return json.dumps(data["standard_deviation_bounds"])

    updated_df_param["standard_deviation_bounds"] = updated_df_param[
        "standard_deviation_bounds"
    ].apply(strip_top_level)

    dfs_parameter_prod_updated[name] = updated_df_param

output_prefix = "s3://ccao-data-backup-us-east-1/0005_retroactively_add_dvc_md5_hash_and_fix_sd_bounds/parameter_updated/"

for name, df in dfs_parameter_prod_updated.items():
    output_path = os.path.join(output_prefix, f"{name}.parquet")
    wr.s3.to_parquet(
        df=df,
        path=output_path,
        index=False,
    )


# sale.metadata migration
# important: during the migration we will only delete and add the 2025-11-17_16:15-blissful-billy.parquet
# file since this is the only run we have done with dvc implementation

# Ingest single parquet we want to change
df_metadata_prod_prior = wr.s3.read_parquet(
    "s3://ccao-data-backup-us-east-1/0005_retroactively_add_dvc_md5_hash_and_fix_sd_bounds/metadata_prior/2025-11-17_16:15-blissful-billy.parquet"
)
df_metadata_prod_updated = df_metadata_prod_prior.copy()
df_metadata_prod_updated["dvc_md5_sales_ingest"] = (
    "75cceeac0b21f484f184668e06935c8b"
)

# Iterate through existing data to grab and transform sd bounds
df_metadata_updated_output_path = "s3://ccao-data-backup-us-east-1/0005_retroactively_add_dvc_md5_hash_and_fix_sd_bounds/metadata_updated/2025-11-17_16:15-blissful-billy.parquet"

wr.s3.to_parquet(
    df=df_metadata_prod_updated,
    path=df_metadata_updated_output_path,
    index=False,
)
