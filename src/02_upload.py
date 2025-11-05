import os
import subprocess as sp

import pandas as pd
import yaml

import utils

# Use yaml as inputs
with open("inputs.yaml", "r") as stream:
    inputs = yaml.safe_load(stream)

# Set working dir to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "src"))

output_dir = os.path.join(root, "output")

df_to_write = pd.read_parquet(os.path.join(output_dir, "df_to_write.parquet"))
df_parameter = pd.read_parquet(
    os.path.join(output_dir, "df_parameter.parquet")
)
df_group_mean_to_write = pd.read_parquet(
    os.path.join(output_dir, "df_group_mean_to_write.parquet")
)
df_metadata = pd.read_parquet(os.path.join(output_dir, "df_metadata.parquet"))

tables_to_write = {
    "flag": df_to_write,
    "parameter": df_parameter,
    "group_mean": df_group_mean_to_write,
    "metadata": df_metadata,
}

for table, df in tables_to_write.items():
    utils.write_to_table(
        df=df,
        table_name=table,
        run_id=df_metadata.run_id[0],
        output_environment=inputs["output_environment"],
    )
    print(f"{table} table successfully written")
