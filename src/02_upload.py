import os
import subprocess as sp

import pandas as pd
import yaml

import utils

root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "src"))

# Use yaml as inputs
with open("inputs.yaml", "r") as stream:
    inputs = yaml.safe_load(stream)

output_dir = os.path.join(root, "output")

flag = pd.read_parquet(os.path.join(output_dir, "flag.parquet"))
parameter = pd.read_parquet(os.path.join(output_dir, "parameter.parquet"))
group_mean = pd.read_parquet(os.path.join(output_dir, "group_mean.parquet"))
metadata = pd.read_parquet(os.path.join(output_dir, "metadata.parquet"))

tables_to_write = {
    "flag": flag,
    "parameter": parameter,
    "group_mean": group_mean,
    "metadata": metadata,
}

for table, df in tables_to_write.items():
    utils.write_to_table(
        df=df,
        table_name=table,
        run_id=metadata.run_id[0],
        output_environment=inputs["output_environment"],
    )
    print(f"{table} table successfully written")
