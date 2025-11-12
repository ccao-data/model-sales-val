import os

import pandas as pd

import constants
import utils

output_dir = "output"

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
        output_environment=constants.INPUTS["output_environment"],
    )
    print(f"{table} table successfully written")
