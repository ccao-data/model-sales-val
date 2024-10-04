# Migration script to recompute the `sv_is_ptax_outlier` column for all
# existing sales val data. We needed to do this because an earlier version
# of the sales val pipeline accidentally used the wrong reason codes to
# compute this column, resulting in every row getting set to False
import os
import awswrangler as wr
import pandas as pd
import numpy as np
from glue import sales_val_flagging as flg
import subprocess as sp


def read_parquets_to_dfs(table):
    """
    Reads all parquet files from a specified S3 path into separate pandas
    DataFrames, and returns them in a dictionary.

    Parameters:
    - table: The table for which we want to read from (flag, parameter, group_mean, or meteadata)

    Returns:
    A dictionary of DataFrames keyed by their names.
    """
    # List all parquet files in the specified S3 path
    parquet_files = wr.s3.list_objects(
        os.path.join(os.getenv("AWS_BUCKET_SV_BACKUP"),
                     "0003_fix_is_ptax_outlier_column",
                     "old_prod_data",
                     table
                     ),
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
        df_name = f"{name_part}"

        # Read the parquet file into a DataFrame
        df = wr.s3.read_parquet(file)

        # Store the DataFrame in the dictionary with the constructed name
        dfs[df_name] = df

    return dfs


def update_sv_is_ptax_outlier(dfs):
    """
    Returns a new dictionary of DataFrames with updated 'sv_is_ptax_outlier' column.

    Parameters:
    - dfs: A dictionary of DataFrames.

    Returns:
    - new_dfs: A new dictionary with the same structure as 'dfs',
      where 'sv_is_ptax_outlier' is set to True where 'sv_outlier_reason2' is 'PTAX-203 Exclusion'.
    """
    new_dfs = {}
    for name, df in dfs.items():
        df_new = df.copy()

        # The PTAX value will always be in sv_outlier_reason2
        # due to the way we constructed the priority order of the columns
        mask = df_new['sv_outlier_reason2'] == 'PTAX-203 Exclusion'
        df_new.loc[mask, 'sv_is_ptax_outlier'] = True
        # Add the modified DataFrame to the new dictionary
        new_dfs[name] = df_new
    return new_dfs


def write_dfs_to_s3(dfs, bucket, table):
    """
    Writes dicctionary of dfs to bucket
    """

    for df_name, df in dfs.items():
        file_path = os.path.join(
            bucket,
            "0003_fix_is_ptax_outlier_column",
            "new_prod_data",
            table,
            df_name + ".parquet"
        )

        wr.s3.to_parquet(df=df, path=file_path, index=False)

dfs_sale_flag = read_parquets_to_dfs("flag")

dfs_ptax_edited = update_sv_is_ptax_outlier(dfs_sale_flag)


# Check for changes and numbers
for i in dfs_sale_flag:
    print(f"Checking for {i}")
    print(dfs_sale_flag[i].sv_is_ptax_outlier.value_counts())
    print(dfs_ptax_edited[i].sv_is_ptax_outlier.value_counts())
    print("\n")

# Write data to backup bucket 
write_dfs_to_s3(dfs_ptax_edited, os.getenv("AWS_BUCKET_SV_BACKUP"), "flag")
