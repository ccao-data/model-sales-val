import awswrangler as wr
import boto3
import os
import subprocess as sp
import numpy as np
from pyathena import connect
from pyathena.pandas.util import as_pandas

# Set working dir to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root))


def read_parquet_files_from_s3(input_path):
    """
    Reads all Parquet files from a specified S3 path into a dictionary of DataFrames.

    Parameters:
        input_path (str): The S3 bucket path where Parquet files are stored, e.g., 's3://my-bucket/my-folder/'

    Returns:
        dict: A dictionary where each key is the filename and the value is the corresponding DataFrame.
    """
    # Initialize the S3 session
    session = boto3.Session()

    # List all objects in the given S3 path that are Parquet files
    s3_objects = wr.s3.list_objects(path=input_path, boto3_session=session)

    # Filter objects to get only parquet files
    parquet_files = [obj for obj in s3_objects if obj.endswith(".parquet")]

    # Dictionary to store DataFrames
    dataframes = {}

    # Read each Parquet file into a DataFrame and store it in the dictionary
    for file_path in parquet_files:
        # Read the Parquet file into a DataFrame
        df = wr.s3.read_parquet(path=file_path, boto3_session=session)

        # Extract the filename without the path for use as the dictionary key
        filename = file_path.split("/")[-1].replace(".parquet", "")

        # Add the DataFrame to the dictionary
        dataframes[filename] = df

    return dataframes


def process_dataframe(df, recode_dict):
    """
    Transforms old structure with sv_outlier_type
    to new structure with 3 separate outlier reason columns
    """
    # Insert new columns filled with NaN
    pos = df.columns.get_loc("sv_outlier_type") + 1
    for i in range(1, 4):
        df.insert(pos, f"sv_outlier_reason{i}", np.nan)
        pos += 1

    # Use the dictionary to populate the new columns
    for key, value in recode_dict.items():
        mask = df["sv_outlier_type"] == key
        for col, val in value.items():
            df.loc[mask, col] = val

    df = df.drop(columns=["sv_outlier_type"])

    return df


def write_dfs_to_s3(dfs, bucket, table):
    """
    Writes dictionary of dfs to bucket
    """

    for df_name, df in dfs.items():
        file_path = f"{bucket}/0002_update_outlier_column_structure_w_iasworld_2024_update/new_prod_data/{table}/{df_name}.parquet"
        wr.s3.to_parquet(df=df, path=file_path, index=False)


dfs_flag = read_parquet_files_from_s3(
    os.path.join(
        os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        "sale",
        "flag",
    )
)
"""
for i in dfs_flag:
    print(i)

dfs_flag["2024-01-19_18:46-clever-boni"].sv_outlier_type.value_counts()
"""

recode_dict = {
    "PTAX-203 flag (Low)": {
        "sv_outlier_reason1": "PTAX-203 Exclusion",
        "sv_outlier_reason2": np.nan,
    },
    "PTAX-203 flag (High)": {
        "sv_outlier_reason1": "PTAX-203 Exclusion",
        "sv_outlier_reason2": np.nan,
    },
    "Non-person sale (low)": {
        "sv_outlier_reason1": "Low price",
        "sv_outlier_reason2": "Non-person sale",
    },
    "Non-person sale (high)": {
        "sv_outlier_reason1": "High price",
        "sv_outlier_reason2": "Non-person sale",
    },
    "High price (raw)": {
        "sv_outlier_reason1": "High price",
        "sv_outlier_reason2": np.nan,
    },
    "Low price (raw)": {
        "sv_outlier_reason1": "Low price",
        "sv_outlier_reason2": np.nan,
    },
    "Anomaly (high)": {
        "sv_outlier_reason1": "High price",
        "sv_outlier_reason2": "Statistical Anomaly",
    },
    "Anomaly (low)": {
        "sv_outlier_reason1": "Low price",
        "sv_outlier_reason2": "Statistical Anomaly",
    },
    "Low Price (raw & sqft)": {
        "sv_outlier_reason1": "Low price",
        "sv_outlier_reason2": "Low price per square foot",
    },
    "High price (raw and sqft)": {
        "sv_outlier_reason1": "High price",
        "sv_outlier_reason2": "High price per square foot",
    },
    "High price (sqft)": {
        "sv_outlier_reason1": "High price per square foot",
        "sv_outlier_reason2": np.nan,
    },
    "Low price (sqft)": {
        "sv_outlier_reason1": "Low price per square foot",
        "sv_outlier_reason2": np.nan,
    },
    "Home flip sale (high)": {
        "sv_outlier_reason1": "High price",
        "sv_outlier_reason2": "Price swing / Home flip",
    },
    "Home flipe sale (low)": {
        "sv_outlier_reason1": "Low price",
        "sv_outlier_reason2": "Price swing / Home flip",
    },
    "Family sale (high)": {
        "sv_outlier_reason1": "High price",
        "sv_outlier_reason2": "Family sale",
    },
    "Family sale (low)": {
        "sv_outlier_reason1": "Low price",
        "sv_outlier_reason2": "Family sale",
    },
    "High price swing": {
        "sv_outlier_reason1": "High price",
        "sv_outlier_reason2": "Price swing / Home flip",
    },
    "Low price swing": {
        "sv_outlier_reason1": "Low price",
        "sv_outlier_reason2": "Price swing / Home flip",
    },
}

for key in dfs_flag:
    dfs_flag[key] = process_dataframe(dfs_flag[key], recode_dict)


write_dfs_to_s3(dfs_flag, os.getenv("AWS_BUCKET_SV"), "flag")
