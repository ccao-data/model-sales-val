import awswrangler as wr
import boto3
import os
import subprocess as sp

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


dfs_flag = read_parquet_files_from_s3(
    os.path.join(
        os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        "sale",
        "flag",
    )
)

for i in dfs_flag:
    print(i)


dfs_flag["2024-01-19_18:46-clever-boni"].sv_outlier_type.value_counts()

# "PTAX-203 Exclusion (High)", "PTAX-203 Exclusion (Low)"
"""
Characteristic reasons
Short-term owner
PTAX-203 Exclusion
Family sale
Non-person sale
Statistical Anomaly
Price swing / Home Flip
"""
