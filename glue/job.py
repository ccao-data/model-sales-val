import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# set custom logging on
logger = glueContext.get_logger()
# -------------------------

import pandas as pd
import numpy as np
import subprocess as sp
import os
import yaml
import s3fs
import boto3
import time

# Create clients
s3 = boto3.client('s3')
glue_client = boto3.client('glue', region_name='us-east-1')

# import flagging functions and yaml file from s3 
s3.download_file('ccao-glue-assets-us-east-1', 'scripts/flagging.py', '/tmp/flagging.py')
s3.download_file('ccao-glue-assets-us-east-1', 'scripts/inputs.yaml', '/tmp/inputs.yaml')

# Load the python script
exec(open("/tmp/flagging.py").read())

# load yaml
with open("inputs.yaml", 'r') as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


athena = boto3.client('athena', region_name='us-east-1')

query = """
SELECT * 
FROM sale.val_test
"""

# intermediate results
output = 's3://ccao-athena-results-us-east-1/'  

# Execute query
response = athena.start_query_execution(
    QueryString=query,
    QueryExecutionContext={
        'Database': 'sale' 
    },
    ResultConfiguration={
        'OutputLocation': output,
    }
)

query_execution_id = response['QueryExecutionId']

while True:
    response = athena.get_query_execution(QueryExecutionId=query_execution_id)
    if response['QueryExecution']['Status']['State'] in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
        break
    time.sleep(5)  # Wait for the query to complete

result_file = f'{output}{query_execution_id}.csv'
print(f'Results are in: {result_file}')

fs = s3fs.S3FileSystem()

# pandas can take a file-like object, so we provide a file opened for reading
with fs.open(f's3://ccao-athena-results-us-east-1/{query_execution_id}.csv', 'rb') as f:
    df = pd.read_csv(f)

cols_to_write = df.columns

# temp dtype solutions
df['meta_sale_date'] = pd.to_datetime(df['meta_sale_date'])
df = df[df['class'] != 'EX'] #incorporate into athena pull?

# run outlier flagging methodology (go comes from flagging.py)
df_flag = go(df=df, 
             groups=tuple(inputs['stat_groups']),
             iso_forest_cols=inputs['iso_forest'],
             dev_bounds=tuple(inputs['dev_bounds']))

# utilize ptaxsim to complete
df_final = (df_flag
      .rename(columns={'sv_is_outlier': 'sv_is_autoval_outlier'})
      .assign(sv_is_autoval_outlier = lambda df: df['sv_is_autoval_outlier'] == "Outlier")
      .assign(sv_is_outlier = lambda df:
               df['sv_is_autoval_outlier'] | df['sale_filter_is_outlier'])
      .assign(sv_outlier_type = lambda df: 
              np.where((df['sv_outlier_type'] == "Not outlier") & df['sale_filter_is_outlier'], 
                        "PTAX-203 flag", df['sv_outlier_type']))
      .assign(sv_is_outlier = lambda df: df['sv_outlier_type'] != "Not outlier"))

bucket = 's3://ccao-data-warehouse-us-east-1/sale/val_test/'
file_name = 'sales_val_test.parquet'
s3_file_path = bucket + file_name

# establish a connection to s3
fs = s3fs.S3FileSystem(anon=False)

# To use S3 Filesystem to write to S3:
with fs.open(s3_file_path, 'wb') as f:
    df_final[cols_to_write].to_parquet(f)

# Trigger reporting glue crawler to update athena
glue_client.start_crawler(Name='ccao-data-warehouse-sale-crawler')