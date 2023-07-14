import pandas as pd
import numpy as np
import subprocess as sp
import os
import yaml
import s3fs
from pyathena import connect
from pyathena.pandas.util import as_pandas
import awswrangler as wr
import boto3

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create clients
s3 = boto3.client('s3')
glue_client = boto3.client('glue', region_name='us-east-1')

# import flagging functions and yaml file from s3 
s3.download_file('ccao-glue-assets-us-east-1', 'scripts/flagging.py', '/tmp/flagging.py')
s3.download_file('ccao-glue-assets-us-east-1', 'scripts/inputs.yaml', '/tmp/inputs.yaml')

# Load the python script
exec(open("/tmp/flagging.py").read())

# load yaml
with open("/tmp/inputs.yaml", 'r') as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# connect to athena
conn = connect(
    s3_staging_dir='s3://ccao-athena-results-us-east-1',
    region_name='us-east-1'
)

SQL_QUERY = """
SELECT * 
FROM sale.val_test
"""

# execute query and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description
df = as_pandas(cursor)

# Convert column types
def sql_type_to_pd_type(sql_type):

    #This function translates SQL data types to equivalent pandas dtypes.

    if sql_type in ['decimal']:
        return 'float64'
    
df = df.astype({col[0]: sql_type_to_pd_type(col[1]) for col in metadata})
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

cols_to_write = ['meta_sale_document_num', 'year', 'township_code', 'class',
'meta_sale_price', 'meta_sale_date', 'meta_sale_seller_name',
'meta_sale_buyer_name', 'sale_filter_is_outlier', 'pin', 'char_bldg_sf',
'sv_outlier_type', 'sv_special_flags', 'sv_is_outlier']

wr.s3.to_parquet(
    df=df_final[cols_to_write],
    path=s3_file_path
)

# Trigger reporting glue crawler to update athena
glue_client.start_crawler(Name='ccao-data-warehouse-sale-crawler')