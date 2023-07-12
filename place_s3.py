import pandas as pd
import numpy as np
import subprocess as sp
import os
import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas
import awswrangler as wr

# set working to root, to pull from src
root = sp.getoutput('git rev-parse --show-toplevel')
os.chdir(root)

# import flagging functions
from src import flagging as flg

# inputs yaml as inputs
with open("inputs.yaml", 'r') as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# connect to athena
conn = connect(
    s3_staging_dir=os.getenv('AWS_ATHENA_S3_STAGING_DIR'),
    region_name=os.getenv('AWS_REGION')
)

SQL_QUERY = """
SELECT
    sale.sale_price AS meta_sale_price,
    sale.sale_date AS meta_sale_date,
    sale.doc_no AS meta_sale_document_num,
    sale.seller_name AS meta_sale_seller_name,
    sale.buyer_name AS meta_sale_buyer_name,
    sale.sale_filter_is_outlier,
    res.class AS class,
    res.township_code AS township_code,
    res.year AS year,
    res.pin AS pin,
    res.char_bldg_sf AS char_bldg_sf
FROM default.vw_card_res_char res
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = res.pin
    AND sale.year = res.year
WHERE (res.year
    BETWEEN '2014'
    AND '2022')
AND NOT sale.is_multisale
AND Year(sale.sale_date) >= 2014
"""

# execute query and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description
df = as_pandas(cursor)


#
# Flagging -------
#

# Convert column types
def sql_type_to_pd_type(sql_type):
    """
    This function translates SQL data types to equivalent pandas dtypes.
    """
    if sql_type in ['decimal']:
        return 'float64'
    

df = df.astype({col[0]: sql_type_to_pd_type(col[1]) for col in metadata})
df = df[df['class'] != 'EX'] #incorporate into athena pull?

# run outlier flagging methodology 
df_flag = flg.go(df=df, 
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

# partially flag and subset
mask = df_final['year'] != '2014'
df_final.loc[mask, 'sv_is_outlier'] = np.nan
df_final.loc[mask, 'sv_special_flags'] = np.nan
df_final.loc[mask, 'sv_outlier_type'] = np.nan

df_final = df_final[['meta_sale_document_num', 'year', 'township_code', 'class',
'meta_sale_price', 'meta_sale_date', 'meta_sale_seller_name',
'meta_sale_buyer_name', 'sale_filter_is_outlier', 'pin', 'char_bldg_sf',
'sv_outlier_type', 'sv_special_flags', 'sv_is_outlier']]

#
# place in bucket for crawler to pick up
#

bucket = 's3://ccao-data-warehouse-us-east-1/sale/val_test/'
file_name = 'sales_val_test.parquet'
s3_file_path = bucket + file_name

wr.s3.to_parquet(
    df=df_final,
    path=s3_file_path
)






