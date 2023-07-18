import pandas as pd
import numpy as np
import subprocess as sp
import os
import datetime
import pytz
import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas
import awswrangler as wr

# set working to root, to pull from src
root = sp.getoutput('git rev-parse --show-toplevel')
os.chdir(root)

# set time for run_id
chicago_tz = pytz.timezone('America/Chicago')

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
AND NOT res.pin_is_multicard
AND Year(sale.sale_date) >= 2014
AND Year(sale.sale_date) <= 2022
"""

# execute query and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description
df_ingest = as_pandas(cursor)

# ---------------
# subset years for testing
# ---------------

mask = df_ingest['meta_sale_date'].dt.year < 2020
df = df_ingest.loc[mask]
df_masked = df

# ----
# intitial flagging
# ----

# Convert column types
def sql_type_to_pd_type(sql_type):
    """
    This function translates SQL data types to equivalent pandas dtypes.
    """
    if sql_type in ['decimal']:
        return 'float64'
    

df = df.astype({col[0]: sql_type_to_pd_type(col[1]) for col in metadata})
df = df[df['class'] != 'EX'] #incorporate into athena pull?

# run outlier heuristic flagging methodology 
df_flag = flg.go(df=df, 
                 groups=tuple(inputs['stat_groups']),
                 iso_forest_cols=inputs['iso_forest'],
                 dev_bounds=tuple(inputs['dev_bounds']))



# utilize ptaxsim, complete binary columns 
df_final = (df_flag
      .rename(columns={'sv_is_outlier': 'sv_is_autoval_outlier'})
      .assign(sv_is_autoval_outlier = lambda df: df['sv_is_autoval_outlier'] == "Outlier")
      .assign(sv_is_outlier = lambda df:
               df['sv_is_autoval_outlier'] | df['sale_filter_is_outlier'])
      # incorporate PTAX in sv_outlier_type
      .assign(sv_outlier_type = lambda df: 
              np.where((df['sv_outlier_type'] == "Not outlier") & df['sale_filter_is_outlier'], 
                        "PTAX-203 flag", df['sv_outlier_type']))
      # change sv_is_outlier to binary
      .assign(sv_is_outlier = lambda df: (df['sv_outlier_type'] != "Not outlier").astype(int))
      # ptax binary
      .assign(sv_is_ptax_outlier = lambda df: 
              np.where(df['sv_outlier_type'] == "PTAX-203 flag", 1, 0))
      # heuristics flagging binary column
      .assign(sv_is_outlier_heuristics = lambda df:
              np.where((df['sv_outlier_type'] != 'PTAX-203 flag') & (df['sv_is_outlier'] == 1), 1, 0))
              # current time
      .assign(run_id = datetime.datetime.now(chicago_tz).strftime('%Y-%m-%d_%H:%M'))
            )

cols_to_write = ['run_id', 'meta_sale_document_num', 'sv_is_outlier', 'sv_is_ptax_outlier',
       'sv_is_outlier_heuristics', 'sv_outlier_type']

bucket = 's3://ccao-data-warehouse-us-east-1/sale/val_test/'
file_name = 'sales_val_test.parquet'
s3_file_path = bucket + file_name

wr.s3.to_parquet(
    df=df_final[cols_to_write],
    path=s3_file_path
)

