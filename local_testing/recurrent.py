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

SQL_QUERY_VAL_TEST = """
SELECT *
FROM sale.val_test
"""

# ----
# execute queries and return as pandas df
# ----

cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description
df_ingest_full = as_pandas(cursor)
df = df_ingest_full

def sql_type_to_pd_type(sql_type):
    """
    This function translates SQL data types to equivalent pandas dtypes.
    """
    if sql_type in ['decimal']:
        return 'float64'
    

df = df.astype({col[0]: sql_type_to_pd_type(col[1]) for col in metadata})
df = df[df['class'] != 'EX'] #incorporate into athena pull?

cursor.execute(SQL_QUERY_VAL_TEST)
df_ingest_val_test = as_pandas(cursor)
df_val_test = df_ingest_val_test

merged_df = pd.merge(df_ingest_full, df_val_test, on='meta_sale_document_num', how='left', indicator=True)
"""
# check for NAs ----- weird NAs coming up here, small number in completed years
pd.set_option('display.max_columns', 20)
merged_df.groupby(merged_df['meta_sale_date'].dt.year)['sv_is_outlier'].apply(lambda x: x.isna().sum())
df_ingest_val_test.groupby(df_ingest_val_test['meta_sale_date'].dt.year)['sv_is_outlier'].apply(lambda x: x.isna().sum())
merged_df[merged_df['meta_sale_date'].dt.year.isin(merged_df[merged_df['sv_is_outlier'].isna()]['meta_sale_date'].dt.year.unique())]
"""



# ----
# re-flagging
# ----


# gather years needed for re-flagging
na_rows = merged_df[merged_df['sv_is_outlier'].isna()]
years_to_tag = na_rows.year.unique().tolist()
df_to_flag = df[df.year.isin(years_to_tag)]

# run outlier heuristic flagging methodology 
df_flag = flg.go(df=df_to_flag, 
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

# append unseen rows to sales_val table
rows_to_append = df_final[~df_final['meta_sale_document_num'].isin(df_val_test['meta_sale_document_num'])]
sales_val_updated = pd.concat([df_val_test, rows_to_append[cols_to_write]], ignore_index=True)