import awswrangler as wr
import boto3
import datetime
import numpy as np
import pandas as pd
import pytz
import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas

# Create S3 client
s3 = boto3.client('s3')

# Set timezone for run_id
chicago_tz = pytz.timezone('America/Chicago')

# Import flagging functions and yaml file from s3
s3.download_file('ccao-glue-assets-us-east-1', 'scripts/sales-val/flagging.py', '/tmp/flagging.py')
s3.download_file('ccao-glue-assets-us-east-1', 'scripts/sales-val/inputs.yaml', '/tmp/inputs.yaml')

# Load the python script and yaml
exec(open("/tmp/flagging.py").read())
with open("/tmp/inputs.yaml", 'r') as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Connect to athena
conn = connect(
    s3_staging_dir='s3://ccao-athena-results-us-east-1',
    region_name='us-east-1'
)

"""
This query grabs all data needed to flag unflagged values.
It takes 11 months of data prior to the earliest unflagged sale up
to the monthly data of the latest unflagged sale
"""
SQL_QUERY = """
WITH NA_Dates AS (
    SELECT
        MIN(DATE_TRUNC('MONTH', sale.sale_date)) - INTERVAL '11' MONTH AS StartDate,
        MAX(DATE_TRUNC('MONTH', sale.sale_date)) AS EndDate
    FROM default.vw_card_res_char res
    INNER JOIN default.vw_pin_sale sale
        ON sale.pin = res.pin
        AND sale.year = res.year
    LEFT JOIN sale.flag flag
        ON flag.meta_sale_document_num = sale.doc_no
    WHERE flag.sv_is_outlier IS NULL
        AND sale.sale_date >= DATE '2013-01-01'
        AND NOT sale.is_multisale
        AND NOT res.pin_is_multicard
)
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
    res.char_bldg_sf AS char_bldg_sf,
    flag.run_id,
    flag.sv_is_outlier,
    flag.sv_is_ptax_outlier,
    flag.sv_is_heuristic_outlier,
    flag.sv_outlier_type
FROM default.vw_card_res_char res
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = res.pin
    AND sale.year = res.year
LEFT JOIN sale.flag flag
    ON flag.meta_sale_document_num = sale.doc_no
INNER JOIN NA_Dates
    ON sale.sale_date BETWEEN NA_Dates.StartDate AND NA_Dates.EndDate
WHERE NOT sale.is_multisale
AND NOT res.pin_is_multicard
"""

SQL_QUERY_SALES_VAL = """
SELECT *
FROM sale.flag
"""

# ----
# Execute queries and return as pandas df
# ----

# Instantiate cursor
cursor = conn.cursor()

# Grab existing sales val table for later join
cursor.execute(SQL_QUERY_SALES_VAL)
df_ingest_sales_val = as_pandas(cursor)
df_sales_val = df_ingest_sales_val

# Get data needed to flag non-flagged data
cursor.execute(SQL_QUERY)
metadata = cursor.description
df_ingest_full = as_pandas(cursor)
df = df_ingest_full

def sql_type_to_pd_type(sql_type):
    """
    This function translates SQL data types to equivalent 
    pandas dtypes, using athena parquet metadata
    """

    # this is used to fix dtype so there is not error thrown in 
    # deviation_dollars() in flagging script on line 375
    if sql_type in ['decimal']:
        return 'float64'
    

df = df.astype({col[0]: sql_type_to_pd_type(col[1]) for col in metadata})

# Exempt sale handling
exempt_data = df[df['class'] == 'EX']
df = df[df['class'] != 'EX'] 

# - - - - - - - - 
# Create rolling window
# - - - - - - - -
max_date = df['meta_sale_date'].max()

df_to_flag = (
    # Creates dt column with 12 month dates
    df.assign(rolling_window=df['meta_sale_date']
              .apply(lambda x: pd.date_range(start=x, 
                                             periods=12, 
                                             freq='M')))
    # Expand rolling_windows dates to individual rows
    .explode('rolling_window')
    # Tag original observations 
    .assign(original_observation = lambda df: df['meta_sale_date'].dt.month == df['rolling_window'].dt.month)
    # Simplify to month level
    .assign(rolling_window=lambda df: df['rolling_window'].dt.to_period('M'))
    # Filter such that rolling_window isn't extrapolated into future, we are concerned with historic and present-month data
    .loc[lambda df: df['rolling_window'] <= max_date.to_period('M')]
    # Back to float for flagging script 
    .assign(rolling_window=lambda df: df['rolling_window']
            .apply(lambda x: x.strftime('%Y%m')).astype(float))
)

# ----
# Re-flagging
# ----

# Run outlier heuristic flagging methodology 
df_flag = go(df=df_to_flag, 
                 groups=tuple(inputs['stat_groups']),
                 iso_forest_cols=inputs['iso_forest'],
                 dev_bounds=tuple(inputs['dev_bounds']))

# Remove duplicate rows
df_flag = df_flag[df_flag['original_observation']]

# Discard pre-2014 data
df_flag = df_flag[df_flag['meta_sale_date'] >= '2014-01-01']

# Utilize PTAX-203, complete binary columns 
df_final = (df_flag
      .rename(columns={'sv_is_outlier': 'sv_is_autoval_outlier'})
      .assign(sv_is_autoval_outlier = lambda df: df['sv_is_autoval_outlier'] == "Outlier")
      .assign(sv_is_outlier = lambda df:
               df['sv_is_autoval_outlier'] | df['sale_filter_is_outlier'])
      # Incorporate PTAX in sv_outlier_type
      .assign(sv_outlier_type = lambda df: 
              np.where((df['sv_outlier_type'] == "Not outlier") & df['sale_filter_is_outlier'], 
                        "PTAX-203 flag", df['sv_outlier_type']))
      # Change sv_is_outlier to binary
      .assign(sv_is_outlier = lambda df: (df['sv_outlier_type'] != "Not outlier").astype(int))
      # PTAX-203 binary
      .assign(sv_is_ptax_outlier = lambda df: 
              np.where(df['sv_outlier_type'] == "PTAX-203 flag", 1, 0))
      # Heuristics flagging binary column
      .assign(sv_is_outlier_heuristics = lambda df:
              np.where((df['sv_outlier_type'] != 'PTAX-203 flag') & (df['sv_is_outlier'] == 1), 1, 0))
            )

# Manually impute ex values as non-outliers
exempt_to_append = exempt_data.meta_sale_document_num.reset_index().drop(columns='index')
exempt_to_append['sv_is_outlier'] = 0
exempt_to_append['sv_is_ptax_outlier'] = 0
exempt_to_append['sv_is_heuristic_outlier'] = 0
exempt_to_append['sv_outlier_type'] = 'Not Outlier'

cols_to_write = ['meta_sale_document_num', 'sv_is_outlier', 
                 'sv_is_ptax_outlier', 'sv_is_heuristic_outlier', 'sv_outlier_type']

# Merge exempt values and assign run_id
run_id = datetime.datetime.now(chicago_tz).strftime('%Y-%m-%d_%H:%M')
df_to_write = pd.concat([df_final[cols_to_write], exempt_to_append])
df_to_write['run_id'] = run_id

# Filter to keep only flags not already present in the flag table
rows_to_append = (df_to_write[~df_to_write['meta_sale_document_num']
                              .isin(df_sales_val['meta_sale_document_num'])])

# - - - - 
# Write parquet to bucket with newly flagged values
# - - - -

bucket = 's3://ccao-data-warehouse-us-east-1/sale/flag/'
file_name = run_id + '.parquet'
s3_file_path = bucket + file_name

wr.s3.to_parquet(
    df=rows_to_append,
    path=s3_file_path
)