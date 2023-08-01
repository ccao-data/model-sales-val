from src import flagging_rolling as flg
from src.flagging_rolling import SHORT_TERM_OWNER_THRESHOLD
import awswrangler as wr
import os
import datetime
import numpy as np
import pandas as pd
import pytz
import subprocess as sp
import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas
from random_word import RandomWords

# Set working to root, to pull from src
root = sp.getoutput('git rev-parse --show-toplevel')
os.chdir(root)

# Set time for run_id
chicago_tz = pytz.timezone('America/Chicago')

# Inputs yaml as inputs
with open("inputs.yaml", 'r') as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Connect to athena
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
WHERE (sale.sale_date
    BETWEEN DATE '2019-02-01'
    AND DATE '2021-12-31')
AND NOT sale.is_multisale
AND NOT res.pin_is_multicard
"""

"""
WHERE (sale.sale_date
    BETWEEN DATE '2018-05-01'
    AND DATE '2020-12-31')
"""

# Execute query and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description
df_ingest = as_pandas(cursor)
df = df_ingest

# -----
# Data cleaning
# -----


def sql_type_to_pd_type(sql_type):
    """
    This function translates SQL data types to equivalent
    pandas dtypes, using athena parquet metadata
    """

    # This is used to fix dtype so there is not error thrown in
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

df = (
    # Creates dt column with 12 month dates
    df.assign(rolling_window=df['meta_sale_date']
              .apply(lambda x: pd.date_range(start=x,
                                             periods=12,
                                             freq='M')))
    # Expand rolling_windows dates to individual rows
    .explode('rolling_window')
    # Tag original observations
    .assign(original_observation=lambda df: df['meta_sale_date'].dt.month == df['rolling_window'].dt.month)
    # Simplify to month level
    .assign(rolling_window=lambda df: df['rolling_window'].dt.to_period('M'))
    # Filter such that rolling_window isn't extrapolated into future, we are
    # concerned with historic and present-month data
    .loc[lambda df: df['rolling_window'] <= max_date.to_period('M')]
    # Back to float for flagging script
    .assign(rolling_window=lambda df: df['rolling_window']
            .apply(lambda x: x.strftime('%Y%m')).astype(int))
)

# - - - -
# Intitial flagging
# - - - -

# Run outlier heuristic flagging methodology
df_flag = flg.go(df=df,
                 groups=tuple(inputs['stat_groups']),
                 iso_forest_cols=inputs['iso_forest'],
                 dev_bounds=tuple(inputs['dev_bounds']))

# Remove duplicate rows
df_flag = df_flag[df_flag['original_observation']]
# Discard pre-2014 data
df_flag = df_flag[df_flag['meta_sale_date'] >= '2020-01-01']

# Utilize PTAX-203, complete binary columns
df_final = (df_flag
            .rename(columns={'sv_is_outlier': 'sv_is_autoval_outlier'})
            .assign(sv_is_autoval_outlier=lambda df: df['sv_is_autoval_outlier'] == "Outlier")
            .assign(sv_is_outlier=lambda df:
                    df['sv_is_autoval_outlier'] | df['sale_filter_is_outlier'])
            # Incorporate PTAX in sv_outlier_type
            .assign(sv_outlier_type=lambda df:
                    np.where((df['sv_outlier_type'] == "Not outlier") & df['sale_filter_is_outlier'],
                             "PTAX-203 flag", df['sv_outlier_type']))
            # Change sv_is_outlier to binary
            .assign(sv_is_outlier=lambda df: (df['sv_outlier_type'] != "Not outlier").astype(int))
            # PTAX-203 binary
            .assign(sv_is_ptax_outlier=lambda df:
                    np.where(df['sv_outlier_type'] == "PTAX-203 flag", 1, 0))
            # Heuristics flagging binary column
            .assign(sv_is_heuristic_outlier=lambda df:
                    np.where((df['sv_outlier_type'] != 'PTAX-203 flag') & (df['sv_is_outlier'] == 1), 1, 0))
            )

# Manually impute ex values as non-outliers
exempt_to_append = exempt_data.meta_sale_document_num.reset_index().drop(columns='index')
exempt_to_append['sv_is_outlier'] = 0
exempt_to_append['sv_is_ptax_outlier'] = 0
exempt_to_append['sv_is_heuristic_outlier'] = 0
exempt_to_append['sv_outlier_type'] = 'Not Outlier'

cols_to_write = [
    'meta_sale_document_num',
    'sv_is_outlier',
    'sv_is_ptax_outlier',
    'sv_is_heuristic_outlier',
    'sv_outlier_type']

# Merge exempt values and assign run_id
r = RandomWords()
random_word_id = r.get_random_word()
timestamp = datetime.datetime.now(chicago_tz).strftime('%Y-%m-%d_%H:%M')
run_id = random_word_id + '-' + timestamp
df_to_write = pd.concat([df_final[cols_to_write], exempt_to_append]).reset_index(drop=True)
df_to_write['run_id'] = run_id

bucket = 's3://ccao-data-warehouse-us-east-1/sale/flag/'
file_name = 'initial-run.parquet'
s3_file_path = bucket + file_name

wr.s3.to_parquet(
    df=df_to_write,
    path=s3_file_path
)

# - - - - -
# Metadata
# - - - - -

def get_group_means(df: pd.DataFrame, groups: list, means_col: str) -> dict :
    """
    This function gets all group means from the flagging script, for a 
    given column. 

    Inputs:
        df: dataframe after outlier analysis has been performed
        groups: groups used to run the flagging script
        means_col: column for which we want the means

    Outputs:
        result_dict: a dictionary with key as specific grouping and 
        value as the corresponding mean
    """
    cols = groups + [means_col]
    df_to_dict = df.drop_duplicates(subset=means_col, keep='first')[cols]
    
    # Create the 'key' column by concatenating the columns in inputs['stat_groups']
    df_to_dict['key'] = df_to_dict[groups].astype(str).apply('-'.join, axis=1)
    result_dict = dict(zip(df_to_dict['key'], df_to_dict[means_col]))

    return result_dict


# Parameters table
new_sales_flagged = df_to_write.shape[0]
earliest_sale_ingest = df_ingest.meta_sale_date.min()
latest_sale_ingest = df_ingest.meta_sale_date.max()
short_term_owner_threshold = SHORT_TERM_OWNER_THRESHOLD
iso_forest_cols = inputs['iso_forest']
stat_groups = inputs['stat_groups']
dev_bounds = inputs['dev_bounds']
mean_price = get_group_means(df=df_final, 
                             groups=inputs['stat_groups'], 
                             means_col='sv_mean_price_rolling_window_township_code_class')
mean_price_per_sqft = get_group_means(df=df_final, 
                             groups=inputs['stat_groups'], 
                             means_col='sv_mean_price_per_sqft_rolling_window_township_code_class')


parameter_dict_to_df = {
    "run_id": [run_id],
    "new_sales_flagged": [new_sales_flagged],
    "earliest_data_ingest": [earliest_sale_ingest],
    "latest_data_ingest": [latest_sale_ingest],
    "short_term_owner_threshold" : [short_term_owner_threshold],
    "iso_forest_cols" : [iso_forest_cols],
    "stat_groups": [stat_groups],
    "dev_bounds": [dev_bounds],
    "price_group_means": [mean_price],
    "price_group_means_sqft": [mean_price_per_sqft]
}

pd.DataFrame(parameter_dict_to_df)


# Metadata table
commit_sha = sp.getoutput('git rev-parse HEAD')
# add start time, end time, short/long sha

metadata_dict_to_df = {
    "run_id": [run_id],
    "long_commit_sha": commit_sha,
    "short_commit_sha": commit_sha[0:8]
}

pd.DataFrame(metadata_dict_to_df)

