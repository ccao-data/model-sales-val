from maunual_flagging.src import flagging_rolling as flg
from manual_flagging.src.flagging_rolling import SHORT_TERM_OWNER_THRESHOLD
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
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(root)

# Set time for run_id
chicago_tz = pytz.timezone("America/Chicago")

# Inputs yaml as inputs
with open("manual_flagging/inputs.yaml", "r") as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Connect to athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
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
    if sql_type in ["decimal"]:
        return "float64"


df = df.astype({col[0]: sql_type_to_pd_type(col[1]) for col in metadata})

# Exempt sale handling
exempt_data = df[df["class"] == "EX"]
df = df[df["class"] != "EX"]

# - - - - - - - -
# Create rolling window
# - - - - - - - -
max_date = df["meta_sale_date"].max()

df = (
    # Creates dt column with 12 month dates
    df.assign(
        rolling_window=df["meta_sale_date"].apply(
            lambda x: pd.date_range(start=x, periods=12, freq="M")
        )
    )
    # Expand rolling_windows dates to individual rows
    .explode("rolling_window")
    # Tag original observations
    .assign(
        original_observation=lambda df: df["meta_sale_date"].dt.month
        == df["rolling_window"].dt.month
    )
    # Simplify to month level
    .assign(rolling_window=lambda df: df["rolling_window"].dt.to_period("M"))
    # Filter such that rolling_window isn't extrapolated into future, we are
    # concerned with historic and present-month data
    .loc[lambda df: df["rolling_window"] <= max_date.to_period("M")]
    # Back to float for flagging script
    .assign(
        rolling_window=lambda df: df["rolling_window"]
        .apply(lambda x: x.strftime("%Y%m"))
        .astype(int)
    )
)


# - - - -
# Intitial flagging
# - - - -

# Run outlier heuristic flagging methodology
df_flag = flg.go(
    df=df,
    groups=tuple(inputs["stat_groups"]),
    iso_forest_cols=inputs["iso_forest"],
    dev_bounds=tuple(inputs["dev_bounds"]),
)

# Remove duplicate rows
df_flag = df_flag[df_flag["original_observation"]]
# Discard pre-2014 data
df_flag = df_flag[df_flag["meta_sale_date"] >= "2020-01-01"]

# Utilize PTAX-203, complete binary columns
df_final = (
    df_flag.rename(columns={"sv_is_outlier": "sv_is_autoval_outlier"})
    .assign(
        sv_is_autoval_outlier=lambda df: df["sv_is_autoval_outlier"] == "Outlier",
        sv_is_outlier=lambda df: df["sv_is_autoval_outlier"] | df["sale_filter_is_outlier"],
        # Incorporate PTAX in sv_outlier_type
        sv_outlier_type=lambda df: np.where(
            (df["sv_outlier_type"] == "Not outlier") & df["sale_filter_is_outlier"],
            "PTAX-203 flag",
            df["sv_outlier_type"],
        ),
    )
    .assign(
        # Change sv_is_outlier to binary
        sv_is_outlier=lambda df: (df["sv_outlier_type"] != "Not outlier").astype(int),
        # PTAX-203 binary
        sv_is_ptax_outlier=lambda df: np.where(df["sv_outlier_type"] == "PTAX-203 flag", 1, 0),
        # Heuristics flagging binary column
        sv_is_heuristic_outlier=lambda df: np.where(
            (df["sv_outlier_type"] != "PTAX-203 flag") & (df["sv_is_outlier"] == 1), 1, 0
        ),
    )
)

# Manually impute ex values as non-outliers
exempt_to_append = exempt_data.meta_sale_document_num.reset_index().drop(columns="index")
exempt_to_append["sv_is_outlier"] = 0
exempt_to_append["sv_is_ptax_outlier"] = 0
exempt_to_append["sv_is_heuristic_outlier"] = 0
exempt_to_append["sv_outlier_type"] = "Not Outlier"

cols_to_write = [
    "meta_sale_document_num",
    "rolling_window",
    "sv_is_outlier",
    "sv_is_ptax_outlier",
    "sv_is_heuristic_outlier",
    "sv_outlier_type",
]

# Create run_id
r = RandomWords()
random_word_id = r.get_random_word()
timestamp = datetime.datetime.now(chicago_tz).strftime("%Y-%m-%d_%H:%M")
run_id = timestamp + "-" + random_word_id

# Incorporate exempt values and finalize to write to flag table
df_to_write = (
    # TODO: exempt will have an NA for rolling_window - make sure that is okay
    pd.concat([df_final[cols_to_write], exempt_to_append])
    .reset_index(drop=True)
    .assign(
        run_id=run_id,
        version=1,
        rolling_window=lambda df: pd.to_datetime(df["rolling_window"], format="%Y%m").dt.date
    )
)

bucket = "s3://ccao-data-warehouse-us-east-1/sale/flag/"
file_name = "initial-run.parquet"
s3_file_path = bucket + file_name

wr.s3.to_parquet(df=df_to_write, path=s3_file_path)

# - - - - -
# Metadata / Params / Means
# - - - - -

# Parameters table
sales_flagged = df_to_write.shape[0]
earliest_sale_ingest = df_ingest.meta_sale_date.min()
latest_sale_ingest = df_ingest.meta_sale_date.max()
short_term_owner_threshold = SHORT_TERM_OWNER_THRESHOLD
iso_forest_cols = inputs["iso_forest"]
stat_groups = inputs["stat_groups"]
dev_bounds = inputs["dev_bounds"]

parameter_dict_to_df = {
    "run_id": [run_id],
    "sales_flagged": [sales_flagged],
    "earliest_data_ingest": [earliest_sale_ingest],
    "latest_data_ingest": [latest_sale_ingest],
    "short_term_owner_threshold": [short_term_owner_threshold],
    "iso_forest_cols": [iso_forest_cols],
    "stat_groups": [stat_groups],
    "dev_bounds": [dev_bounds],
}

df_parameters = pd.DataFrame(parameter_dict_to_df)

bucket = "s3://ccao-data-warehouse-us-east-1/sale/parameter/"
file_name = run_id + ".parquet"
s3_file_path = bucket + file_name

wr.s3.to_parquet(df=df_parameters, path=s3_file_path)

# Means Table
unique_groups = (
    df_final.drop_duplicates(subset=inputs["stat_groups"], keep="first")
    .reset_index(drop=True)
    .assign(
        rolling_window=lambda df: pd.to_datetime(df["rolling_window"], format="%Y%m").dt.date
        )
    )


cols_to_write_means = [
    "rolling_window",
    "township_code",
    "class",
    "sv_mean_price_rolling_window_township_code_class",
    "sv_mean_price_per_sqft_rolling_window_township_code_class",
]


df_means = unique_groups[cols_to_write_means]

# Make columns less verbose
df_means = df_means.rename(columns={
    "sv_mean_price_rolling_window_township_code_class": "mean_price_grouped",
    "sv_mean_price_per_sqft_rolling_window_township_code_class": "mean_price_sqft_grouped",
})

df_means["run_id"] = run_id

bucket = "s3://ccao-data-warehouse-us-east-1/sale/group_mean/"
file_name = run_id + ".parquet"
s3_file_path = bucket + file_name

wr.s3.to_parquet(df=df_means, path=s3_file_path)


# Metadata table
commit_sha = sp.getoutput("git rev-parse HEAD")

metadata_dict_to_df = {
    "run_id": [run_id],
    "long_commit_sha": commit_sha,
    "short_commit_sha": commit_sha[0:8],
    "run_timestamp": timestamp,
    "run_type": "initial_flagging"
}

df_metadata = pd.DataFrame(metadata_dict_to_df)

bucket = "s3://ccao-data-warehouse-us-east-1/sale/metadata/"
file_name = run_id + ".parquet"
s3_file_path = bucket + file_name

wr.s3.to_parquet(df=df_metadata, path=s3_file_path)
