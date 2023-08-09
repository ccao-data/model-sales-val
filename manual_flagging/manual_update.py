from manual_flagging.src import flagging_rolling as flg
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

# Set working to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, 'manual_flagging'))

# Set time for run_id
chicago_tz = pytz.timezone("America/Chicago")

# Inputs yaml as inputs
with open(os.path.join("yaml", "inputs_update.yaml"), "r") as stream:
    try:
        inputs = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Connect to athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
)

# Parse yaml to get which sales to flag
if inputs["time_frame"]["end"] == None:
    sql_time_frame = f"sale.sale_date >= DATE '{inputs['time_frame']['start']}'"
else:
    sql_time_frame = f"""(sale.sale_date 
        BETWEEN DATE '{inputs['time_frame']['start']}'
        AND DATE '{inputs['time_frame']['end']}')"""

SQL_QUERY = f"""
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
WHERE {sql_time_frame}
AND NOT sale.is_multisale
AND NOT res.pin_is_multicard
"""

SQL_QUERY_SALES_VAL = """
SELECT *
FROM sale.flag
"""

# Execute queries and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description
df_ingest = as_pandas(cursor)
df = df_ingest

cursor.execute(SQL_QUERY_SALES_VAL)
df_ingest_flag = as_pandas(cursor)
df_flag = df_ingest_flag

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
df_flagged = flg.go(
    df=df,
    groups=tuple(inputs["stat_groups"]),
    iso_forest_cols=inputs["iso_forest"],
    dev_bounds=tuple(inputs["dev_bounds"]),
)

# Remove duplicate rows
df_flagged = df_flagged[df_flagged["original_observation"]]
# Discard pre-2014 data
df_flagged = df_flagged[df_flagged["meta_sale_date"] >= "2019-01-01"]

# Utilize PTAX-203, complete binary columns
df_finish_flagging = (
    df_flagged.rename(columns={"sv_is_outlier": "sv_is_autoval_outlier"})
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
df_final = (
    # TODO: exempt will have an NA for rolling_window - make sure that is okay
    pd.concat([df_finish_flagging[cols_to_write], exempt_to_append])
    .reset_index(drop=True)
    .assign(
        run_id=run_id,
        rolling_window=lambda df: pd.to_datetime(df["rolling_window"], format="%Y%m").dt.date,
    )
)

# - - - - - -
# Update version of re-flagged entries
# - - - - - -

# Group the existing data by 'ID' and find the maximum 'version' for each sale
existing_max_version = (
    df_flag.groupby("meta_sale_document_num")["version"]
    .max()
    .reset_index()
    .rename(columns={"version": "existing_version"})
)

# Merge, compute new version, and drop unnecessary columns
df_to_write = (
    df_final.merge(existing_max_version, on="meta_sale_document_num", how="left")
    .assign(
        version=lambda x: x["existing_version"]
        .apply(lambda y: y + 1 if pd.notnull(y) else 1)
        .astype(int)
    )
    .drop(columns=["existing_version"])
)

# - - - - -
# Write to flag table
# - - - - -

file_name = run_id + "initial-run.parquet"
s3_file_path = os.path.join(os.getenv("AWS_S3_WAREHOUSE_BUCKET"), 'sale', 'flag', file_name)
wr.s3.to_parquet(df=df_to_write, path=s3_file_path)

# - - - - -
# Write to parameter table
# - - - - -

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

file_name = run_id + ".parquet"
s3_file_path = os.path.join(os.getenv("AWS_S3_WAREHOUSE_BUCKET"), 'sale', 'parameter', file_name)
wr.s3.to_parquet(df=df_parameters, path=s3_file_path)

# - - - - -
# Write to group_mean table
# - - - - -

unique_groups = (
    df_finish_flagging.drop_duplicates(subset=inputs["stat_groups"], keep="first")
    .reset_index(drop=True)
    .assign(rolling_window=lambda df: pd.to_datetime(df["rolling_window"], format="%Y%m").dt.date)
)

groups_string_col = "_".join(map(str, inputs["stat_groups"]))
suffixes = ["mean_price", "mean_price_per_sqft"]

cols_to_write_means = inputs["stat_groups"] + [
    f"sv_{suffix}_{groups_string_col}" for suffix in suffixes
]
rename_dict = {f"sv_{suffix}_{groups_string_col}": f"{suffix}" for suffix in suffixes}

df_means = (
    unique_groups[cols_to_write_means]
    .rename(columns=rename_dict)
    .assign(
        run_id=run_id,
        group=lambda df: df[inputs["stat_groups"]].astype(str).apply("_".join, axis=1),
    )
    .drop(columns=inputs["stat_groups"])
)

file_name = run_id + ".parquet"
s3_file_path = os.path.join(os.getenv("AWS_S3_WAREHOUSE_BUCKET"), 'sale', 'group_mean', file_name)
wr.s3.to_parquet(df=df_means, path=s3_file_path)

# - - - - -
# Write to metadata table
# - - - - -

commit_sha = sp.getoutput("git rev-parse HEAD")

metadata_dict_to_df = {
    "run_id": [run_id],
    "long_commit_sha": commit_sha,
    "short_commit_sha": commit_sha[0:8],
    "run_timestamp": timestamp,
    "run_type": "manual_update",
}

df_metadata = pd.DataFrame(metadata_dict_to_df)

file_name = run_id + ".parquet"
s3_file_path = os.path.join(os.getenv("AWS_S3_WAREHOUSE_BUCKET"), 'sale', 'metadata', file_name)
wr.s3.to_parquet(df=df_metadata, path=s3_file_path)