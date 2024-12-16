# TODO: Add context
import os
import awswrangler as wr
import pandas as pd
import numpy as np
from glue import sales_val_flagging as flg
import subprocess as sp
from pyathena import connect
from pyathena.pandas.util import as_pandas

pedantic_matt_path = os.path.join(
    os.getenv("AWS_BUCKET_SV_BACKUP"),
    "0004_remove_mydec_flags",
    "old_data_file",
    "2024-12-10_14:17-pedantic-matt.parquet",
)

compassionate_rina_path = os.path.join(
    os.getenv("AWS_BUCKET_SV_BACKUP"),
    "0004_remove_mydec_flags",
    "old_data_file",
    "2024-12-10_13:30-compassionate-rina.parquet",
)

df_to_edit_pedantic_matt = wr.s3.read_parquet(pedantic_matt_path)
df_to_edit_compassionate_rina = wr.s3.read_parquet(compassionate_rina_path)

# # # - - - - - -
# The task here is to, after flagging 2023 and onward ias sales,
# find flagged doc_no's that are in both of the above files
# and NOT in default.vw_pin_sale, and then remove those observations
#
# These sales in theory should be isolated to 2023 and 2024
#
# # # - - - - - -

# Connect to Athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
)

# Fetch sales and characteristics from Athena
SQL_QUERY_PIN_SALE = """
select *
from default.vw_pin_sale
where sv_is_outlier is not null
"""

# Execute query and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY_PIN_SALE)
metadata = cursor.description

df_ingest = as_pandas(cursor)
df = df_ingest


# wr.s3.to_parquet(df=df, path=file_path, index=False)
