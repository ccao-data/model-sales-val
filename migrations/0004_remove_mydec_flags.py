# TODO: Add context
import os
import awswrangler as wr
import pandas as pd
import numpy as np
from glue import sales_val_flagging as flg
import subprocess as sp
from pyathena import connect
from pyathena.pandas.util import as_pandas
from pyathena.pandas.cursor import PandasCursor

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
# Connect to Athena
cursor = connect(
    # We add '+ "/"' to the end of the line below because enabling unload
    # requires that the staging directory end with a slash
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR") + "/",
    region_name=os.getenv("AWS_REGION"),
    cursor_class=PandasCursor,
).cursor(unload=True)


# Fetch sales and characteristics from Athena
SQL_QUERY_PIN_SALE = """
select *
from default.vw_pin_sale
where sv_is_outlier is not null
"""

# Execute query and return as pandas df
cursor.execute(SQL_QUERY_PIN_SALE)
df = cursor.as_pandas()


# wr.s3.to_parquet(df=df, path=file_path, index=False)

# Normalize to string to avoid dtype mismatches
df_ids = set(df["doc_no"].astype(str))

# Filter pedantic_matt
df_to_edit_pedantic_matt_filtered = df_to_edit_pedantic_matt[
    df_to_edit_pedantic_matt["meta_sale_document_num"].astype(str).isin(df_ids)
]

# Filter compassionate_rina
df_to_edit_compassionate_rina_filtered = df_to_edit_compassionate_rina[
    df_to_edit_compassionate_rina["meta_sale_document_num"].astype(str).isin(df_ids)
]
