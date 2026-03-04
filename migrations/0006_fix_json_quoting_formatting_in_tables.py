"""
This migration fixes JSON quoting in sale.parameter. All JSON columns except
standard_deviation_bounds were saved using Python's str() representation, which
uses single-quotes for string keys. Per the JSON spec, string keys must use
double-quotes. Single-quotes cause parsing errors in Python and break Athena
JSON operations like json_extract().

This script reads existing prod parameter data (backed up to the prior/ prefix),
converts all affected JSON columns from single-quote to double-quote format using
ast.literal_eval + json.dumps, and writes the corrected files to the updated/ prefix
for QC before copying back to prod.
"""

import ast
import json
import os

import awswrangler as wr

# Columns stored as str(dict) that need to be converted to proper JSON (double-quoted).
# standard_deviation_bounds is excluded as it was already saved with json.dumps.
JSON_COLUMNS = [
    "run_filter",
    "iso_forest_cols",
    "stat_groups",
    "sales_to_write_filter",
    "housing_market_class_codes",
    "time_frame",
]

BACKUP_BASE = "s3://ccao-data-backup-us-east-1/model-sales-val/migrations/0006_fix_json_quoting_formatting_in_tables"

parquet_files_prod_prior = wr.s3.list_objects(
    os.path.join(BACKUP_BASE, "parameter_prior/"),
    suffix=".parquet",
)

dfs_parameter_prod_prior = {}
dfs_parameter_prod_updated = {}

# Read existing prod data (copied into backup bucket from prod)
for file in parquet_files_prod_prior:
    name_part = file.split("/")[-1].split(".")[0]
    df = wr.s3.read_parquet(file)
    dfs_parameter_prod_prior[name_part] = df


def fix_json_quoting(value):
    """
    Converts a Python str()-style representation (single-quoted keys/values)
    to a proper double-quoted JSON string. Empty strings are returned as-is.
    """
    if not isinstance(value, str) or value == "":
        return value
    return json.dumps(ast.literal_eval(value))


# Apply double-quote fix to all affected JSON columns
for name, df in dfs_parameter_prod_prior.items():
    updated_df = df.copy()
    for col in JSON_COLUMNS:
        if col in updated_df.columns:
            updated_df[col] = updated_df[col].apply(fix_json_quoting)
    dfs_parameter_prod_updated[name] = updated_df

output_prefix = os.path.join(BACKUP_BASE, "parameter_updated/")

for name, df in dfs_parameter_prod_updated.items():
    output_path = os.path.join(output_prefix, f"{name}.parquet")
    wr.s3.to_parquet(
        df=df,
        path=output_path,
        index=False,
    )
