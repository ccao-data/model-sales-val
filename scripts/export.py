#!/usr/bin/env python3
# Export sales val flags to a CSV for upload to iasworld. Outputs the CSV
# to stdout.
#
# Example usage:
#
#   python3 scripts/export.py > export.csv
#
import logging
import os
import sys

import pandas
import pyathena
import pyathena.pandas.util

PIN_FIELD = "PARID"
SALE_KEY_FIELD = "SALEKEY"
IS_OUTLIER_FIELD = "USER26"
OUTLIER_TYPE_FIELD = "USER27"
RUN_ID_FIELD = "USER28"
ANALYST_DETERMINATION_FIELD = "USER29"
ANALYST_REVIEW_DATE_FIELD = "UDATE1"

OUTLIER_TYPE_CODES = {
    "Anomaly (high)": "1",
    "Anomaly (low)": "2",
    "Family sale (high)": "3",
    "Family sale (low)": "4",
    "High price (raw & sqft)": "5",
    "High price (raw)": "6",
    "High price (sqft)": "7",
    "High price swing": "8",
    "Home flip sale (high)": "9",
    "Home flip sale (low)": "10",
    "Low price (raw & sqft)": "11",
    "Low price (raw)": "12",
    "Low price (sqft)": "13",
    "Low price swing": "14",
    "Non-person sale (high)": "15",
    "Non-person sale (low)": "16",
    "PTAX-203 flag (High)": "17",
    "PTAX-203 flag (Low)": "17",
    "Not outlier": pandas.NA,
}

if __name__ == "__main__":
    # Setup a logger that logs to stderr so it does not get captured as part
    # of the script's data output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stderr))

    logger.info("Connecting to Athena")

    conn = pyathena.connect(
        s3_staging_dir=os.getenv(
            "AWS_ATHENA_S3_STAGING_DIR", "s3://ccao-athena-results-us-east-1/"
        ),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

    FLAG_LATEST_VERSION_QUERY = """
        SELECT meta_sale_document_num, MAX(version) AS version
        FROM sale.flag
        GROUP BY meta_sale_document_num
    """

    FLAG_QUERY = f"""
    SELECT
        sale.pin AS {PIN_FIELD},
        sale.sale_key AS {SALE_KEY_FIELD},
        CASE
            WHEN flag.sv_is_outlier = TRUE
            THEN 'Y'
            ELSE 'N'
        END AS {IS_OUTLIER_FIELD},
        flag.sv_outlier_type AS {OUTLIER_TYPE_FIELD},
        flag.run_id AS {RUN_ID_FIELD},
        'N' AS {ANALYST_DETERMINATION_FIELD},
        NULL AS {ANALYST_REVIEW_DATE_FIELD}
    FROM sale.flag AS flag
    -- Filter flags for the most recent version
    INNER JOIN ({FLAG_LATEST_VERSION_QUERY}) AS flag_latest_version
        ON flag.meta_sale_document_num = flag_latest_version.meta_sale_document_num
        AND flag.version = flag_latest_version.version
    INNER JOIN default.vw_pin_sale AS sale
        ON flag.meta_sale_document_num = sale.doc_no
    """

    logger.info("Querying sales")

    cursor = conn.cursor()
    cursor.execute(FLAG_QUERY)
    flag_df = pyathena.pandas.util.as_pandas(cursor)

    num_flags = len(flag_df.index)
    logger.info(f"Got {num_flags} sales")

    logger.info("Querying the latest version of sales for comparison")

    cursor.execute(FLAG_LATEST_VERSION_QUERY)
    flag_latest_version_df = pyathena.pandas.util.as_pandas(cursor)

    expected_num_flags = len(flag_latest_version_df.index)
    logger.info(f"Got {expected_num_flags} sales with latest versions")

    logger.info("Transforming columns")

    # Transform outlier type column from string to code
    flag_df[OUTLIER_TYPE_FIELD] = flag_df[OUTLIER_TYPE_FIELD].replace(
        OUTLIER_TYPE_CODES
    )

    logger.info("Running data integrity checks")

    # Run some data integrity checks
    not_null_fields = [PIN_FIELD, SALE_KEY_FIELD, RUN_ID_FIELD]
    for field in not_null_fields:
        assert flag_df[flag_df[field].isnull()].empty, f"{field} contains nulls"

    assert flag_df[
        ~flag_df[OUTLIER_TYPE_FIELD].isin(OUTLIER_TYPE_CODES.values())
    ].empty, f"{OUTLIER_TYPE_FIELD} contains invalid codes"

    assert flag_df[
        (flag_df[IS_OUTLIER_FIELD] == "Y") & (flag_df[OUTLIER_TYPE_FIELD].isna())
    ].empty, f"{OUTLIER_TYPE_FIELD} cannot be null when {IS_OUTLIER_FIELD} is Y"

    assert flag_df[
        (flag_df[IS_OUTLIER_FIELD] == "N") & (~flag_df[OUTLIER_TYPE_FIELD].isna())
    ].empty, f"{OUTLIER_TYPE_FIELD} must be null when {IS_OUTLIER_FIELD} is N"

    num_flags = len(flag_df.index)
    expected_num_flags = len(flag_latest_version_df.index)
    assert (
        num_flags == expected_num_flags
    ), f"Expected {expected_num_flags} flagged sales, got {num_flags}"

    logger.info("Writing CSV to stdout")

    flag_df.to_csv(sys.stdout, index=False, chunksize=10000)
