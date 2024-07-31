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
OUTLIER_REASON1_FIELD = "USER27"
OUTLIER_REASON2_FIELD = "TODO1"
OUTLIER_REASON3_FIELD = "TODO2"
RUN_ID_FIELD = "USER28"

OUTLIER_TYPE_CODES = {
    "Low price": "1",
    "High price": "2",
    "Low price per square foot": "3",
    "High price per square foot": "4",
    "Non-person sale": "5",
    "PTAX-203 Exclusion": "6",
    "Statistical Anomaly": "7",
    "Price swing / Home flip": "8",
    "Family sale": "9",
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

    logger.info("Querying the count of active flags")

    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM ({FLAG_LATEST_VERSION_QUERY}) flags")
    expected_num_flags = cursor.fetchall()[0][0]

    logger.info(f"Got {expected_num_flags} active flags")

    FLAG_QUERY = f"""
    SELECT
        sale.pin AS {PIN_FIELD},
        sale.sale_key AS {SALE_KEY_FIELD},
        CASE
            WHEN flag.sv_is_outlier = TRUE
            THEN 'Y'
            ELSE 'N'
        END AS {IS_OUTLIER_FIELD},
        flag.sv_outlier_reason1 AS {OUTLIER_REASON1_FIELD},
        flag.sv_outlier_reason2 AS {OUTLIER_REASON2_FIELD},
        flag.sv_outlier_reason3 AS {OUTLIER_REASON3_FIELD},
        flag.run_id AS {RUN_ID_FIELD}
    FROM sale.flag AS flag
    -- Filter flags for the most recent version
    INNER JOIN ({FLAG_LATEST_VERSION_QUERY}) AS flag_latest_version
        ON flag.meta_sale_document_num = flag_latest_version.meta_sale_document_num
        AND flag.version = flag_latest_version.version
    INNER JOIN default.vw_pin_sale AS sale
        ON flag.meta_sale_document_num = sale.doc_no
    """

    logger.info("Querying sales with flags")

    cursor.execute(FLAG_QUERY)
    flag_df = pyathena.pandas.util.as_pandas(cursor)

    num_flags = len(flag_df.index)
    logger.info(f"Got {num_flags} sales with flags")

    logger.info("Transforming columns")

    outlier_fields = [
        OUTLIER_REASON1_FIELD,
        OUTLIER_REASON2_FIELD,
        OUTLIER_REASON3_FIELD,
    ]

    for field in outlier_fields:
        flag_df[field] = flag_df[field].replace(OUTLIER_TYPE_CODES)

    logger.info("Running data integrity checks")

    # Run some data integrity checks
    not_null_fields = [PIN_FIELD, SALE_KEY_FIELD, RUN_ID_FIELD]
    for field in not_null_fields:
        assert flag_df[flag_df[field].isnull()].empty, f"{field} contains nulls"

    for field in [OUTLIER_REASON1_FIELD, OUTLIER_REASON2_FIELD, OUTLIER_REASON3_FIELD]:
        invalid_values = flag_df[~flag_df[field].isin(OUTLIER_TYPE_CODES.values())]
        assert invalid_values.empty, f"{field} contains invalid codes"

    # Tests confirming that a price outlier reason is needed
    # for a sale to be counted as an outlier, and vice-versa

    PRICE_CODES = [
        "Low price",
        "High price",
        "Low price per square foot",
        "High price per square foot",
    ]

    for price_code in PRICE_CODES:
        assert (
            price_code in OUTLIER_TYPE_CODES
        ), f"{price_code} is in PRICE_CODES but missing from OUTLIER_TYPE_CODES"

    PRICE_CODES = [OUTLIER_TYPE_CODES[code] for code in PRICE_CODES]

    columns_to_check = [
        OUTLIER_REASON1_FIELD,
        OUTLIER_REASON2_FIELD,
        OUTLIER_REASON3_FIELD,
    ]

    mask_all_not = ~flag_df[columns_to_check].isin(PRICE_CODES).any(axis=1)
    mask_any = flag_df[columns_to_check].isin(PRICE_CODES).any(axis=1)

    assert (flag_df[IS_OUTLIER_FIELD] == "N").equals(
        mask_all_not
    ), "If there is a price reason in one of the outlier reason fields, it should be classified as an outlier"

    assert (flag_df[IS_OUTLIER_FIELD] == "Y").equals(
        mask_any
    ), "If there is no price reason in any of the outlier reason fields, it should not be an outlier"

    assert (
        num_flags == expected_num_flags
    ), f"Expected {expected_num_flags} total sales, got {num_flags}"

    logger.info("Writing CSV to stdout")

    flag_df.to_csv(sys.stdout, index=False, chunksize=10000)
