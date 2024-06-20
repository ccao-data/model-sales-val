from glue.flagging_script_glue import flagging as flg_model
from glue import sales_val_flagging as flg
import awswrangler as wr
import copy
import os
import datetime
import numpy as np
import pandas as pd
import pytz
import subprocess as sp
import yaml
from pyathena import connect
from pyathena.pandas.util import as_pandas

###########################
# PARSE INPUTS
###########################

# Set working dir to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "manual_flagging"))

# Use yaml as inputs
with open(os.path.join("yaml", "inputs.yaml"), "r") as stream:
    inputs = yaml.safe_load(stream)

# Validate the input specification
# Check housing_market_type
# TODO: Add res_all and other res_type exclusivity check
assert "housing_market_type" in inputs, "Missing key: 'housing_market_type'"
assert set(inputs["housing_market_type"]).issubset(
    {"res_single_family", "res_multi_family", "condos", "res_all"}
), "housing_market_type can only contain 'res_single_family', 'res_multi_family', 'condos', 'res_all'"
assert len(inputs["housing_market_type"]) == len(
    set(inputs["housing_market_type"])
), "Duplicate values in 'housing_market_type'"

# Check run_tri
assert "run_tri" in inputs, "Missing key: 'run_tri'"
assert set(inputs["run_tri"]).issubset({1, 2, 3}), "run_tri can only contain 1, 2, 3"
assert len(inputs["run_tri"]) == len(
    set(inputs["run_tri"])
), "Duplicate values in 'run_tri'"

###########################
# INGEST
###########################

# Connect to Athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
)

# Calculate the earliest date needed in order to satisfy the rolling window
# before the configured start date
start_date = datetime.datetime.strptime(
    inputs["time_frame"]["start"], "%Y-%m-%d"
)
result_date = start_date - relativedelta(
    months=inputs["rolling_window_months"] - 1
)
date_floor = result_date.replace(day=1).strftime("%Y-%m-%d")

# Parse yaml to get which sales to flag
if inputs["time_frame"]["end"] == None:
    sql_time_frame = f"sale.sale_date >= DATE '{date_floor}'"
else:
    sql_time_frame = f"""(sale.sale_date
        BETWEEN DATE '{date_floor}'
        AND DATE '{inputs['time_frame']['end']}')"""

# Fetch sales and characteristics from Athena
SQL_QUERY = f"""
WITH CombinedData AS (
    -- Select data from vw_card_res_char
    SELECT
        'res_char' AS source_table,
        'res' AS indicator,  -- Indicator column for 'res'
        res.class AS class,
        res.township_code AS township_code,
        res.year AS year,
        res.pin AS pin,
        res.char_bldg_sf AS char_bldg_sf,
        CAST(DATE_FORMAT(CURRENT_DATE, '%Y') AS INT) - res.char_yrblt AS char_bldg_age,
        res.pin_is_multicard
    FROM default.vw_card_res_char res
    WHERE res.class IN (
        '202', '203', '204', '205', '206', '207', '208', '209',
        '210', '211', '212', '218', '219', '234', '278', '295'
    )

    UNION ALL

    -- Selecting data from vw_pin_condo_char
    SELECT
        'condo_char' AS source_table,
        'condo' AS indicator,  -- Indicator column for 'condo'
        condo.class AS class,
        condo.township_code AS township_code,
        condo.year AS year,
        condo.pin AS pin,
        NULL AS char_bldg_sf,
        NULL AS char_bldg_age,
        FALSE AS pin_is_multicard
    FROM default.vw_pin_condo_char condo
    WHERE condo.class IN ('297', '299', '399')
    AND NOT condo.is_parking_space
    AND NOT condo.is_common_area
),

-- Select neighborhood groups and filter for most recent versions
neighborhood_group AS (
    SELECT nbhd_group.nbhd, nbhd_group.group_name
    FROM location.neighborhood_group AS nbhd_group
    INNER JOIN (
        SELECT nbhd, MAX(version) AS version
        FROM location.neighborhood_group
        GROUP BY nbhd
    ) AS latest_group_version
        ON nbhd_group.nbhd = latest_group_version.nbhd
        AND nbhd_group.version = latest_group_version.version
)

-- Now, join with sale table and filters
SELECT
    sale.sale_price AS meta_sale_price,
    sale.sale_date AS meta_sale_date,
    sale.doc_no AS meta_sale_document_num,
    sale.seller_name AS meta_sale_seller_name,
    sale.buyer_name AS meta_sale_buyer_name,
    sale.nbhd as nbhd,
    nbhd_group.group_name as geography_split,
    sale.sale_filter_ptax_flag AS ptax_flag_original,
    data.class,
    data.township_code,
    data.year,
    data.pin,
    data.char_bldg_sf,
    data.char_bldg_age,
    data.indicator,
    universe.triad_code 
FROM CombinedData data
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = data.pin
    AND sale.year = data.year
INNER JOIN default.vw_pin_universe universe 
    ON universe.pin = data.pin
    AND universe.year = data.year
LEFT JOIN neighborhood_group nbhd_group
    ON sale.nbhd = nbhd_group.nbhd
WHERE {sql_time_frame}
AND NOT sale.sale_filter_same_sale_within_365
AND NOT sale.sale_filter_less_than_10k
AND NOT sale.sale_filter_deed_type
AND NOT sale.is_multisale
AND (
    NOT data.pin_is_multicard
    OR data.source_table = 'condo_char'
)

"""


# Execute query and return as pandas df
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description

df = as_pandas(cursor)
earliest_sale_ingest = df.meta_sale_date.min()
latest_sale_ingest = df.meta_sale_date.max()

if inputs["manual_update"] == True:
    SQL_QUERY_SALES_VAL = """
    SELECT *
    FROM sale.flag
    """
    cursor.execute(SQL_QUERY_SALES_VAL)
    df_ingest_flag = as_pandas(cursor)
    # Group the existing data by 'ID' and find the maximum 'version' for each sale
    existing_max_version = (
        df_ingest_flag.groupby("meta_sale_document_num")["version"]
        .max()
        .reset_index()
        .rename(columns={"version": "existing_version"})
    )

# Convert data types
conversion_dict = {
    col[0]: flg.sql_type_to_pd_type(col[1])
    for col in metadata
    if flg.sql_type_to_pd_type(col[1]) is not None
}
df = df.astype(conversion_dict)
df["ptax_flag_original"].fillna(False, inplace=True)


###############
# PARTITION
###############


# - - - - - - -
# Make correct filters and set up dictionary structure
# Split tris into groups according to their flagging methods
# - - - - - - -

dfs_to_rolling_window = {}  # Dictionary to store DataFrames

for tri in inputs["run_tri"]:
    # Iterate over housing types defined in yaml
    for housing_type in inputs["housing_market_type"]:
        if housing_type not in inputs["stat_groups"][f"tri{tri}"]:
            print(
                f"Skipping flags for '{housing_type}' in tri {tri} since the market is not defined "
                "as a stat group in the tri"
            )
            continue
        # Assign the df a name
        key = f"df_tri{tri}_{housing_type}"

        # Perform filtering based on tri and housing market class codes
        triad_code_filter = df["triad_code"] == str(tri)
        market_filter = df["class"].isin(
            inputs["housing_market_class_codes"][housing_type]
        )

        # Initialize the DataFrame for the current key
        df_filtered = df[triad_code_filter & market_filter].copy()

        # Assign rolling windows
        max_date = df_filtered["meta_sale_date"].max()
        df_filtered = (
            # Creates dt column with 12 month dates
            df_filtered.assign(
                rolling_window=df["meta_sale_date"].apply(
                    lambda x: pd.date_range(
                        start=x,
                        periods=inputs["rolling_window_months"],
                        freq="M"
                    )
                )
            )
            # Expand rolling_windows dates to individual rows
            .explode("rolling_window")
            # Tag original observations
            .assign(
                original_observation=lambda df: (
                    df["meta_sale_date"].dt.month == df["rolling_window"].dt.month
                )
                & (df["meta_sale_date"].dt.year == df["rolling_window"].dt.year)
            )
            # Simplify to month level
            .assign(rolling_window=lambda df: df["rolling_window"].dt.to_period("M"))
            # Filter such that rolling_window isn't extrapolated into future, we are concerned with historic and present-month data
            .loc[lambda df: df["rolling_window"] <= max_date.to_period("M")]
            # Back to float for flagging script
            .assign(
                rolling_window=lambda df: df["rolling_window"]
                .apply(lambda x: x.strftime("%Y%m"))
                .astype(int),
                meta_sale_price_original=lambda df: df["meta_sale_price"],
            )
        )
        dfs_to_rolling_window[key] = {"df": df_filtered}

        # Extract the specific housing type configuration
        housing_type_config = inputs["stat_groups"][f"tri{tri}"][housing_type]

        # Perform column transformations
        columns = housing_type_config["columns"]
        transformed_columns = []

        for col in columns:
            if isinstance(col, dict):
                # Validate the structure of the column dictionary
                for required_attr in ("column", "bins"):
                    if required_attr not in col:
                        raise ValueError(
                            "stat_groups column dict is missing required "
                            f"'{required_attr}' attribute: {col}"
                        )

                if "bins" in col:
                    # Initialize bins with 0 and float("inf")
                    bins = [0] + col["bins"] + [float("inf")]

                    # Generate labels based on bins
                    labels = []
                    for i in range(len(bins) - 1):
                        if i == 0:
                            labels.append(f"below-{bins[i+1]}")
                        elif i == len(bins) - 2:
                            labels.append(f"above-{bins[i]}")
                        else:
                            labels.append(f"{bins[i]}-to-{bins[i+1]}")

                    new_col_name = f"{col['column']}_bin"
                    df_filtered[new_col_name] = pd.cut(
                        df_filtered[col["column"]], bins=bins, labels=labels
                    )
                    transformed_columns.append(new_col_name)
            else:
                transformed_columns.append(col)

        dfs_to_rolling_window[key]["columns"] = transformed_columns

        # Add rest of config information
        dfs_to_rolling_window[key]["iso_forest_cols"] = inputs["iso_forest"][
            "res" if "res" in housing_type else "condos"
        ]
        dfs_to_rolling_window[key]["condos_boolean"] = housing_type == "condos"
        dfs_to_rolling_window[key]["market"] = housing_type


################
# FLAG
################

# - - - - -
# Flag Sales
# - - - - -

dfs_flagged = copy.deepcopy(dfs_to_flag)

for df_name, df_info in dfs_to_flag.items():
    print(f"\nFlagging sales for {df_name}")
    df_copy = df_info["df"].copy()
    df_copy = flg_model.go(
        df=df_copy,
        groups=tuple(df_info["columns"]),
        iso_forest_cols=df_info["iso_forest_cols"],
        dev_bounds=tuple(inputs["dev_bounds"]),
        condos=df_info["condos_boolean"],
    )

    # Add the edited or unedited dataframe to the new dictionary
    dfs_flagged[df_name]["df"] = df_copy


#########################
# CLASSIFY
#########################


# - - - - - - - - - - -
# Adjust outliers based on group sizes and incorporate ptax information
# - - - - - - - - - - -

dfs_to_finalize = copy.deepcopy(dfs_flagged)

for df_name, df_info in dfs_flagged.items():
    # Make a copy of the data frame to edit
    print(f"\n Enacting group threshold and creating ptax data for {df_name}")
    df_copy = df_info["df"].copy()

    group_string = "_".join(df_info["columns"])

    # --------------------------
    # Add PTAX indicator columns
    # --------------------------

    if not df_info["condos_boolean"]:
        df_copy["sv_ind_ptax_flag_w_high_price"] = df_copy["ptax_flag_original"] & (
            (df_copy[f"sv_price_deviation_{group_string}"] >= inputs["ptax_sd"][1])
        )

        df_copy["sv_ind_ptax_flag_w_high_price_sqft"] = df_copy["ptax_flag_original"] & (
            (df_copy[f"sv_price_per_sqft_deviation_{group_string}"] >= inputs["ptax_sd"][1])
        )

        df_copy["sv_ind_ptax_flag_w_low_price"] = df_copy["ptax_flag_original"] & (
            (df_copy[f"sv_price_per_sqft_deviation_{group_string}"] <= -inputs["ptax_sd"][0])
        )

        df_copy["sv_ind_ptax_flag_w_low_price_sqft"] = df_copy["ptax_flag_original"] & (
            (df_copy[f"sv_price_per_sqft_deviation_{group_string}"] <= -inputs["ptax_sd"][0])
        )

    else:
        df_copy["sv_ind_ptax_flag_w_high_price"] = df_copy["ptax_flag_original"] & (
            (df_copy[f"sv_price_deviation_{group_string}"] >= inputs["ptax_sd"][1])
        )

        df_copy["sv_ind_ptax_flag_w_low_price"] = df_copy["ptax_flag_original"] & (
            (df_copy[f"sv_price_deviation_{group_string}"] <= -inputs["ptax_sd"][0])
        )

    df_copy["sv_ind_ptax_flag"] = df_copy["ptax_flag_original"].astype(int)

    # ---------------------------------------
    # Add all other outlier indicator columns
    # ---------------------------------------

    stat_groups = df_info["columns"]
    group_counts = df_copy.groupby(stat_groups).size().reset_index(name="count")
    filtered_groups = group_counts[group_counts["count"] <= inputs["min_groups_threshold"]]

    # Merge df_copy_flagged with filtered_groups on the columns to get the matching rows
    df_copy = pd.merge(
        df_copy, filtered_groups[stat_groups], on=stat_groups, how="left", indicator=True
    )

    # Assign blank sv_outlier_reasons
    for idx in range(1, 4):
        df_copy[f"sv_outlier_reason{idx}"] = np.nan

    outlier_type_crosswalk = {
        "sv_ind_price_high_price": "High price",
        "sv_ind_ptax_flag_w_high_price": "High price",
        "sv_ind_price_low_price": "Low price",
        "sv_ind_ptax_flag_w_low_price": "Low price",
        "sv_ind_price_high_price_sqft": "High price per square foot",
        "sv_ind_ptax_flag_w_high_price_sqft": "High price per square foot",
        "sv_ind_price_low_price_sqft": "Low price per square foot",
        "sv_ind_ptax_flag_w_low_price_sqft": "Low price per square foot",
        "sv_ind_ptax_flag": "PTAX-203 Exclusion",
        "sv_ind_char_short_term_owner": "Short-term owner",
        "sv_ind_char_family_sale": "Family Sale",
        "sv_ind_char_non_person_sale": "Non-person sale",
        "sv_ind_char_statistical_anomaly": "Statistical Anomaly",
        "sv_ind_char_price_swing_homeflip": "Price swing / Home flip",
    }

    # During our statistical flagging process, we automatically discard
    # a sale's eligibility for outlier status if the number of sales in
    # the statistical grouping is below a certain threshold. The list
    # `group_thresh_price_fix` along with the ['_merge'] column will allow
    # us to exclude these sales for the sv_is_outlier status.
    #
    # Since the `sv_is_outlier` column requires a price value, we simply
    # do not assign these price outlier flags if the group number is below a certain
    # threshold
    #
    # Note: This doesn't apply for sales that also have a ptax outlier status.
    #       In this case, we still assign the price outlier status.

    group_thresh_price_fix = [
        "sv_ind_price_high_price",
        "sv_ind_price_low_price",
        "sv_ind_price_high_price_sqft",
        "sv_ind_price_low_price_sqft",
    ]

    def fill_outlier_reasons(row):
        reasons_added = set()  # Set to track reasons already added

        for reason_ind_col in outlier_type_crosswalk:
            current_reason = outlier_type_crosswalk[reason_ind_col]
            # Add a check to ensure that only specific reasons are added if _merge is not 'both'
            if (
                reason_ind_col in row
                and row[reason_ind_col]
                and current_reason
                not in reasons_added  # Check if the reason is already added
                # Apply group threshold logic: `row["_merge"]` will be `both` when the group threshold
                # is not met, but only price indicators (`group_thresh_price_fix`) should use this threshold,
                # since ptax indicators don't currently utilize group threshold logic
                and not (
                    row["_merge"] == "both" and reason_ind_col in group_thresh_price_fix
                )
            ):
                row[f"sv_outlier_reason{len(reasons_added) + 1}"] = current_reason
                reasons_added.add(current_reason)  # Add current reason to the set
                if len(reasons_added) >= 3:
                    break

        return row

    df_copy = df_copy.apply(fill_outlier_reasons, axis=1)

    # Drop the _merge column
    df_copy = df_copy.drop(columns=["_merge"])

    # Assign outlier status
    values_to_check = {
        "High price",
        "Low price",
        "High price per square foot",
        "Low price per square foot",
    }

    df_copy["sv_is_outlier"] = np.where(
        df_copy[[f"sv_outlier_reason{idx}" for idx in range(1, 4)]]
        .isin(values_to_check)
        .any(axis=1),
        True,
        False,
    )

    # Add group column to eventually write to athena sale.flag table
    df_copy["group"] = df_copy.apply(
        # Modify the 'group' column by appending '-market_value', this is done
        # to make sure that a two different groups with the same run_id won't
        # be returned with the same value. For example, if res and condos have the
        # same column groupings, joining the group column from sale.flag to sale.group_mean
        # by 'group' and 'run_id' could potentially return two groups. This market type
        # append fixes that. This is also added in the group_mean data.
        lambda row: "_".join([str(row[col]) for col in stat_groups]) + "-" + df_info["market"],
        axis=1
    )

    df_copy = df_copy.assign(
        # PTAX-203 binary
        sv_is_ptax_outlier=lambda df: (df["sv_is_outlier"] == True)
        & (df["sv_ind_ptax_flag"] == 1),
        sv_is_heuristic_outlier=lambda df: (~df["sv_ind_ptax_flag"] == 1)
        & (df["sv_is_outlier"] == True),
    )

    # Add the edited or unedited dataframe to the new dictionary
    dfs_to_finalize[df_name]["df"] = df_copy


################
# FINALIZE
################


# - - - - - - -
# Finalize data to write and create data for all metadata tables
# - - - - - - - -


dfs_to_finalize_list = [details["df"] for details in dfs_to_finalize.values()]
df_to_finalize = pd.concat(dfs_to_finalize_list, axis=0)

# Remove duplicate rows
df_to_finalize = df_to_finalize[df_to_finalize["original_observation"]]
# Discard pre-2014 data
df_to_finalize = df_to_finalize[df_to_finalize["meta_sale_date"] >= start_date]

sales_to_write_filter = inputs["sales_to_write_filter"]
if sales_to_write_filter["column"]:
    df_to_finalize = df_to_finalize[
        df_to_finalize[sales_to_write_filter["column"]].isin(sales_to_write_filter["values"])
    ]

cols_to_write = [
    "meta_sale_document_num",
    "meta_sale_price_original",
    "rolling_window",
    "sv_is_outlier",
    "sv_outlier_reason1",
    "sv_outlier_reason2",
    "sv_outlier_reason3",
    "sv_is_ptax_outlier",
    "ptax_flag_original",
    "sv_is_heuristic_outlier",
    "group",
]

# Create run_id
left = pd.read_csv(
    "https://raw.githubusercontent.com/ccao-data/data-architecture/master/dbt/seeds/ccao/ccao.adjective.csv"
)
right = pd.read_csv(
    "https://raw.githubusercontent.com/ccao-data/data-architecture/master/dbt/seeds/ccao/ccao.person.csv"
)

adj_name_combo = (
    np.random.choice(left["adjective"]) + "-" + np.random.choice(right["person"])
)
timestamp = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime(
    "%Y-%m-%d_%H:%M"
)
run_id = timestamp + "-" + adj_name_combo

# Control flow for incorporating versioning
dynamic_assignment = {
    "run_id": run_id,
    "rolling_window": lambda df: pd.to_datetime(
        df["rolling_window"], format="%Y%m"
    ).dt.date,
}

if not inputs["manual_update"]:
    dynamic_assignment["version"] = 1

# Finalize to write to sale.flag table
df_to_write = df_to_finalize[cols_to_write].assign(**dynamic_assignment).reset_index(drop=True)

if inputs["manual_update"] == True:
    # Merge, compute new version, and drop unnecessary columns
    df_to_write = (
        df_to_write.merge(existing_max_version, on="meta_sale_document_num", how="left")
        .assign(
            version=lambda x: x["existing_version"]
            .apply(lambda y: y + 1 if pd.notnull(y) else 1)
            .astype(int)
        )
        .drop(columns=["existing_version"])
    )

run_filter = str(
    {"housing_market_type": inputs["housing_market_type"], "run_tri": inputs["run_tri"]}
)

# Get sale.parameter data
df_parameters = pd.DataFrame(
    {
        "run_id": [run_id],
        "sales_flagged": [df_to_write.shape[0]],
        "earliest_data_ingest": [earliest_sale_ingest],
        "latest_data_ingest": [latest_sale_ingest],
        "run_filter": [run_filter],
        "iso_forest_cols": [inputs["iso_forest"]],
        "stat_groups": [inputs["stat_groups"]],
        "sales_to_write_filter": [inputs["sales_to_write_filter"]],
        "housing_market_class_codes": [inputs["housing_market_class_codes"]],
        "dev_bounds": [inputs["dev_bounds"]],
        "ptax_sd": [inputs["ptax_sd"]],
        "rolling_window": [inputs["rolling_window_months"]],
        "time_frame": [inputs["time_frame"]],
        "short_term_owner_threshold": [flg_model.SHORT_TERM_OWNER_THRESHOLD],
        "min_group_thresh": [inputs["min_groups_threshold"]],
    }
)

# -------------------------------------------
# Standardize dtypes to prevent Athena errors
# -------------------------------------------

# TODO: Replace this with modify_types code
df_parameter = flg.modify_dtypes(df_parameter)

# Get sale.group_mean data
df_group_means = []  # List to store the transformed DataFrames
for df_name, df_info in dfs_to_finalize.items():
    # Calculate group sizes
    group_df = dfs_flagged[df_name]["df"].copy()
    group_sizes = group_df.groupby(df_info["columns"]).size().reset_index(name="group_size")
    group_df = group_df.merge(group_sizes, on=df_info["columns"], how="left")

    group_df["group"] = group_df.apply(
        lambda row: "_".join([str(row[col]) for col in df_info["columns"]]), axis=1
    )

    if df_info["condos_boolean"]:
        group_df = group_df.drop_duplicates(subset=["group"])[
            ["group", "group_mean", "group_std", "group_size"]
        ].assign(run_id=run_id)
    else:
        group_df = group_df.drop_duplicates(subset=["group"])[
            [
                "group",
                "group_mean",
                "group_std",
                "group_sqft_std",
                "group_sqft_mean",
                "group_size",
            ]
        ].assign(run_id=run_id)

    market_value = df_info["market"]
    df_group_mean["group"] = df_group_mean["group"].astype(str) + "-" + market_value
    df_group_means.append(df_group_mean)

df_group_mean_to_write = pd.concat(df_group_means, ignore_index=True)

# Get sale.metadata table
commit_sha = sp.getoutput("git rev-parse HEAD")
df_metadata = pd.DataFrame(
    {
        "run_id": [run_id],
        "long_commit_sha": commit_sha,
        "short_commit_sha": commit_sha[0:8],
        "run_timestamp": timestamp,
        "run_type": "initial_flagging" if not inputs["manual_update"] else "manual_update",
        "run_note": inputs["run_note"],
    }
)


######################
# UPLOAD
######################


# - - - -
# Write tables
# - - - -

tables_to_write = {
    "flag": df_to_write,
    "parameter": df_parameter,
    "group_mean": df_group_mean_to_write,
    "metadata": df_metadata,
}

file_name = run_id + ".parquet"
for table, df in tables_to_write.items():
    s3_file_path = os.path.join(
        os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
        "sale",
        table,
        file_name
    )
    wr.s3.to_parquet(df=df, path=s3_file_path)
    print(f"{table} table successfully written")
