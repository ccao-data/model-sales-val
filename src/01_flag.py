import copy
import os
import subprocess as sp

import pandas as pd
from pyathena import connect
from pyathena.pandas.util import as_pandas

import constants
import model
import utils

root = sp.getoutput("git rev-parse --show-toplevel")

# Validate the input specification
if constants.INPUTS["output_environment"] not in {"dev", "prod"}:
    raise ValueError("output_environment must be either 'dev' or 'prod'")

# Check housing_market_type
# TODO: Add res_all and other res_type exclusivity check
assert "housing_market_type" in constants.INPUTS, (
    "Missing key: 'housing_market_type'"
)
assert set(constants.INPUTS["housing_market_type"]).issubset(
    {"res_single_family", "res_multi_family", "condos", "res_all"}
), (
    "housing_market_type can only contain 'res_single_family', 'res_multi_family', 'condos', 'res_all'"
)
assert len(constants.INPUTS["housing_market_type"]) == len(
    set(constants.INPUTS["housing_market_type"])
), "Duplicate values in 'housing_market_type'"

# Check run_tri
assert "run_tri" in constants.INPUTS, "Missing key: 'run_tri'"
assert set(constants.INPUTS["run_tri"]).issubset({1, 2, 3}), (
    "run_tri can only contain 1, 2, 3"
)
assert len(constants.INPUTS["run_tri"]) == len(
    set(constants.INPUTS["run_tri"])
), "Duplicate values in 'run_tri'"

# Ingest
df = pd.read_parquet(os.path.join(root, "input", "sales_ingest.parquet"))
df_ingest = pd.read_parquet(
    os.path.join(root, "input", "sales_ingest.parquet")
)

if constants.INPUTS["manual_update"] is True:
    # TODO: grab maxes from this query to avoid large data ingest
    SQL_QUERY_SALES_VAL = """
    SELECT *
    FROM sale.flag
    """
    # Connect to Athena
    conn = connect(
        s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
        region_name=os.getenv("AWS_REGION"),
    )
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY_SALES_VAL)
    df_ingest_flag = as_pandas(cursor)
    df_flag_table = df_ingest_flag


def create_bins_and_labels(input_list):
    """
    Some of the groups used for flagging are partitions of
    building size or age, this helper function returns the
    bins and labels for the column creation based on input data
    from our config file.
    """

    # Initialize bins with 0 and float("inf")
    bins = [0] + input_list + [float("inf")]

    # Generate labels based on bins
    labels = []
    for i in range(len(bins) - 1):
        if i == 0:
            labels.append(f"below-{bins[i + 1]}")
        elif i == len(bins) - 2:
            labels.append(f"above-{bins[i]}")
        else:
            labels.append(f"{bins[i]}-to-{bins[i + 1]}")

    return bins, labels


# - - - - - - -
# Make correct filters and set up dictionary structure
# Split tris into groups according to their flagging methods
# - - - - - - -

dfs_to_rolling_window = {}  # Dictionary to store DataFrames

for tri in constants.INPUTS["run_tri"]:
    # Iterate over housing types defined in yaml
    for housing_type in constants.INPUTS["housing_market_type"]:
        if housing_type not in constants.INPUTS["stat_groups"][f"tri{tri}"]:
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
            constants.INPUTS["housing_market_class_codes"][housing_type]
        )

        # Initialize the DataFrame for the current key
        df_filtered = df[triad_code_filter & market_filter].copy()
        dfs_to_rolling_window[key] = {
            "df": df_filtered  # Store the filtered DataFrame
        }

        # Extract the specific housing type configuration
        housing_type_config = constants.INPUTS["stat_groups"][f"tri{tri}"][
            housing_type
        ]

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
                    bins, labels = create_bins_and_labels(col["bins"])
                    new_col_name = f"{col['column']}_bin"
                    df_filtered[new_col_name] = pd.cut(
                        df_filtered[col["column"]], bins=bins, labels=labels
                    )
                    transformed_columns.append(new_col_name)
            else:
                transformed_columns.append(col)

        dfs_to_rolling_window[key]["columns"] = transformed_columns

        # Add rest of config information
        dfs_to_rolling_window[key]["iso_forest_cols"] = constants.INPUTS[
            "iso_forest"
        ]["res" if "res" in housing_type else "condos"]
        dfs_to_rolling_window[key]["condos_boolean"] = housing_type == "condos"
        dfs_to_rolling_window[key]["market"] = housing_type


# - - - - - -
# Make rolling window
# - - - - - -

dfs_to_flag = copy.deepcopy(dfs_to_rolling_window)

for df_name, df_info in dfs_to_rolling_window.items():
    print(f"Assigning rolling window for {df_name}")
    df_copy = df_info["df"].copy()

    df_copy = utils.add_rolling_window(
        df_copy, num_months=constants.INPUTS["rolling_window_months"]
    )
    dfs_to_flag[df_name]["df"] = df_copy

# - - - - -
# Flag Sales
# - - - - -

dfs_flagged = copy.deepcopy(dfs_to_flag)

for df_name, df_info in dfs_to_flag.items():
    print(f"\nFlagging sales for {df_name}")
    df_copy = df_info["df"].copy()
    df_copy = model.go(
        df=df_copy,
        groups=tuple(df_info["columns"]),
        iso_forest_cols=df_info["iso_forest_cols"],
        dev_bounds=tuple(constants.INPUTS["dev_bounds"]),
        condos=df_info["condos_boolean"],
        raw_price_threshold=constants.INPUTS["raw_price_threshold"],
    )

    # Add the edited or unedited dataframe to the new dictionary
    dfs_flagged[df_name]["df"] = df_copy

# - - - - - - - - - - -
# Adjust outliers based on group sizes and incorporate ptax information
# - - - - - - - - - - -

dfs_to_finalize = copy.deepcopy(dfs_flagged)

for df_name, df_info in dfs_flagged.items():
    # Make a copy of the data frame to edit
    print(f"\n Enacting group threshold and creating ptax data for {df_name}")
    df_copy = df_info["df"].copy()

    df_copy = utils.ptax_adjustment(
        df=df_copy,
        groups=df_info["columns"],
        ptax_sd=constants.INPUTS["ptax_sd"],
        condos=df_info["condos_boolean"],
    )

    df_copy = utils.classify_outliers(
        df=df_copy,
        stat_groups=df_info["columns"],
        min_threshold=constants.INPUTS["min_groups_threshold"],
    )

    """
    Modify the 'group' column by appending '-market_value', this is done
    to make sure that a two different groups with the same run_id won't
    be returned with the same value. For example, if res and condos have the 
    same column groupings, joining the group column from sale.flag to sale.group_mean
    by 'group' and 'run_id' could potentially return two groups. This market type
    append fixes that. This is also added in the group_mean data.
    """
    market_value = df_info["market"]
    df_copy["group"] = df_copy["group"].astype(str) + "-" + market_value

    # Add the edited or unedited dataframe to the new dictionary
    dfs_to_finalize[df_name]["df"] = df_copy

# - - - - - - -
# Finalize data to write and create data for all metadata tables
# - - - - - - - -

if constants.INPUTS["manual_update"] is True:
    # Group the existing data by 'ID' and find the maximum 'version' for each sale
    existing_max_version = (
        df_flag_table.groupby("meta_sale_document_num")["version"]
        .max()
        .reset_index()
        .rename(columns={"version": "existing_version"})
    )


dfs_to_finalize_list = [details["df"] for details in dfs_to_finalize.values()]
df_to_finalize = pd.concat(dfs_to_finalize_list, axis=0)

df_to_write, run_id, timestamp = utils.finish_flags(
    df=df_to_finalize,
    start_date=constants.INPUTS["time_frame"]["start"],
    manual_update=constants.INPUTS["manual_update"],
    sales_to_write_filter=constants.INPUTS["sales_to_write_filter"],
)

if constants.INPUTS["manual_update"] is True:
    # Merge, compute new version, and drop unnecessary columns
    df_to_write = (
        df_to_write.merge(
            existing_max_version, on="meta_sale_document_num", how="left"
        )
        .assign(
            version=lambda x: x["existing_version"]
            .apply(lambda y: y + 1 if pd.notnull(y) else 1)
            .astype(int)
        )
        .drop(columns=["existing_version"])
    )
    # Additional filtering if manual_update_only_new_sales is True
    # If this is set to true, only unseen sales will get flag updates
    if constants.INPUTS["manual_update_only_new_sales"] is True:
        df_to_write = df_to_write[df_to_write["version"] == 1]

run_filter = str(
    {
        "housing_market_type": constants.INPUTS["housing_market_type"],
        "run_tri": constants.INPUTS["run_tri"],
    }
)

# Get parameters df
df_parameter = utils.get_parameter_df(
    df_to_write=df_to_write,
    df_ingest=df_ingest,
    run_filter=run_filter,
    iso_forest_cols=constants.INPUTS["iso_forest"],
    stat_groups=constants.INPUTS["stat_groups"],
    sales_to_write_filter=constants.INPUTS["sales_to_write_filter"],
    housing_market_class_codes=constants.INPUTS["housing_market_class_codes"],
    dev_bounds=constants.INPUTS["dev_bounds"],
    ptax_sd=constants.INPUTS["ptax_sd"],
    rolling_window=constants.INPUTS["rolling_window_months"],
    time_frame=constants.INPUTS["time_frame"],
    short_term_threshold=model.SHORT_TERM_OWNER_THRESHOLD,
    min_group_threshold=constants.INPUTS["min_groups_threshold"],
    raw_price_threshold=constants.INPUTS["raw_price_threshold"],
    run_id=run_id,
)

# Standardize dtypes to prevent Athena errors
df_parameter = utils.modify_dtypes(df_parameter)

# Get sale.group_mean data
df_group_means = []  # List to store the transformed DataFrames

for df_name, df_info in dfs_to_finalize.items():
    df_group_mean = utils.get_group_mean_df(
        df=dfs_flagged[df_name]["df"],
        stat_groups=df_info["columns"],
        run_id=run_id,
        condos=df_info["condos_boolean"],
    )
    market_value = df_info["market"]
    df_group_mean["group"] = (
        df_group_mean["group"].astype(str) + "-" + market_value
    )
    df_group_means.append(df_group_mean)

df_group_mean_to_write = pd.concat(df_group_means, ignore_index=True)

# Get sale.metadata table
commit_sha = sp.getoutput("git rev-parse HEAD")

run_type = (
    "initial_flagging"
    if not constants.INPUTS["manual_update"]
    else "manual_update_only_new_sales"
    if constants.INPUTS["manual_update_only_new_sales"]
    else "manual_update"
)

df_metadata = utils.get_metadata_df(
    run_id=run_id,
    timestamp=timestamp,
    run_type=run_type,
    commit_sha=commit_sha,
    run_note=constants.INPUTS["run_note"],
)

# - - - -
# Output tables locally
# - - - -

tables_to_write = {
    "flag.parquet": df_to_write,
    "parameter.parquet": df_parameter,
    "group_mean.parquet": df_group_mean_to_write,
    "metadata.parquet": df_metadata,
}

for filename, df in tables_to_write.items():
    output_path = os.path.join(os.path.join(root, "output"), filename)
    df.to_parquet(output_path, index=False)
    print(f"Saved {filename}")
