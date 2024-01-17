from glue.flagging_script_glue import flagging as flg_model
from glue import sales_val_flagging as flg
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

# Set working dir to manual_update, standardize yaml and src locations
root = sp.getoutput("git rev-parse --show-toplevel")
os.chdir(os.path.join(root, "manual_flagging"))

# Use yaml as inputs
with open(os.path.join("yaml", "inputs_initial.yaml"), "r") as stream:
    inputs = yaml.safe_load(stream)

# Connect to Athena
conn = connect(
    s3_staging_dir=os.getenv("AWS_ATHENA_S3_STAGING_DIR"),
    region_name=os.getenv("AWS_REGION"),
)

date_floor = flg.months_back(
    date_str=inputs["time_frame"]["start"],
    num_months=inputs["rolling_window_months"] - 1,
)

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
        res.char_yrblt as yrblt,
        res.pin AS pin,
        res.char_bldg_sf AS char_bldg_sf,
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
        NULL AS yrblt,
        condo.pin AS pin,
        NULL AS char_bldg_sf,
        FALSE AS pin_is_multicard
    FROM default.vw_pin_condo_char condo
    WHERE condo.class IN ('297', '299', '399')
    AND NOT condo.is_parking_space
    AND NOT condo.is_common_area
)

-- Now, join with sale table and filters
SELECT
    sale.sale_price AS meta_sale_price,
    sale.sale_date AS meta_sale_date,
    sale.doc_no AS meta_sale_document_num,
    sale.seller_name AS meta_sale_seller_name,
    sale.buyer_name AS meta_sale_buyer_name,
    sale.nbhd as nbhd,
    sale.sale_filter_ptax_flag AS ptax_flag_original,
    data.class,
    data.township_code,
    data.yrblt,
    data.year,
    data.pin,
    data.char_bldg_sf,
    data.indicator  -- Selecting the indicator column
FROM CombinedData data
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = data.pin
    AND sale.year = data.year
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


# Execute query and return as pandas data frame
cursor = conn.cursor()
cursor.execute(SQL_QUERY)
metadata = cursor.description

df_ingest = as_pandas(cursor)
df = df_ingest

conversion_dict = {
    col[0]: flg.sql_type_to_pd_type(col[1])
    for col in metadata
    if flg.sql_type_to_pd_type(col[1]) is not None
}
df = df.astype(conversion_dict)

df["ptax_flag_original"].fillna(False, inplace=True)
# - - -
# Testing new ingest
# - - -

# Subset to only City Tri data
df = df[df["township_code"].isin(["70", "71", "72", "73", "74", "75", "76", "77"])]

# Pre-process res data
df_res = df[df["indicator"] == "res"].reset_index(drop=True)
# - - -
# Create new geographies and building age features
# - - -
new_nbhd_splits = [
    ["75031", "75032", "75033", "75060"],
    ["75010", "75021", "75022", "75023", "75040"],
    ["70260", "70260", "70260", "70260"],
    ["70210", "70220", "70230", "70120", "70220", "70230", "70210", "70250"],
    [
        "70111",
        "70111",
        "70030",
        "70111",
        "70120",
        "70111",
        "70120",
        "70121",
        "70111",
        "70120",
        "70080",
        "70080",
        "70091",
        "70170",
    ],
    ["70101", "70240", "70241", "70130", "70180", "70280", "70180"],
    ["70080", "70130", "70151", "70100", "70140", "70130", "70151"],
    ["70080", "70083", "70100", "70150"],
    ["70070", "70080", "70120"],
    ["70030", "70030", "70070"],
    ["70030", "70030"],
    ["70020", "70020", "70020"],
    ["70010", "70020"],
    ["70010"],
    [
        "76041",
        "76030",
        "76030",
        "76042",
        "76041",
        "76041",
        "76040",
        "76042",
        "76040",
        "76041",
        "76042",
    ],
    ["76011", "76012", "76011", "76011"],
    ["76030", "76050"],
    ["76060", "76060", "76060"],
    [
        "71021",
        "71021",
        "71463",
        "71461",
        "71410",
        "71440",
        "71022",
        "71461",
        "71560",
        "71010",
        "71060",
        "71090",
        "71101",
        "71150",
        "71402",
        "71430",
        "71010",
        "71071",
        "71580",
        "71090",
        "71101",
        "71150",
        "71171",
        "71022",
        "71030",
        "71050",
    ],
    [
        "71171",
        "71200",
        "71210",
        "71250",
        "71260",
        "71280",
        "71150",
        "71200",
        "71210",
        "71250",
        "71270",
        "71280",
        "71180",
        "71520",
        "71200",
        "71270",
        "71210",
        "71200",
        "71120",
        "71120",
        "71371",
        "71371",
        "71070",
        "71120",
        "71600",
        "71150",
        "71600",
        "71371",
        "71140",
    ],
    [
        "71420",
        "71440",
        "71430",
        "71390",
        "71390",
        "71430",
        "71390",
        "71361",
        "71362",
        "71560",
    ],
    ["71070", "71180", "71070", "71074"],
    ["71110", "71110", "71082"],
    ["71081", "71082", "71082", "71070", "71082", "71101", "71120", "71074"],
    ["71041", "71050", "71042", "71050", "71050", "71042", "71030", "71041"],
    ["74022", "74030", "74022", "74030", "74022", "74030", "74030", "74030"],
    ["74011", "74022"],
    ["74012", "74012", "74014", "74012", "74014", "74012"],
    ["74013", "74013"],
    ["73022", "73011", "73031", "73011"],
    ["73093", "73093", "73063", "73093"],
    ["73050", "73060", "73011", "73031", "73031", "73044", "73060", "73032", "73050"],
    ["73031", "73012", "73050"],
    [
        "73081",
        "73200",
        "73012",
        "73041",
        "73041",
        "73070",
        "73081",
        "73012",
        "73032",
        "73041",
        "73042",
        "73070",
        "73081",
    ],
    [
        "73084",
        "73060",
        "73062",
        "73092",
        "73159",
        "73081",
        "73084",
        "73110",
        "73120",
        "73150",
        "73200",
    ],
    ["77141", "77141", "77141", "77141", "77132", "77141"],
    ["77011", "77013", "77013", "77080", "77011", "77020"],
    ["77102", "77104", "77085", "77080", "77091", "77092", "77102", "77103"],
    ["77060", "77152", "77170"],
    ["77132", "77131", "77132", "77132", "77141", "77132"],
    ["77115", "77091"],
    ["77030", "77150", "77151"],
    ["77052", "77101", "77103", "77120", "77131", "77120", "77051"],
    ["77120", "77120", "77131", "77131", "77120"],
    ["77020", "77030", "77040"],
    ["77101", "77102", "77103", "77101", "77102"],
    ["72052", "72092", "72293", "72092", "72293"],
    ["72051", "72052", "72080", "72090", "72092", "72120"],
    ["72030", "72061", "72350", "72361", "72040", "72070", "72361"],
    ["72070", "72071", "72080", "72110", "72071", "72110", "72071", "72110", "72150"],
    ["72151", "72191", "72200", "72200", "72200", "72151", "72200", "72230"],
    ["72192", "72194", "72191", "72192", "72193", "72194", "72030", "72030", "72350"],
    [
        "72212",
        "72282",
        "72212",
        "72221",
        "72222",
        "72223",
        "72251",
        "72282",
        "72281",
        "72282",
        "72285",
        "72282",
        "72282",
        "72321",
        "72282",
        "72260",
        "72282",
    ],
    ["72091", "72092", "72130", "72170", "72171", "72121", "72171"],
    ["72423", "72423", "72422", "72423", "72422", "72324", "72431", "72432", "72423"],
    ["72274", "72300", "72312", "72323"],
    ["72380", "72350", "72380", "72361", "72350"],
    ["72271", "72274", "72274"],
    ["72310", "72330", "72345", "72321", "72330", "72321", "72330"],
    ["72170", "72260"],
]

# Create a new column with default value
df_res["geography_split"] = None

# Iterate over each list in the list of lists
for i, lst in enumerate(new_nbhd_splits):
    # Create a string by concatenating the elements of the list
    group_name = ",".join(lst)
    # Assign this string to rows where 'nbhd' value is in the current list
    df_res.loc[df_res["nbhd"].isin(lst), "geography_split"] = group_name

# Calculate the building's age
current_year = datetime.datetime.now().year
df_res["bldg_age"] = current_year - df_res["yrblt"]

# - - -
# Single Family
# - - -
single_family_classes = [
    "202",
    "203",
    "204",
    "205",
    "206",
    "207",
    "208",
    "209",
    "210",
    "218",
    "219",
    "234",
    "278",
    "295",
]

df_res_single_fam = df_res[df_res["class"].isin(single_family_classes)]

# Define bins for char_bldg_sf
char_bldg_sf_bins = [0, 1200, 2400, float("inf")]
char_bldg_sf_labels = ["below_1200", "1200_to_2400", "above_2400"]

# Bin the char_bldg_sf data
df_res_single_fam["char_bldg_sf_bin"] = pd.cut(
    df_res_single_fam["char_bldg_sf"],
    bins=char_bldg_sf_bins,
    labels=char_bldg_sf_labels,
)

# Define bins for building age
bldg_age_bins = [0, 40, float("inf")]
bldg_age_labels = ["below_40_years", "above_40_years"]

# Bin the building age data
df_res_single_fam["bldg_age_bin"] = pd.cut(
    df_res_single_fam["bldg_age"], bins=bldg_age_bins, labels=bldg_age_labels
)

# - - - -
# Multi Family
# - - - -

multi_family_classes = ["211", "212"]
df_res_multi_fam = df_res[df_res["class"].isin(multi_family_classes)]

# Define bins for building age
bldg_age_bins = [0, 20, float("inf")]
bldg_age_labels = ["below_20_years", "above_20_years"]

# Bin the building age data
df_res_multi_fam["bldg_age_bin"] = pd.cut(
    df_res_multi_fam["bldg_age"], bins=bldg_age_bins, labels=bldg_age_labels
)

# - - -
# End testing new ingest
# - - -

# Separate res and condo sales based on the indicator column
# df_res = df[df["indicator"] == "res"].reset_index(drop=True)
df_condo = df[df["indicator"] == "condo"].reset_index(drop=True)

# Create condo stat groups. Condos are all collapsed into a single class,
# since there are very few 297s or 399s
condo_stat_groups = inputs["stat_groups"].copy()
condo_stat_groups.remove("class")

# - - -
# Create rolling windows
# - - -

# Rolling window for single family
df_res_single_fam_to_flag = flg.add_rolling_window(
    df_res_single_fam, num_months=inputs["rolling_window_months"]
)

# Rolling window for multi_family
df_res_multi_fam_to_flag = flg.add_rolling_window(
    df_res_multi_fam, num_months=inputs["rolling_window_months"]
)

# - - -
# Check for counts
# - - -
new_groups_single_fam = (
    df_res_single_fam_to_flag.groupby(
        ["geography_split", "char_bldg_sf_bin", "bldg_age_bin", "rolling_window"]
    )
    .agg(
        count=("geography_split", "size"),
        median_sale_price=("meta_sale_price", "median"),
    )
    .reset_index()
)

new_groups_multi_fam = (
    df_res_multi_fam_to_flag.groupby(
        ["geography_split", "bldg_age_bin", "rolling_window"]
    )
    .agg(
        count=("geography_split", "size"),
        median_sale_price=("meta_sale_price", "median"),
    )
    .reset_index()
)


def percentage_over_30(df):
    count_over_30 = df[df["count"] >= 30].shape[0]
    total_count = df.shape[0]
    percentage = (count_over_30 / total_count) * 100
    return percentage


# Check percentage of groups over 30
percentage_over_30(new_groups_single_fam)
percentage_over_30(new_groups_multi_fam)

# Check total number of sales within groups above and below 30
new_groups_single_fam[new_groups_single_fam["count"] < 30]["count"].sum()
new_groups_single_fam[new_groups_single_fam["count"] >= 30]["count"].sum()

# Filtering out non-important rolling window stats
percentage_over_30(
    new_groups_single_fam[
        ~new_groups_single_fam["rolling_window"].astype(str).str.startswith("2013")
    ]
)
percentage_over_30(
    new_groups_multi_fam[
        ~new_groups_multi_fam["rolling_window"].astype(str).str.startswith("2013")
    ]
)

# Saving the DataFrame to an Excel file
new_groups_single_fam[
    ~new_groups_single_fam["rolling_window"].astype(str).str.startswith("2013")
].to_excel("single_fam_v0.xlsx", index=False)
new_groups_multi_fam[
    ~new_groups_multi_fam["rolling_window"].astype(str).str.startswith("2013")
].to_excel("multi_fam_v0.xlsx", index=False)

# - - -
# End Check for counts
# - - -

df_condo_to_flag = flg.add_rolling_window(
    df_condo, num_months=inputs["rolling_window_months"]
)
# - - -
# Separate flagging for both
# - - -
# Flag outliers using the main flagging model
df_res_single_fam_flagged = flg_model.go(
    df=df_res_single_fam_to_flag,
    groups=tuple(inputs["stat_groups"]["single_family"]),
    iso_forest_cols=inputs["iso_forest"],
    dev_bounds=tuple(inputs["dev_bounds"]),
    condos=False,
)

# Flag outliers using the main flagging model
df_res_multi_fam_flagged = flg_model.go(
    df=df_res_multi_fam_to_flag,
    groups=tuple(inputs["stat_groups"]["multi_family"]),
    iso_forest_cols=inputs["iso_forest"],
    dev_bounds=tuple(inputs["dev_bounds"]),
    condos=False,
)
# - - -
# Separate group size for both
# - - -
# Discard any flags with a group size under the threshold
df_res_single_fam_flagged_updated = flg.group_size_adjustment(
    df=df_res_single_fam_flagged,
    stat_groups=inputs["stat_groups"]["single_family"],
    min_threshold=inputs["min_groups_threshold"],
    condos=False,
)

df_res_multi_fam_flagged_updated = flg.group_size_adjustment(
    df=df_res_multi_fam_flagged,
    stat_groups=inputs["stat_groups"]["multi_family"],
    min_threshold=inputs["min_groups_threshold"],
    condos=False,
)

# Flag condo outliers, here we remove price per sqft as an input
# for the isolation forest model since condos don't have a unit sqft
condo_iso_forest = inputs["iso_forest"].copy()
condo_iso_forest.remove("sv_price_per_sqft")

df_condo_flagged = flg_model.go(
    df=df_condo_to_flag,
    groups=tuple(condo_stat_groups),
    iso_forest_cols=condo_iso_forest,
    dev_bounds=tuple(inputs["dev_bounds"]),
    condos=True,
)

df_condo_flagged_updated = flg.group_size_adjustment(
    df=df_condo_flagged,
    stat_groups=condo_stat_groups,
    min_threshold=inputs["min_groups_threshold"],
    condos=True,
)

# Disregard condos for now
# df_flagged_merged = pd.concat(
#    [df_res_flagged_updated, df_condo_flagged_updated]
# ).reset_index(drop=True)

# Iterate through both ptac grousp
# Update the PTAX flag column with an additional std dev conditional
df_res_single_fam_flagged_ptax = flg.ptax_adjustment(
    df=df_res_single_fam_flagged_updated,
    groups=inputs["stat_groups"]["single_family"],
    ptax_sd=inputs["ptax_sd"],
)

df_res_multi_fam_flagged_ptax = flg.ptax_adjustment(
    df=df_res_multi_fam_flagged_updated,
    groups=inputs["stat_groups"]["multi_family"],
    ptax_sd=inputs["ptax_sd"],
)

df_flagged_merged = pd.concat(
    [df_res_single_fam_flagged_ptax, df_res_multi_fam_flagged_ptax]
).reset_index(drop=True)

# Finish flagging and subset to write to flag table
df_to_write, run_id, timestamp = flg.finish_flags(
    df=df_flagged_merged,
    start_date=inputs["time_frame"]["start"],
    manual_update=False,
)

# Write to sale.flag table
flg.write_to_table(
    df=df_to_write,
    table_name="flag",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to sale.parameter table
df_parameters = flg.get_parameter_df(
    df_to_write=df_to_write,
    df_ingest=df_ingest,
    iso_forest_cols=inputs["iso_forest"],
    res_stat_groups=inputs["stat_groups"],
    condo_stat_groups=condo_stat_groups,
    dev_bounds=inputs["dev_bounds"],
    ptax_sd=inputs["ptax_sd"],
    rolling_window=inputs["rolling_window_months"],
    date_floor=inputs["time_frame"]["start"],
    short_term_thresh=flg_model.SHORT_TERM_OWNER_THRESHOLD,
    min_group_thresh=inputs["min_groups_threshold"],
    run_id=run_id,
)

# Standardize dtypes to prevent Athena errors
df_parameters = flg.modify_dtypes(df_parameters)

flg.write_to_table(
    df=df_parameters,
    table_name="parameter",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to sale.group_mean table
df_res_group_mean = flg.get_group_mean_df(
    df=df_res_flagged, stat_groups=inputs["stat_groups"], run_id=run_id, condos=False
)

df_condo_group_mean = flg.get_group_mean_df(
    df=df_condo_flagged, stat_groups=condo_stat_groups, run_id=run_id, condos=True
)

df_group_mean_merged = pd.concat([df_res_group_mean, df_condo_group_mean]).reset_index(
    drop=True
)

flg.write_to_table(
    df=df_group_mean_merged,
    table_name="group_mean",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)

# Write to sale.metadata table
commit_sha = sp.getoutput("git rev-parse HEAD")
df_metadata = flg.get_metadata_df(
    run_id=run_id,
    timestamp=timestamp,
    run_type="initial_flagging",
    commit_sha=commit_sha,
)

flg.write_to_table(
    df=df_metadata,
    table_name="metadata",
    s3_warehouse_bucket_path=os.getenv("AWS_S3_WAREHOUSE_BUCKET"),
    run_id=run_id,
)
