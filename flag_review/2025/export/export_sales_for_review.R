library(DBI)
library(ccao)
library(readxl)
library(noctua)
library(arrow)
library(dplyr)
library(purrr)
library(writexl)
library(openxlsx)

noctua_options(cache_size = 10, unload = TRUE)

# Establish connection
AWS_ATHENA_CONN_NOCTUA <- dbConnect(
  noctua::athena(),
  rstudio_conn_tab = FALSE
)

# Get flag data
df_flags <- dbGetQuery(
  conn = AWS_ATHENA_CONN_NOCTUA,
  "
  SELECT
    flag.*,
    vw_ps.pin,
    vw_ps.year,
    vw_ps.class,
    vw_ps.township_code,
    vw_ps.nbhd,
    vw_ps.sale_price,
    vw_pa.prop_address_full || ', ' || vw_pa.prop_address_city_name AS address,
    EXISTS (
        SELECT 1
        FROM default.vw_pin_permit vw_pp
        WHERE vw_pp.pin = vw_ps.pin
          AND vw_pp.assessable = 'A'
          AND CAST(vw_pp.year AS integer)
                BETWEEN CAST(vw_ps.year AS integer) - 2
                    AND CAST(vw_ps.year AS integer)
    ) AS has_recent_assessable_permit
  FROM default.vw_pin_sale AS vw_ps
  INNER JOIN sale.flag AS flag
    ON vw_ps.doc_no = flag.meta_sale_document_num
    AND vw_ps.sv_run_id = flag.run_id
  LEFT JOIN default.vw_pin_universe vw_pu
    ON vw_ps.year = vw_pu.year
    AND vw_ps.pin = vw_pu.pin
  LEFT JOIN default.vw_pin_address vw_pa
    ON vw_ps.year = vw_pa.year
    AND vw_ps.pin = vw_pa.pin
  WHERE vw_pu.triad_name = 'South'
    AND vw_ps.year = '2025'
  "
)


df_flags_res <-
  df_flags %>% filter(class != "299")

df_flags_condos <-
  df_flags %>% filter(class == "299")

# - - - - - - - - - - -
# Part 1: Acquire sales that we can flag for valuations to look at for
# characteristic corrections
# - - - - - - - - - - -

top_residential <- df_flags_res %>%
  group_by(township_code) %>%
  {
    top_by_raw_price <- slice_max(., order_by = abs(sv_price_deviation), n = 30, with_ties = FALSE)
    top_by_sqft <- slice_max(., order_by = abs(sv_price_per_sqft_deviation), n = 30, with_ties = FALSE)
    bind_rows(top_by_raw_price, top_by_sqft)
  } %>%
  arrange(desc(pmax(abs(sv_price_deviation), abs(sv_price_per_sqft_deviation)))) %>%
  # De-duplicate overlap between raw price and sqft selections
  distinct(meta_sale_document_num, .keep_all = TRUE) %>%
  slice_head(n = 30) %>%
  ungroup() %>%
  filter(
    abs(sv_price_deviation) >= 1.3 |
      abs(sv_price_per_sqft_deviation >= 1.3)
  ) %>%
  # Re-sort rows across the entire dataframe now that we're done grouping
  arrange(desc(pmax(abs(sv_price_deviation), abs(sv_price_per_sqft_deviation))))

## --- Top 10 per group for CONDOS ---
top_condos <- df_flags_condos %>%
  group_by(township_code) %>%
  slice_max(order_by = abs(sv_price_deviation), n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(desc(abs(sv_price_deviation))) %>%
  select(-sv_price_per_sqft_deviation) %>%
  filter(abs(sv_price_deviation) >= 1.5)

# - - - - - - - - - - - -
# Part 2: Select a sample of data for human analysts to label data
# - - - - - - - - - - - -

df_model_evaluation_res <- df_flags_res %>%
  filter(
    (
      between(abs(sv_price_deviation), 1.8, 2.2) |
        between(abs(sv_price_per_sqft_deviation), 1.8, 2.2)
    ) |
      (
        ptax_flag_original &
          (
            between(abs(sv_price_deviation), 1.4, 1.6) |
              between(abs(sv_price_per_sqft_deviation), 1.4, 1.6)
          )
      )
  )

df_model_evaluation_condos <- df_flags_condos %>% filter(
  (
    between(sv_price_deviation, 2.8, 3.5) |
      between(sv_price_deviation, -2.5, -1.8)
  )
)

df_model_evaluation <- rbind(
  df_model_evaluation_res, df_model_evaluation_condos
)

# Gets the data ready for human-readable excel output
format_sales <- function(df) {
  df %>%
    mutate(`Township Name` = ccao::town_convert(township_code)) %>%
    rename(
      PIN = pin,
      `Sale Doc No` = meta_sale_document_num,
      Address = address,
      `Sale Year` = year,
      Class = class,
      `Township Code` = township_code,
      `Nbhd code` = nbhd,
      `Sale Price` = meta_sale_price_original,
      `Has Recent Assessable Permit` = has_recent_assessable_permit,
      `Flagged as Outlier sale?` = sv_is_outlier,
      `Flagged as Outlier Based on PTAX-203 Line Items?` = sv_is_ptax_outlier,
      `Outlier Reason 1` = sv_outlier_reason1,
      `Outlier Reason 2` = sv_outlier_reason2,
      `Outlier Reason 3` = sv_outlier_reason3
    ) %>%
    select(
      PIN, `Sale Doc No`, Address, `Sale Year`, Class,
      `Township Code`, `Township Name`, `Nbhd code`,
      `Sale Price`,
      `Has Recent Assessable Permit`,
      `Flagged as Outlier sale?`,
      `Flagged as Outlier Based on PTAX-203 Line Items?`,
      `Outlier Reason 1`, `Outlier Reason 2`, `Outlier Reason 3`,
      starts_with("sv_price_"),
      group,
      run_id,
      version
    ) %>%
    mutate(
      `Sale is Arm’s Length` = "",
      `Characteristic Change` = "",
      `Date of Characteristic Change` = "",
      `Class Change` = "",
      Notes = ""
    )
}


cleaned_top_res <- format_sales(top_residential)
cleaned_top_condos <- format_sales(top_condos)
cleaned_model_evaluation <- format_sales(df_model_evaluation)

# Double check we aren't duplicating work with
# res
activity_center_tasks <- read_excel(
  path = "251201_ActivityCenter_CurrentTasks.xlsx",
  sheet = 3
)

activity_center_tasks_pins_to_check <- activity_center_tasks %>%
  filter(
    !(EVENT_CATEGORY_NAME %in% c("Case Appeals", "Appeal")),
    format(ASSIGNED_DATE, "%Y") == "2025",
  )
# We basically see zero overlap here. 1 from res and 0 from condo.
act_ids <- activity_center_tasks_pins_to_check$SUBJECT_ID

cleaned_top_res_dedupe <- cleaned_top_res %>%
  filter(!(PIN %in% act_ids))

cleaned_top_condos_dedupe <- cleaned_top_condos %>%
  filter(!(PIN %in% act_ids))

# Write output
write_xlsx(
  cleaned_top_res_dedupe,
  path = "res_characteristic_potentials.xlsx"
)

write_xlsx(
  cleaned_top_condos_dedupe,
  path = "condos_characteristic_potentials.xlsx"
)

write_xlsx(
  cleaned_model_evaluation,
  path = "model_evaluation_sales.xlsx"
)

# - - -
# 12/15/25 update:
# Second export for valuations. We are going to output sales that weren't
# included in the first export, constructed prior to this comment

# We make largely the same selections as above, but we
# - widen the net to get more sales
# - exclude the sales from the output that we have already supplied to valuations
# - no longer group by township (to get a better signal on true problematic
#   sales since we've already sent over the cream of the crop)
# - - -

top_residential_updated <- df_flags_res %>%
  {
    top_by_raw_price <- slice_max(
      .,
      order_by = abs(sv_price_deviation),
      n = 500,
      with_ties = FALSE
    )

    top_by_sqft <- slice_max(
      .,
      order_by = abs(sv_price_per_sqft_deviation),
      n = 500,
      with_ties = FALSE
    )

    bind_rows(top_by_raw_price, top_by_sqft)
  } %>%
  distinct(meta_sale_document_num, .keep_all = TRUE) %>%
  filter(
    abs(sv_price_deviation) >= 1.5 |
      abs(sv_price_per_sqft_deviation) >= 1.5
  ) %>%
  arrange(desc(pmax(
    abs(sv_price_deviation),
    abs(sv_price_per_sqft_deviation)
  )))


top_condos_updated <- df_flags_condos %>%
  {
    top_by_raw_price <- slice_max(
      .,
      order_by = abs(sv_price_deviation),
      n = 500,
      with_ties = FALSE
    )
  } %>%
  distinct(meta_sale_document_num, .keep_all = TRUE) %>%
  filter(
    abs(sv_price_deviation) >= 1.5
  ) %>%
  arrange(desc(
    abs(sv_price_deviation)
  ))

# Remove any overlap from prior export
top_residential_updated_filtered <-
  top_residential_updated %>%
  filter(
    !(meta_sale_document_num %in% top_residential$meta_sale_document_num)
  )

top_condos_updated_filtered <- top_condos_updated %>%
  filter(
    !(meta_sale_document_num %in% top_condos$meta_sale_document_num),
  )

cleaned_top_res_second_round <- format_sales(top_residential_updated_filtered)
cleaned_top_condos_second_round <- format_sales(top_condos_updated_filtered)

write_xlsx(
  cleaned_top_res_second_round,
  path = "res_characteristic_potentials_second_round.xlsx"
)

write_xlsx(
  cleaned_top_condos_second_round,
  path = "condos_characteristic_potentials_second_round.xlsx"
)


# Update 12/17/2025
#
# Valuations can review more sales, so we are going to output 3 more files.
#
# - The rest of south tri 2025 sales
# - 2024 south tri sales
# - 2025 city tri sales with high outlier significance
#

# jscpd:ignore-start
# Re-do query to grab 2024 and city tri sales
df_flags_larger_scope <- dbGetQuery(
  conn = AWS_ATHENA_CONN_NOCTUA,
  "
  SELECT
    flag.*,
    vw_ps.pin,
    vw_ps.year,
    vw_ps.class,
    vw_ps.township_code,
    vw_pu.triad_name,
    vw_ps.nbhd,
    vw_ps.sale_price,
    vw_pa.prop_address_full || ', ' || vw_pa.prop_address_city_name AS address,
    EXISTS (
        SELECT 1
        FROM default.vw_pin_permit vw_pp
        WHERE vw_pp.pin = vw_ps.pin
          AND vw_pp.assessable = 'A'
          AND CAST(vw_pp.year AS integer)
                BETWEEN CAST(vw_ps.year AS integer) - 2
                    AND CAST(vw_ps.year AS integer)
    ) AS has_recent_assessable_permit
  FROM default.vw_pin_sale AS vw_ps
  INNER JOIN sale.flag AS flag
    ON vw_ps.doc_no = flag.meta_sale_document_num
    AND vw_ps.sv_run_id = flag.run_id
  LEFT JOIN default.vw_pin_universe vw_pu
    ON vw_ps.year = vw_pu.year
    AND vw_ps.pin = vw_pu.pin
  LEFT JOIN default.vw_pin_address vw_pa
    ON vw_ps.year = vw_pa.year
    AND vw_ps.pin = vw_pa.pin
  WHERE vw_pu.triad_name IN ('South', 'City')
    AND vw_ps.year IN ('2024', '2025')
  "
)


# Make some tweaks to this function to standardize with valuations' workflow
format_sales <- function(df) {
  df %>%
    mutate(`Township Name` = ccao::town_convert(township_code)) %>%
    rename(
      PIN = pin,
      `Sale Doc No` = meta_sale_document_num,
      Address = address,
      `Sale Year` = year,
      Class = class,
      `Township Code` = township_code,
      `Nbhd code` = nbhd,
      `Sale Price` = meta_sale_price_original,
      `Has Recent Assessable Permit` = has_recent_assessable_permit,
      `Flagged as Outlier sale?` = sv_is_outlier,
      `Flagged as Outlier Based on PTAX-203 Line Items?` = sv_is_ptax_outlier,
      `Outlier Reason 1` = sv_outlier_reason1,
      `Outlier Reason 2` = sv_outlier_reason2,
      `Outlier Reason 3` = sv_outlier_reason3
    ) %>%
    select(
      PIN, `Sale Doc No`, Address, `Sale Year`, Class,
      `Township Code`, `Township Name`, `Nbhd code`,
      `Sale Price`,
      `Has Recent Assessable Permit`,
      `Flagged as Outlier sale?`,
      `Flagged as Outlier Based on PTAX-203 Line Items?`,
      `Outlier Reason 1`, `Outlier Reason 2`, `Outlier Reason 3`,
      starts_with("sv_price_"),
      group,
      run_id,
      version
    ) %>%
    mutate(
      `Sale is Arm’s Length` = "",
      `Characteristic Change` = "",
      Notes = "",
      `Class Change` = "",
      FLIPS = "",
      `FIELD CHECK` = "",
      `WORKDRAWER FOR POSSIBLE REHAB` = ""
    )
}
# jscpd:ignore-end

# Remove doc_nos already exported in the first and second exports
exported_doc_nos <- c(
  cleaned_top_res_dedupe$`Sale Doc No`,
  cleaned_top_condos_dedupe$`Sale Doc No`,
  cleaned_top_res_second_round$`Sale Doc No`,
  cleaned_top_condos_second_round$`Sale Doc No`
) %>% unique()

# Helper: unified sort function (condos only have raw deviation)
# This puts the highest price deviation sales near the top of the workbooks
sort_data <- function(df) {
  has_sqft <- "sv_price_per_sqft_deviation" %in% names(df)

  if (!has_sqft) {
    df %>%
      mutate(sort_key = abs(sv_price_deviation)) %>%
      arrange(desc(sort_key))
  } else {
    df %>%
      mutate(
        sort_key = if_else(
          Class == "299" | is.na(sv_price_per_sqft_deviation),
          abs(sv_price_deviation),
          pmax(abs(sv_price_deviation), abs(sv_price_per_sqft_deviation), na.rm = TRUE)
        )
      ) %>%
      arrange(desc(sort_key)) %>%
      select(-sort_key)
  }
}

# Rest of south triad 2025 sales, excluding doc_nos already exported
south_2025_rest <- df_flags_larger_scope %>%
  filter(triad_name == "South", year == "2025") %>%
  filter(!(meta_sale_document_num %in% exported_doc_nos))

cleaned_south_2025_rest <- format_sales(south_2025_rest) %>%
  sort_data()

# 2) 2024 south triad sales (all)
south_2024_all <- df_flags_larger_scope %>%
  filter(triad_name == "South", year == "2024")

cleaned_south_2024_all <- format_sales(south_2024_all) %>%
  sort_data()

# 2025 city triad sales with "same sd + township-bound strategy" as initial
# - Res: top 300 per township by raw deviation AND top 30 by $/sqft,
#   de-dupe, threshold >= 1.3 on either metric
# - Condos (class 299): top 200 per township by raw deviation, threshold >= 1.5
df_city_2025 <- df_flags_larger_scope %>%
  filter(triad_name == "City", year == "2025")

df_city_2025_res <- df_city_2025 %>% filter(class != "299")
df_city_2025_condos <- df_city_2025 %>% filter(class == "299")

city_top_res <- df_city_2025_res %>%
  group_by(township_code) %>%
  {
    top_by_raw_price <- slice_max(., order_by = abs(sv_price_deviation), n = 300, with_ties = FALSE)
    top_by_sqft <- slice_max(., order_by = abs(sv_price_per_sqft_deviation), n = 300, with_ties = FALSE)
    bind_rows(top_by_raw_price, top_by_sqft)
  } %>%
  ungroup() %>%
  distinct(meta_sale_document_num, .keep_all = TRUE) %>%
  mutate(sort_key = pmax(abs(sv_price_deviation), abs(sv_price_per_sqft_deviation), na.rm = TRUE)) %>%
  filter(
    abs(sv_price_deviation) >= 1.3 |
      abs(sv_price_per_sqft_deviation) >= 1.3
  )

city_top_condos <- df_city_2025_condos %>%
  group_by(township_code) %>%
  slice_max(order_by = abs(sv_price_deviation), n = 200, with_ties = FALSE) %>%
  ungroup() %>%
  filter(abs(sv_price_deviation) >= 1.5) %>%
  mutate(sort_key = abs(sv_price_deviation))

city_2025_high_sig <- bind_rows(city_top_res, city_top_condos)

cleaned_city_2025_high_sig <- format_sales(city_2025_high_sig) %>%
  sort_data()

add_ccrd_link_to_pin <- function(df, path) {
  df_with_link <- df %>%
    mutate(`CCRD Link` = "") %>%
    relocate(`CCRD Link`, .after = Address)

  link_col_idx <- which(names(df_with_link) == "CCRD Link")

  pins <- as.character(df_with_link$PIN)
  urls <- paste0("https://crs.cookcountyclerkil.gov/Search/ResultByPin?id1=", pins)
  formulas <- paste0('HYPERLINK("', urls, '","CCRD Link")')

  wb <- createWorkbook()
  addWorksheet(wb, "Sheet1")
  writeData(wb, "Sheet1", df_with_link, withFilter = TRUE)

  writeFormula(wb, "Sheet1", formulas, startCol = link_col_idx, startRow = 2)
  link_style <- createStyle(fontColour = "#0563C1", textDecoration = "underline")
  addStyle(wb, "Sheet1", link_style, rows = 2:(nrow(df_with_link) + 1), cols = link_col_idx, gridExpand = TRUE)

  saveWorkbook(wb, path, overwrite = TRUE)
}

# Write output
add_ccrd_link_to_pin(cleaned_south_2025_rest, "south_2025_rest_sales_third_round.xlsx")
add_ccrd_link_to_pin(cleaned_south_2024_all, "south_2024_sales_third_round.xlsx")
add_ccrd_link_to_pin(cleaned_city_2025_high_sig, "city_2025_high_outlier_significance_third_round.xlsx")
