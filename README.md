# Sales validation model

This repository contains code to identify and flag sales that may be non-arms-length transactions. A non-arms-length sale occurs when the buyer and seller have a relationship that might influence the transaction price, leading to a sale that doesn't reflect the true market value of the property. These sales can distort our analyses and models, since they don't adhere to the principle of an open and competitive market.

Table of Contents
================

- [Overview](#overview)
- [Flags at a glance](#flags-at-a-glance)
- [Structure of data](#structure-of-data)
- [Flagging details](#important-flagging-details)
- [AWS Glue job documentation](#aws-glue-job-documentation)
- [Next steps](#next-steps)

# Flags at a glance

Sales from 2014 - present have been processed using our sales validation program. The data we exclude still exist in our data ecosystem, but we don't assign them flags and they aren't used in our groups to calculate group means. For example, we don't include multi-card as the price (multiple buildings per sale) would distort our statistical flagging process. We flag properties from residential and condo data with the following specifications:

### Residential

- Include classes:
  - 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 218, 219, 234, 278, 295
- Exclude:
  - Multi-card sales

### Condos

- Include classes:
  - 297, 299, 399
- Exclude:
  - Parking spaces
  - Building common areas

### Common Exclusions for Both Residential and Condos

- Exclude any sales with a duplicate within 365 days
- Exclude sales less than $10,000
- Exclude multi-pin sales


## Outlier Types

In order to be flagged as on outlier type, the property needs to be a statistical price outlier. A statistical price outlier is a sale price that is some number of standard deviations outside a grouping of similar properties (eg. township, class, time frame).  However, there are also special flags that combine with the statistical outlier type, these sales *should* be more likely to be non-arms length sales than the regular price outlier sales. Examples of these special flags are:
- **PTAX flag**: The PTAX-203 form is required to be filled out for an Illinois Real Estate Transfer Declaration for non-residential property over $1 million and/or as required by Illinois Department of Revenue. If there are certain fields filled out on this form, we mark the sale with a ptax flag. 
- **Non-person sale**: We flag a keyword that suggests the sale involves a legal entity (industrial buyer, bank, real estate firm, construction, etc)
- **Flip Sale**: The owner of the home owned the property for less than 1 year
- **Anomaly**: Our isolation forest statistical outlier algorithm flagged the sale

The following is a list of all flag types:

### High Price

| Indicator               | Criteria                                                  |
|-------------------------|-----------------------------------------------------------|
| PTAX outlier (high)     | PTAX flag & [1 high statistical outlier type]             |
| Home flip sale (high)   | Short-term owner < 1 year & [1 high statistical outlier type] |
| Family sale (high)      | Last name match & [1 high statistical outlier type]       |
| Non-person sale (high)  | Legal / corporate entity + [1 high statistical outlier type] |
| Anomaly (High)          | Anomaly algorithm (high) & [1 high statistical outlier type] |
| High price (raw & sqft) | High price & high price per sq. ft.                       |
| High price swing        | Large swing away from mean & high price outlier           |
| High price (raw)        | High price                                                |
| High price (per sqft)   | High price per sq. ft.                                     |

### Low Price

| Indicator               | Criteria                                                  |
|-------------------------|-----------------------------------------------------------|
| PTAX outlier (low)      | PTAX flag & [1 low statistical outlier type]              |
| Home flip sale (low)    | Short-term owner < 1 year & [1 low statistical outlier type] |
| Family sale (low)       | Last name match & [1 low statistical outlier type]        |
| Non-person sale (low)   | Legal / corporate entity + [1 low statistical outlier type] |
| Anomaly                 | Anomaly algorithm (low) & [1 low statistical outlier type] |
| Low price (raw & sqft)  | Low price & low price per sq. ft.                         |
| Low price swing         | Large swing away from mean & low price outlier            |
| Low price (raw)         | Low price (or under $10k)                                 |
| Low price (per sqft)    | Low price per sq. ft.                                     |


## Distribution of Outlier Types

<!--
/*
This query is used to generate the total sales that have some sort of outlier classification
/*

WITH TotalRecords AS (
    SELECT COUNT(*) as total_count
    FROM sale.flag
), NotOutlierCount AS (
    SELECT COUNT(*) as not_outlier_count
    FROM sale.flag
    WHERE sv_outlier_type <> 'Not outlier'
)

SELECT
    ROUND(
        (not_outlier_count * 100.0) / total_count,
        3
    ) AS not_outlier_percentage
FROM
    TotalRecords, NotOutlierCount;
-->

Around **7.2%** of the total sales have some sort of `Outlier` classification.  Within that 7.2%, the makeup of the outlier distribution is approximately as follows:

<!--
/*
This query is used to generate the proportion of different outlier types 
/*

WITH TotalRecords AS (
    SELECT COUNT(*) as total_count
    FROM sale.flag
    WHERE sv_outlier_type <> 'Not outlier'
)

SELECT 
    sv_outlier_type, 
    ROUND((COUNT(*) * 1.0 / total_count) * 100, 2) as proportion
FROM 
    sale.flag 
CROSS JOIN 
    TotalRecords
WHERE 
    sv_outlier_type <> 'Not outlier'
GROUP BY 
    sv_outlier_type, total_count
ORDER BY 
    proportion DESC;
-->

|Outlier Type           |Percentage|
|-----------------------|----------|
|PTAX-203 flag          |40.1      |
|Non-person sale (low)  |17.56     |
|Non-person sale (high) |8.14      |
|Anomaly (high)         |6.35      |
|High price (raw)       |5.77      |
|Low price (raw)        |5.36      |
|Low price (raw & sqft) |5.19      |
|Low price (sqft)       |2.12      |
|Anomaly (low)          |1.96      |
|High price (sqft)      |1.92      |
|Home flip sale (high)  |1.86      |
|High price (raw & sqft)|1.65      |
|Home flip sale (low)   |1.38      |
|Family sale (low)      |0.56      |
|Family sale (high)     |0.05      |
|High price swing       |0.02      |
|Low price swing        |0.01      |



*These outliers are flagged if relevent price columns (log10 transformed and normalized) are 2 standard deviations below or above the mean within a given group*


# Structure of data

All flagging runs populate 3 Athena tables with metadata, flag results, and other information. These tables can be used to determine _why_ an individual sale was flagged as an outlier. The structure of the tables is:

```mermaid
erDiagram
    flag }|--|| metadata : describes
    flag }|--|{ parameter : describes
    flag }|--|{ group_mean : describes

    flag {
        string meta_sale_document_num PK
        date rolling_window
        boolean sv_is_outlier
        boolean sv_is_ptax_outlier
        boolean ptax_flag_original
        boolean sv_is_heuristic_outlier
        string sv_outlier_type
        string run_id FK
        bigint version PK

    }
    metadata {
        string run_id PK
        string long_commit_sha
        string short_commit_sha
        string run_timestamp
        string run_type
    }
    parameter {
        string run_id PK
        bigint sales_flagged
        timestamp earliest_data_ingest
        timestamp latest_data_ingest
        arraystring iso_forest_cols
        arraystring res_stat_groups
        arraystring condo_stat_groups
        arraybigint dev_bounds
        arraybigint ptax_sd
        bigint rolling_window
        string date_floor
        bigint short_term_owner_threshold
        bigint min_group_thresh
    }

    group_mean {
        bigint group_size
        double mean_price
        double mean_price_sqft
        string run_id PK
        string group PK
    }
```

# Important flagging details

## Sales validation pipeline run modes

The Sales Validation Pipeline (hereafter referred to as “the pipeline”) can be executed in three distinct run modes, depending on the state of the sales data and the specific requirements for flagging:

1. **Initial Run:** This mode is triggered when no sales have been flagged. It's the first step in the pipeline to instantiate tables and flag sales.
2. **Glue Job:** This mode applies when there are already flagged sales in the system. It's an automated scheduled job that flags new, unflagged sales.
3. **Manual Update:** This mode is used when sales need to be re-flagged, either due to errors or methodology updates. This allows for the selective re-flagging of sales.

```mermaid

%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryTextColor': '#000',
    }
  }
}%%

graph TD
    subgraph "Manual Update Mode"
        A3{{"Sales must be re-flagged"}}
        B3[Specify subset in yaml]
        C3[Run manual_update.py]
        D3[Increment version if sale already flagged]
        E3[Assign Version = 1 if sale unflagged]
        F3[Update flags in default.vw_pin_sale]
        G3[Save results to S3 with new run ID]

        A3 -->|Manual selection| B3
        B3 -->|Run update| C3
        C3 -->|Version check| D3
        D3 -->|Update process| F3
        C3 -->|New flag| E3
        E3 -->|Update process| F3
        F3 -->|Persist results| G3
    end

    subgraph "Glue Job Mode"
        A2[Schedule triggers glue job]
        B2{{"Some sales are already flagged"}}
        C2[Ingest data for unflagged sales]
        D2[Run flagging model within glue job]
        E2[Write sales data to sale.flag]
        F2[Join flags to<br>default.vw_pin_sale]
        G2[Save results to S3 with unique run ID]

        A2 -->|Trigger| B2
        B2 -->|Process new sales| C2
        C2 -->|Run model| D2
        D2 -->|Output flags| E2
        E2 -->|Join data| F2
        F2 -->|Persist results| G2
    end

    subgraph "Initial Run Mode"
        A1{{"No sales are flagged"}}
        B1[Run initial_flagging.py]
        C1[Flag sales as outliers or non-outliers<br>with Version = 1]
        D1[Join flags to<br>default.vw_pin_sale]
        E1[Save results to S3 with unique run ID]

        A1 -->|Initial setup| B1
        B1 -->|Flag sales| C1
        C1 -->|Store flags| D1
        D1 -->|Persist results| E1
    end

    style A1 fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style B2 fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style A3 fill:#bbf,stroke:#333,stroke-width:2px,color:#000;

```

### Rolling window

The flagging model uses group means to determine the statistical deviation of sales, and flags them beyond a certain threshold. Group means are constructed using a rolling window strategy.

The current implementation uses a 12 month rolling window. This means that for any sale, the "group" contains all sales within the same month, along with all sales from the previous 11 months. This 12 month window can be changed by editing the config files: `manual_flagging/yaml/` and `main.tf`. Additional notes on the rolling window implementation:

- We take every sale in the same month of the sale date, along with all sale data from the previous N months. This window contains roughly 1 year of data.
- This process starts with an `.explode()` call. Example [here](https://github.com/ccao-data/model-sales-val/blob/283a1403545019be135b4b9dbc67d86dabb278f4/glue/sales_val_flagging.py#L15).
- It ends by subsetting to the `original_observation` data. Example [here](https://github.com/ccao-data/model-sales-val/blob/499f9e31c92882312051837f35455d078d2507ee/glue/sales_val_flagging.py#L57).
- Corresponding functions in [Mansueto](https://miurban.uchicago.edu/)'s flagging model accommodate this rolling window integration, these functions are defined `glue/flagging_script_glue/flagging.py`.

# AWS Glue job documentation

This repository manages the configurations, scripts, and details for an AWS Glue Job. It's essential to maintain consistency and version control for all changes related to the job. Therefore, specific procedures have been established.

## ⚠️ Important guidelines

1. **DO NOT** modify the Glue job script, its associated flagging python script, or any of its job details directly via the AWS Console.
2. All changes to these components should originate from this repository. This ensures that every modification is tracked and version-controlled.
3. The **only** advisable actions in the AWS Console concerning this Glue job are:
    - Running the job
4. To test a change to the Glue job script or the flagging script, make an edit on a branch and open a pull request. Our GitHub Actions configuration will deploy a staging version of your job, named `ci_<your-branch-name>_sales_val_flagging`, that you can run to test your changes. See the [Modifying the Glue job](#modifying-the-glue-job-its-flagging-script-or-its-settings) section below for more details.

## Modifying the Glue job, its flagging script, or its settings

The Glue job and its flagging script are written in Python, while the job details and settings are defined in a [Terraform](https://developer.hashicorp.com/terraform/intro) configuration file. These files can be edited to modify the Glue job script, its flagging script, or its job settings.

1. Locate the desired files to edit:
    - Glue script: `glue/sales_val_flagging.py`
    - Flagging script: `glue/flagging_script_glue/flagging.py`
    - Job details/settings: `main.tf`, under the resource block `aws_glue_job.sales_val_flagging` (see [the Terraform AWS provider docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/glue_job) for details)
2. Any changes to these files should be made in the following sequence:
    - Make a new git branch for your changes.
    - Edit the files as necessary.
    - Open a pull request for your changes against the `main` branch. A GitHub Actions workflow called `deploy-terraform` will deploy a staging version of your job named `ci_<your-branch-name>_sales_val_flagging` that you can run to test your changes.
      - By default, this configuration will deploy an empty version of the `sale.flag` table, which simulates an environment in which there are no preexisting flags prior to a run.
      - If you would like to test your job against a subset of the production data, copy your data subset from the production job bucket to the bucket created by Terraform for your job (or leave the new bucket empty to simulate running the job when no flags exist). Then, run the crawler created by Terraform for your PR in order to populate the staging version of the `sale.flag` database that your staging job uses. If you're having trouble finding your staging bucket, job, or crawler, check the GitHub Actions output for the first successful run of your PR and look for the Terraform output displaying the IDs of these resources.
    - If you need to make further changes, push commits to your branch and GitHub Actions will deploy the changes to the staging job and its associated resources.
    - Once you're happy with your changes, request review on your PR.
    - Once your PR is approved, merge it into `main`. A GitHub Actions workflow called `cleanup-terraform` will delete the staging resources that were created for your branch, while a separate `deploy-terraform` run will deploy your changes to the production job and its associated resources.
  
# Next steps

## iasWorld Integration

We plan to integrate this data into our iasWorld database, so that these flags are widely available in the CCAO. We plan to upload 4-5 columns:
- Is non-arms-length (boolean)
- Flag reason (text)
- Flag run ID (hidden)
- Analyst override (boolean)
- Analyst Notes (text)

```mermaid

graph TD
    A[Data Team Data Lake] --> C1[Is non-arms-length]
    A --> C2[Flag reason]
    A --> C3[Flag run ID]
    A --> C4[Analyst override]
    
    C1 --> J[Join on doc_no]
    C2 --> J
    C3 --> J
    C4 --> J
    J --> D[iasWorld Sales Data]

    class C1 red;
    class C2 red;
    class C3 red;
    class C4 red;

```

## Analyst Review

In the future, it is our goal that human sales validation analysts will be able to review these flags and manually override them. In the longer-term future we could train a supervised model using the analysts' judgement.
