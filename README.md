# model-sales-val (docs in progress) 
  
| :exclamation:  IMPORTANT   |
|-----------------------------------------|  

If an edit needs to made to anything in the `glue` directory, there a specific process to be found [here](#aws-glue-job-documentation)  
  
Table of Contents
================

- [Overview](#overview)  
- [Structure of Data](#structure-of-data)  
- [Important Flagging Details](#important-flagging-details)
- [AWS Glue integration](#aws-glue-job-documentation)

# Overview
The model-sales-val system is a critical component of our data integrity framework, designed to oversee the complex process of identifying and flagging sales that may be non-arms-length transactions. These sales can distort our analyses and models, since they don't adhere to the principle of an open and competitive market. A non-arms-length sale occurs when the buyer and seller have a relationship that might influence the transaction price, leading to a sale that doesn't reflect the true market value of the property. This relationship might exist between family members, business partners, or other close connections.

The workflow of the sales flagging is as follows:  
* There will be an intitial run of the `manual_flagging/initial_flagging.py`, which instantiates all tables, and flags a large portion of the sales as either outliers or non-outliers.
* Then, we have `glue/sales_val_flagging.py`, a script connected to AWS glue that flags all new unflagged sales. This script will be automated such that it runs on a schedule (eg. monthly).  
* In the case of an error with the flagging or if we want to update the methodology on already-flagged sales, we can run the `manual_flagging/manual_update.py` and select a subset of sales to re-flag. These updated values will have a version number that is 1 higher than the previous sale. When utilizing our sales views, we will pull the flag data with the highest version value to keep it up-to-date.

On the left, we see the normal workflow of the process. Represented on the right is the use of `manual_update.py` to update/re-flag sales.  

  ``` mermaid
graph TD

    A[Initialize]
    B{{Create Initial Table<br>of Flagged Values}}
    C{{Perform Reoccurring Job:<br>Flag New Non-Flagged Values}}
    D[>Some Need to Re-Flag]
    E[Subset Sales in yaml - Run manual_update.py]
    F[If Sale Already Flagged<br>Incremenet Version Column by 1]
    G{If Sale Unflagged,<br>Assign Version 1}

    class B,C recurringProcess;
    class D,E,F,G secondaryProcess;

    A --> B
    B --> C

    D --> E
    E --> G
    G --> F
    E --> F

    classDef recurringProcess fill:#f9d6d6,stroke:#f26a6a,stroke-width:2px;
    classDef secondaryProcess fill:#e0e0e0,stroke:#a0a0a0,stroke-width:1px;


  ```


# Structure of Data  
With any flagging done, there will be 3 auxiliary tables produced with contain metadata and other information that can help track down exactly why any individual sale was flagged as an outlier.  Structure of data production below:  

``` mermaid
erDiagram
    flag ||--|{ metadata : describes
    flag ||--|{ parameter : describes
    flag ||--|{ group_mean : describes
    
    flag {
        string meta_sale_document_num PK
        date rolling_window
        bigint sv_is_outlier
        bigint sv_is_ptax_outlier
        bigint sv_is_heuristic_outlier
        string sv_outlier_type
        string run_id 
        bigint version PK

    }
    metadata {
        string run_id PK
        string long_commit_sha
        string short_commit_sha
        string run_timestamp
        string run_type
        string flagging_hash
    }
    parameter {
        string run_id PK
        bigint sales_flagged
        timestamp earliest_data_ingest
        timestamp latest_data_ingest 
        bigint short_term_owner_threshold
        arraystring iso_forest_cols
        arraystring stat_groups
        arraybigint dev_bounds
    }

    group_mean {
        double mean_price
        double mean_price_sqft
        string run_id PK
        string group PK
    }
```

# Important Flagging Details  

### Rolling Window    
In a number of outlier calculations, our flagging model looks inside a group of sales and classifies them as outliers based on whether or not they are a certain standard devation away from the mean. In order to make these groups, we use a rolling window strategy. The current implementation uses a 12 month rolling window. This means that for any sale in a month, the group for these sales is within the given month of the sales, along with all sale data from the previous 11 months. This 12 month figure can be changed by editing the config files: `manual_flagging/yaml/` and `glue/sales_val_flagging.json` files. Here are details on where this code lives:  
- We take every sale in the month of the sale date, along with all sale data from the previous N months. This window contain roughly 1 year of data  
- This process start here with a `.explode()` of the data:  https://github.com/ccao-data/model-sales-val/blob/283a1403545019be135b4b9dbc67d86dabb278f4/glue/sales_val_flagging.py#L15  
- And it ends here subsetting subsetting to the `original_observation` data:  https://github.com/ccao-data/model-sales-val/blob/499f9e31c92882312051837f35455d078d2507ee/glue/sales_val_flagging.py#L57  
- Corresponding functions in Mansueto's flagging model accomodate this rolling window integration, these functions are defined in each of the flagging functions, one in `manual_flagging/src/flagging_rolling.py`, and one for the glue job in s3 `glue/flagging_script_glue/flagging_<hash>.py`:     
    - https://github.com/ccao-data/model-sales-val/blob/499f9e31c92882312051837f35455d078d2507ee/manual_flagging/src/flagging_rolling.py#L303
    - https://github.com/ccao-data/model-sales-val/blob/283a1403545019be135b4b9dbc67d86dabb278f4/manual_flagging/src/flagging_rolling.py#L456

# AWS Glue Job Documentation

This repository manages the configurations, scripts, and details for an AWS Glue Job. It's essential to maintain consistency and version control for all changes related to the job. Therefore, specific procedures have been established.

## ⚠️ Important Guidelines

1. **DO NOT** modify the Glue job script, its associated flagging python script, or any of its job details directly via the AWS Console.
2. All changes to these components should originate from this repository. This ensures that every modification is tracked and version-controlled.
3. The **only** advisable actions in the AWS Console concerning this Glue job are:
    - Running the job
    - Pulling updates from the repo through AWS's version control system.

## Modifying the Glue Job Script or Details

1. Locate the desired files:
    - Glue script: `glue/sales_val_flagging.py`
    - Job details/settings: `glue/sales_val_flagging.json`
2. Any changes to these files should be made in the following sequence:
    - Push modifications to the master branch of this repo.
    - Pull these changes from the AWS Console.

    > Note: AWS Glue is integrated with GitHub for version control. Ensure you have the necessary authentication. If required, use a personal access token. See [this guide](https://aws.amazon.com/blogs/big-data/code-versioning-using-aws-glue-studio-and-github/) for more details. Make these changes from the `Version Control` tab of the AWS Glue job.

## Modifying the S3 Flagging Script

The S3 flagging script `glue/flagging_script_glue/flagging_<hash>.py` is uniquely identified through a hash. This helps in tracking changes efficiently.

### How Hashing Works:

- The script name will include the first 6 characters of its hash.
- Upon execucution of a glue job, these characters are then logged in the `sale.metadata` table in Athena.
- This enables us to track and locate any flagging file used by the Glue script, matching the commit hash and the flagging file within `glue/flagging_script_glue/`.

The hashing process utilizes the bash script `glue/flagging_script_glue/hash.sh`.

### Steps to Modify the S3 Flagging Script:

1. Edit file: `glue/flagging_script_glue/hash.sh`
2. Save your changes locally.
3. Run the `hash.sh` script. This action:
    - Rehashes the updated file.
    - Renames the file, appending the first 6 characters of the new hash.
    - Removes the previously hashed flagging file.
4. The updated hash file should reflect in both the S3 bucket and this repository.
5. Push the changes to the master branch.

