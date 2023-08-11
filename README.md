# model-sales-val (docs in progress) 
  
| :exclamation:  IMPORTANT   |
|-----------------------------------------|  

If an edit needs to made to anything in the `glue` directory, there a specific process to be found [here](#aws-integration)  
  
Table of Contents
================

- [Overview](#overview)  
- [Structure of Data](#structure-of-data)  
- [Flagging Details](#flagging-details)
- [AWS Glue integration](#aws-integration)

## Overview
The model-sales-val system is a critical component of our data integrity framework, designed to oversee the complex process of identifying and flagging sales that may be non-arms-length transactions. These sales can distort our analyses and models, since they don't adhere to the principle of an open and competitive market. A non-arms-length sale occurs when the buyer and seller have a relationship that might influence the transaction price, leading to a sale that doesn't reflect the true market value of the property. This relationship might exist between family members, business partners, or other close connections.

The workflow of the sales flagging is as follows:  
* There will be an intitial run of the `manual_flagging/initial_flagging.py`, which instantiates all tables, and flags a large portion of the sales as either outliers or non-outliers.
* Then, we have `glue/sales-val-flagging.py`, a script connected to AWS glue that flags all new unflagged sales. This script will be automated such that it runs on a schedule (eg. monthly).  
* In the case of an error with the flagging or if we want to update the methodology on already-flagged sales, we can run the `manual_flagging/initial_flagging.py` and select a subset of sales to re-flag. These updated values will have a version number that is 1 higher than the previous sale. When utilizing our sales views, we will pull the flag data with the highest version value to keep it up-to-date.  


## Structure of Data

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


## AWS Integration  

| :exclamation: **If you want to update anything in `glue/` look below for the process**  |
|-----------------------------------------|   

  
  
We **should not** update the glue job script, its corresponding flagging python script, or any of its job details in the aws-console. The reason for this is so that we can track changes and version control using this repo. The only actions that should be taken in the aws console related to this glue job are running the job, and pulling from the repo using their version control system.  

### To make changes to glue job script or job details

* The glue script is `glue/sales-val-flagging.py`
* Job details / settings associated with the glue job are here: `glue/sales-val-flagging.json`

Both the script and the glue job properties are linked through aws glue's version control integration with github. This means that after any change made in either of these files, they be done in this order:  
* push changes to master
* pull changes from aws console  

In order to pull from the repository into AWS Glue, there will need to be a personal access token used. Instructions can be found [here](https://aws.amazon.com/blogs/big-data/code-versioning-using-aws-glue-studio-and-github/) for authentication. This can be done working in the `Version Control` tab of the AWS glue job.

### To make changes to flagging script in S3

The way we track changes in the S3 flagging script is through a hash identifier. The name of the script will have the first 6 characters of a hash appended to it, then these 6 characters will be written to the `sale.metadata` table in athena for lookup. This will allow us to find any flagging file used by the glue script by finding the commit hash and looking at the flagging file in `glue/flagging_script_glue/`. The way we implement this is with the bash script `glue/flagging_script_glue/hash.shj`. After changing and saving the script, we run `hash.sh` which rehashes the newly updated file, udpates the file name to include the first 6 characters of the hash, and deletes the old flagging file. This update happens in **both** the S3 bucket and in our repo.
  
If we need to make changes to the flagging script (located in an S3 bucket) used by the glue script, we want to change the python file in the `glue/flagging_script_glue/` directory.   

Steps to make changes to flagging script:   
* Save changes locally
* Run `hash.sh`
* Push to master


