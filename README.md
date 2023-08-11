# model-sales-val (docs in progress) 
  
| :exclamation:  IMPORTANT   |
|-----------------------------------------|  

If an edit needs to made to anything in the `glue` directory, there a specific process to be found [here](#aws-integration)  
  
Table of Contents
================

- [Overview](#overview)  
- [Contents](#contents)  
- [Structure of Data](#structure-of-data)  
- [Flagging Details](#flagging-details)
- [AWS Glue integration](#aws-integration)

## Overview
The model-sales-val system is a critical component of our data integrity framework, designed to oversee the complex process of identifying and flagging sales that may be non-arms-length transactions. These sales can distort our analyses and models, since they don't adhere to the principle of an open and competitive market. A non-arms-length sale occurs when the buyer and seller have a relationship that might influence the transaction price, leading to a sale that doesn't reflect the true market value of the property. This relationship might exist between family members, business partners, or other close connections.

## Contents
This repo is split into two parts.
#### `manual_flagging/`  
This directory hold all files needed to run the `initial_flagging.py` and `manual_update.py` scripts.
#### `glue/`  
This is the reccurent script that runs using AWS glue. Both `glue/sales-val-flagging.json` and `glue/sales-val-flagging.py` are linked to AWS.


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



## Flagging Details

A heuristics-based model has been developed with the help of the Mansueto institute to identify and flag sales as potential outliers, categorizing them into the following types:

* High price (raw)
* High price (raw & sqft)
* Home flip sale (low)
* Low price (sqft)
* Home flip sale (high)
* Not Outlier
* Non-person sale (low)
* Non-person sale (high)
* High price (sqft)
* PTAX-203 flag
* Anomaly (low)
* Family sale (low)
* Anomaly (high)
* Low price (raw & sqft)
* Low price (raw)
* Family sale (high)

## AWS Integration



