# model-sales-val

**Docs in progress**


## Table of Contents

[Overview](https://github.com/ccao-data/model-sales-val#overview)

[Structure](https://github.com/ccao-data/model-sales-val#structure)

[Flagging Details](https://github.com/ccao-data/model-sales-val#flagging-details)

## Overview
The model-sales-val system is a critical component of our data integrity framework, designed to oversee the complex process of identifying and flagging sales that may be non-arms-length transactions. These sales can distort our analyses and models, since they don't adhere to the principle of an open and competitive market. A non-arms-length sale occurs when the buyer and seller have a relationship that might influence the transaction price, leading to a sale that doesn't reflect the true market value of the property. This relationship might exist between family members, business partners, or other close connections.


## Structure
This repo is split into two parts.
* `initial_run_local/`
* `glue/`

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



