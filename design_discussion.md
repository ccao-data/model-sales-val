# Design Discussion Document. 

Here is an early prototype for a new config schema. I think there are ways to make this cleaner and more readable, but so far this is a functional example insofar as it [sets up data to be flagged](https://github.com/ccao-data/model-sales-val/blob/98-make-flagging-script-more-flexible-with-respect-to-geography/manual_flagging/flagging.py#L209-L245), even better than the current method on main. 

```yaml
run_geography: []
sales_to_write_filter:
  column:
  values:
```

```yaml
stat_groups_map:
  geography:
    data_filter:
      column: 
      values: 
    market_type1:
      columns:
      data_filter:
        column: 
        values: 
      iso_forest: "res"
      example_bin_specification:
    market_type2:
    ...
    ...
    ...
```
  
The `geography` key defines all of the pins needed for the statistical groups to be calculated. If we want to assign sales for all neighborhoods in the city tri we would filter using the `data_filter` key to grab all city tri sales. If we wanted to flag within 10 census tracts, we could also use `data_filter`. This works cleanly if we want to write flags for the same sales necessary for the statistical groupings. If that is not the case and we want to flag a subset of the sales necessary for the statistical grouping, we can use the `sales_to_write` filter.
  
Within the `geography_key`, we can specify different markets. For example, in our current city tri and north tri configs it looks like 
  
```yaml
geography: city_tri
  res_multi_family:
  res_single_family:
  condos:
geography: north tri
  res_all:
  condos:
```
  
where each of these markets (one level under the `geography`) have their own filters, grouping columns, etc. If we didn't need any further market partition after the geography, we could just use a single value like 
  
```yaml
geography:
  all_properties:

    rest_of_config

```
  
To speficy a run configuration for this 



To flesh out: 
- Add yaml option for data read in
- Define ranking for which method takes precedence
