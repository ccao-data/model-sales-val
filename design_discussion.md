# Design Discussion Document. 

### Stat groups schema
Here is an early prototype for a new config schema. I think there are ways to make this cleaner and more readable, but so far this is a functional example insofar as it [sets up data to be flagged](https://github.com/ccao-data/model-sales-val/blob/98-make-flagging-script-more-flexible-with-respect-to-geography/manual_flagging/flagging.py#L209-L245), even better than the current method on main. The current config can be viewed in `manual_flagging/yaml/inputs.yaml` The following structure holds an entire config at the level of the `geography` key.

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
  
Within the `geography_key`, we can specify different markets. For example, with our current city tri and north tri configs:
  
```yaml
geography: city_tri
  res_multi_family:
  res_single_family:
  condos:
geography: north tri
  res_all:
  condos:
```
  
where each of these markets (one level under the `geography`) have their own filters, grouping columns, etc. If we didn't need any further market partition after the geography, we could just use a single value:
```yaml
geography:
  all_properties:

    rest_of_config

```
### Run Config
To speficy a run configuration for we can use these data structures in the yaml file:

```yaml
run_geography: []
sales_to_write_filter:
  column:
  values:
```
where the `run_geography` is an array (athena) or list (python) object that takes in any number of `geography` configurations from `stat_groups_map`. We could have `run_geography = ["north_tri", "south_tri"]` or any type of geographically defined flagging config. If we want to target a specific update we can use `sales_to_write_filter`. For example, let's say we want to update flags that use the `city_tri` config but we only want to update flags in a specific census tract within that config. We can specify:

```yaml
run_geography: ["city_tri]
sales_to_write_filter:
  column: "census_tract"
  values: "some_census_tract_value"
```
  
  W

<>  
<>  
<>    
To flesh out: 
- Add yaml option for data read in
- Define ranking for which method takes precedence
