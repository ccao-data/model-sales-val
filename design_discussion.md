# Design Discussion Document

## Stat groups schema
Here is an early prototype for a new config schema. I think there are ways to make this cleaner and more readable, but so far this is a functional example insofar as it [sets up data to be flagged](https://github.com/ccao-data/model-sales-val/blob/98-make-flagging-script-more-flexible-with-respect-to-geography/manual_flagging/flagging.py#L209-L245), even better than the current set-up loop on main. The only thing not incorporated into the set up loop is the binning features. The current config can be viewed in `manual_flagging/yaml/inputs.yaml` on this branch. The following structure holds an entire config at the level of the `geography` key. 
  
Defining terms:
- In the rest of this document `geography` will refer to this second-level `geography` key below which is *all data required to run flagging over all of the statistical groupings necessary to create groups*.
- `Statistical grouping columns` will refer to the column by which the statistical groups are divided up (township/groups of nbhds/bins/rolling window).
  
This is an example structure with no real data filled in. In the actualy design, `geography` and `market_type$n` are replaced with the actual names such as `north_tri` or `res_single_fam`.  

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
      iso_forest:
      example_bin_specification:
    market_type2:
    ...
    ...
    ...
```
  
The `geography` key defines all of the pins needed for the statistical groups to be calculated. If we want to assign sales for all neighborhoods in the city tri we would filter using the `data_filter` key to grab all city tri sales. If we wanted to flag within 10 census tracts, we could use a name like `xyz_census_tracts` instead of `city_tri` and define the geography in the `data_filter` key. This works cleanly if we want to write flags for the same sales necessary for the statistical groupings. If that is not the case and we want to flag a subset of the sales necessary for the statistical grouping, we can use the `sales_to_write` filter.  
  
Here our `geography` key is `city_tri`. Within the `geography` key, we can specify different markets. For example, with our current city tri and north tri configs:
  
```yaml
city_tri:
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
## Run Config
To speficy a run configuration for we can use these data structures in the yaml file:

```yaml
run_geography: []
sales_to_write_filter:
  column:
  values:
```
where the `run_geography` is an array (athena) or list (python) object that takes in any number of `geography` configurations from `stat_groups_map`. We could have `run_geography = ["north_tri", "south_tri"]` or any type of geographically defined flagging config. If we want to target a specific update we can use `sales_to_write_filter`. For example, let's say we want to manually update flags that use the `city_tri` config but we only want to update flags in a specific census tract within that config. We can specify:

```yaml
run_geography: ["city_tri]
sales_to_write_filter:
  column: "census_tract"
  values: "some_census_tract_value"
```
  
## Oustanding engineering questions

### Add yaml option for data read in

One thing I haven't yet taken care of is the integration of the external data. We currently have the neighborhood groupings defined for the city tri in the `data/` directory. One idea is that we could include the name of the excel/cvs file as a key value somewhere in the `stat_groups_map` schema. It could work similarly to the bin specification where if we see the value in the config, we act on it and join the data such that we get a new column that can be used for grouping. Maybe something like:

```yaml
data_to_join:
  file_name: "file.csv"
  shared_key: "column_to_join_on"

```

### Define ranking for which method takes precedence

Even with the entire configuration above we still run into an issue. This is the case of non-mutually exclusive geographies used for flagging. This is mostly a problem for the recurring script which will automatically flag any non-flagged sales on a schedule. If we have mutually exclusive grouping columns for every single sale/every single geography specification, then this is not a problem. However if there is overlap then it will be a problem. 
  
I'll explain the problem with an example. Suppose we have `city_tri`, `north_tri` and `south_tri` geography. City tri and south tri have mutually exlusive statistical gropuping columns (township and nbhd groups both fit neatly in townships). But let's say that instead of north tri, we filter the `geography` configs on the census tracts that are in the north tri area. Some of these census tracts extend in the city tri. This results in some pins in both the new census tract geography around the north tri and some pins in the city_tri geography. In this situation in a recurring script we would have three values in the `run_geography`: `run_geography: [south_tri, city_tri, new_census_tracts_that_include_north_tri_pins]`. The default setting for this job would be this specificaiton. In this event where multiple pins are flagged with both `city_tri` and `new_census_tracts_that_include_north_tri_pins`, I think we could could assign a ranking system for the methodologys that would formalize the following logic: If there are pins with mulitple flag values we keep the flag value according the the ranking specification. This could be something like:

```yaml
city_tri: 1
new_census_tracts_that_include_north_tri_pins: 2
south_tri: 3
```

where the lowest number takes precedence. 

## Other considerations

I think this method ia good in terms of flexibility, but it is burdensome for a user to add a new specification, especially if we are adding a new `geography` config. I think we can split up this data structure better.
