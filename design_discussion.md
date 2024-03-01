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

The `geography` key defines all of the pins needed for the statistical groups to be calculated. If we want to filter 




To flesh out: 
- Add yaml option for data read in
- Define ranking for which method takes precedence
