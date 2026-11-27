# Flag review

Valuations analysts review a selection of sales by hand each year. We export
workbooks of sales for them to fill in, and their answers are loaded into the
`sale.flag_review` table in Athena. `sale.vw_outlier` combines that table with
the flags from the sales validation model.
