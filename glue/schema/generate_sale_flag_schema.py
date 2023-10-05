# Generate a sale_flag_schema.parquet file that is completely empty
# except for metadata that can be used to construct the sale.flag table.
# This is useful for deploying an empty version of the sale.flag table
# for testing Glue job runs.
import pyarrow as pa
import pyarrow.parquet as pq


if __name__ == "__main__":
    schema = {
        "meta_sale_document_num": "string",
        "rolling_window": "date64",
        "sv_is_outlier": "int64",
        "sv_is_ptax_outlier": "int64",
        "ptax_flag_original": pa.bool_(),
        "sv_is_heuristic_outlier": "int64",
        "sv_outlier_type": "string",
        "run_id": "string",
        "version": "int64",
    }
    table_data = pa.table([[] for _ in schema.keys()], names=[col for col in schema.keys()])
    schema = pa.schema([pa.field(col, dtype) for col, dtype in schema.items()])
    table = table_data.cast(schema)
    pq.write_table(table, "sale_flag_schema.parquet")
