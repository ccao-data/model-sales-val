import os
import pandas as pd
import subprocess as sp
import time

from pyathena import connect
import boto3
import s3fs

# import flagging functions
from src import flagging 

# Define s3 and Athena paths

# set working to root, to pull from source
root = sp.getoutput('git rev-parse --show-toplevel')
os.chdir(root)


# Define AWS boto3 clients
athena_client = boto3.client('athena')
glue_client = boto3.client('glue', region_name='us-east-1')
s3_client = boto3.client('s3')

# Define s3 and Athena paths
athena_db = 'default'

s3_bucket = 'ccao-data-warehouse-us-east-1'
s3_prefix = 'sale/val_test'
s3_output = 's3://'+ s3_bucket + '/' + s3_prefix


# Functions to help with Athena queries ----
def poll_status(athena_client, execution_id):
    """ Checks the status of the a query using an incoming execution id and returns
    a 'pass' string value when the status is either SUCCEEDED, FAILED or CANCELLED. """

    result = athena_client.get_query_execution(QueryExecutionId=execution_id)
    state  = result['QueryExecution']['Status']['State']

    if state == 'SUCCEEDED':
        return 'pass'
    if state == 'FAILED':
        return 'pass'
    if state == 'CANCELLED':
        return 'pass'
    else:
        return 'not pass'

def poll_result(athena_client, execution_id):
    """ Gets the query result using an incoming execution id. This function is ran after the
    poll_status function and only if we are sure that the query was fully executed. """

    result = athena_client.get_query_execution(QueryExecutionId=execution_id)

    return result

def run_query_get_result(
  athena_client,
  s3_bucket,
  query,
  database,
  s3_output,
  s3_prefix):
    """ Runs an incoming query and returns the output as an s3 file like object.
    """

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': s3_output,
    })

    QueryExecutionId = response.get('QueryExecutionId')

    # Wait until query is executed
    while poll_status(athena_client, QueryExecutionId) != 'pass':
        time.sleep(2)
        pass

    result = poll_result(athena_client, QueryExecutionId)

    data_object = None

    # Only return file like object when the query succeeded
    if result['QueryExecution']['Status']['State'] == 'SUCCEEDED':
        print("Query SUCCEEDED: {}".format(QueryExecutionId))

        s3_key = s3_prefix + '/' + QueryExecutionId + '.csv'  # note the additional '/'
        data_object = boto3.resource('s3').Object(s3_bucket, s3_key)

    # troubleshooting ------------
    response = athena_client.get_query_execution(
    QueryExecutionId = QueryExecutionId
    )
    print(response['QueryExecution']['Status']['State'])
    #print(response['QueryExecution']['Status']['StateChangeReason'])

    return data_object

SQL_QUERY = """
SELECT
    sale.sale_price AS meta_sale_price,
    sale.sale_date AS meta_sale_date,
    sale.doc_no AS meta_sale_document_num,
    sale.seller_name AS meta_sale_seller_name,
    sale.buyer_name AS meta_sale_buyer_name,
    sale.sale_filter_is_outlier,
    res.class AS class,
    res.township_code AS township_code,
    res.year AS year,
    res.pin AS pin,
    res.char_bldg_sf AS char_bldg_sf
FROM default.vw_card_res_char res
INNER JOIN default.vw_pin_sale sale
    ON sale.pin = res.pin
    AND sale.year = res.year
WHERE (res.year
    BETWEEN '2014'
    AND '2022')
AND NOT sale.is_multisale
AND Year(sale.sale_date) >= 2014
"""


# Run run_query_get_result to get file like object ----
data_object = run_query_get_result(
    athena_client,
    s3_bucket,
    SQL_QUERY,
    athena_db,
    s3_output,
    s3_prefix
)


"""
target = 's3://'+ s3_bucket + '/' + data_object.key
pull = pd.read_csv(target)

pull.to_csv('training_data.csv', index=False)
"""
