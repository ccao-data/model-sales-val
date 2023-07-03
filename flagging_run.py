import pandas as pd
import subprocess as sp
import os
import yaml

# import flagging functions
from src import flagging as flg

# set working to root, to pull from source
root = sp.getoutput('git rev-parse --show-toplevel')
os.chdir(root)

# inputs yaml as inputs
with open("inputs.yaml", 'r') as stream:
    try:
        inputs = yaml.safe_load(stream)
        print(inputs)
    except yaml.YAMLError as exc:
        print(exc)

s3_file = "s3://ccao-data-warehouse-us-east-1/sale/val_test/80900d4f-ad2f-454d-9eac-6dd0e17dedf5.csv"
df = pd.read_csv(s3_file)

# ----- data type conversion problems -----
df['meta_sale_date'] = pd.to_datetime(df['meta_sale_date'])
df = df[df['class'] != 'EX'] #incorporate into athena pull
# --------

df_final = flg.go(df=df, 
                  groups=tuple(inputs['stat_groups']),
                  iso_forest_cols=inputs['iso_forest'],
                  dev_bounds=tuple(inputs['dev_bounds']))