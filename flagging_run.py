import pandas as pd
import subprocess as sp
import numpy as np
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


s3_file = "s3://ccao-data-warehouse-us-east-1/sale/val_test/d8947f0d-89d4-4114-ab88-6c9739951e95.csv"
df = pd.read_csv(s3_file)

# ----- data type conversion problems -----
df['meta_sale_date'] = pd.to_datetime(df['meta_sale_date'])
df = df[df['class'] != 'EX'] #incorporate into athena pull?
# --------

# run outlier flagging methodology 
df_flag = flg.go(df=df, 
                 groups=tuple(inputs['stat_groups']),
                 iso_forest_cols=inputs['iso_forest'],
                 dev_bounds=tuple(inputs['dev_bounds']))

# utilize ptaxsim to complete
df_final = (df_flag
      .rename(columns={'sv_is_outlier': 'sv_is_autoval_outlier'})
      .assign(sv_is_autoval_outlier = lambda df: df['sv_is_autoval_outlier'] == "Outlier")
      .assign(sv_is_outlier = lambda df:
               df['sv_is_autoval_outlier'] | df['sale_filter_is_outlier'])
      .assign(sv_outlier_type = lambda df: 
              np.where((df['sv_outlier_type'] == "Not outlier") & df['sale_filter_is_outlier'], 
                        "PTAX-203 flag", df['sv_outlier_type']))
      .assign(sv_is_outlier = lambda df: df['sv_outlier_type'] != "Not outlier"))
