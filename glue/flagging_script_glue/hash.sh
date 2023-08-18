#!/bin/bash

: <<comment
This script is to be run after an update to the flagging script 
in ccao_sales_val/glue/flagging_script_glue/. This script will update 
the hash of the new flagging script, rename the edited file using 
the new hash, and delete the previous file. 

Then, the old flagging script will also be deleted and replaced with 
new file and its corresponding hash identifier in AWS.
comment

# Standardize paths
toplevel=$(git rev-parse --show-toplevel)
cd "$toplevel/glue/flagging_script_glue" || exit 1 # Exit the script if the directory doesn't exist or is inaccessible

# Find existing flagging script
existing_flagging_file=""
for file in *; do
    if [[ $file =~ ^flagging_[0-9a-z]{6}\.py$ ]]; then
        existing_flagging_file="$file"
        break
    fi
done

# Check if we found a matching file
if [[ -z $existing_flagging_file ]]; then
  echo "Error: No matching flagging script found."
  exit 1
fi

# Extract the hash part from the existing file
existing_hash=${existing_flagging_file:9:6}

hash=$(md5sum "$existing_flagging_file" | awk '{print $1}')
short_hash=${hash:0:6}

# Compare the existing hash with the newly computed short hash
if [ "$existing_hash" == "$short_hash" ]; then
  echo "Exiting: identical hash, no changes made to file"
  exit 0
fi

flag_hash_script="flagging_$short_hash.py"
mv "$existing_flagging_file" "$flag_hash_script"

bucket_name="$AWS_S3_GLUE_BUCKET"

# Delete files that start with 'flagging' followed by an underscore and 6 numbers/lower_case letters with the .py file type
aws s3 ls "$bucket_name/scripts/sales-val/" | awk '{print $4}' | grep -E '^flagging_[0-9a-z]{6}\.py$' | while read -r file
do
  aws s3 rm "$bucket_name/scripts/sales-val/$file"
done
echo "flag_hash_script: $flag_hash_script"
echo "bucket_name: $bucket_name"
# Upload a new file to the same bucket
aws s3 cp "$flag_hash_script" "$bucket_name/scripts/sales-val/$flag_hash_script"
