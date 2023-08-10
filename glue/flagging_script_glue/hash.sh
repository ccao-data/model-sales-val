#!/bin/bash

# HERE need to set correct ws within repo
# use this probably
# git rev-parse --show-toplevel

# find existing flagging script
existing_flagging_file=$(ls | grep -E '^flagging_[0-9a-z]{6}\.py$')

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

# Get glue bucket
bucket_name="$AWS_S3_GLUE_BUCKET"

# Delete files that start with 'flagging' followed by an underscore and 6 numbers/letters with the .py file type
aws s3 ls "$bucket_name/scripts/sales-val/" | awk '{print $4}' | grep -E '^flagging_[0-9a-z]{6}\.py$' | while read -r file
do
  aws s3 rm "$bucket_name/scripts/sales-val/$file"
done

# Upload a new file to the same bucket
aws s3 cp "$flag_hash_script" "$bucket_name/scripts/sales-val/$flag_hash_script"
