hash=$(md5sum flagging.py | awk '{print $1}')
short_hash=${hash:0:6}
flag_hash_script="flagging_$short_hash.py"
echo $flag_hash_script
# 