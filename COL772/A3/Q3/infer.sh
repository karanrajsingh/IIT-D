#!/bin/bash

lang=$1
test_file=$2
output_dir=$3

START_TIME=$(date +%s)
echo "Script started at: $(date)"

mkdir -p $output_dir

python valid_json.py
python convert_jsonl.py
python icl.py $lang $test_file $output_dir


END_TIME=$(date +%s)
echo "Script ended at: $(date)"

DURATION=$((END_TIME - START_TIME))
echo "Total runtime: ${DURATION} seconds"
