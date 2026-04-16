# This script should run the training and save checkpoints in the output/ dir

#!/bin/bash


START_TIME=$(date +%s)
echo "Script started at: $(date)"

set -e
cd ..
export PYTHONPATH=$(pwd)

OUTPUT_DIR=Q1/$1

# python train.py \
#   --train_file data/train_en.jsonl \
#   --output_dir $OUTPUT_DIR

python -m unsupervised_corpus.unsupervised_corpus
python -m Q1.pretrain $OUTPUT_DIR
python -m Q1.train $OUTPUT_DIR

END_TIME=$(date +%s)
echo "Script ended at: $(date)"

DURATION=$((END_TIME - START_TIME))
echo "Total runtime: ${DURATION} seconds"
