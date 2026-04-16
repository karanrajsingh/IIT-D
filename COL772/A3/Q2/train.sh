# This script should run the training and save checkpoints in the output/ dir

#!/bin/bash

# OUTPUT_DIR=$1

# mkdir -p $OUTPUT_DIR

# python train.py data/train_en.jsonl $OUTPUT_DIR


START_TIME=$(date +%s)
echo "Script started at: $(date)"

set -e
cd ..
export PYTHONPATH=$(pwd)

OUTPUT_DIR=Q2/$1

mkdir -p $OUTPUT_DIR

python -m unsupervised_corpus.unsupervised_corpus
python -m Q2.pretrain $OUTPUT_DIR
python -m Q2.train $OUTPUT_DIR


END_TIME=$(date +%s)
echo "Script ended at: $(date)"

DURATION=$((END_TIME - START_TIME))
echo "Total runtime: ${DURATION} seconds"
