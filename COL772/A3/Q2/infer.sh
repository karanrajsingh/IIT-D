#!/bin/bash

LANG=$1
TEST_FILE=$2
MODEL_DIR=$3

python infer.py $LANG $TEST_FILE $MODEL_DIR
