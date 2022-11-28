#!/bin/bash

# get bert model
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased --sequence_length 384 --batch_sizes 128 --provider=cuda -p fp16

# process shape
python process_shape.py
