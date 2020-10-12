#!/bin/bash

SYMBOLIC_SHAPE_INFER="/home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/symbolic_shape_infer.py"

# metadata
FAIL_MODEL_FILE=".fail_model_map"
LATENCY_FILE=".latency_map"
METRICS_FILE=".metrics_map"

cleanup_metadata() {
    rm -f FAIL_MODEL_FILE
    rm -f LATENCY_FILE
    rm -f METRICS_FILE
}

# many models 
if [ "$1" == "many-models" ]
then
    cleanup_metadata
    python3 benchmark_wrapper.py -r validate -m /home/hcsuser/mount/many-models -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m /home/hcsuser/mount/many-models -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
fi

# ONNX model zoo
if [ "$1" == "onnx-zoo-models" ]
then
    MODEL_LIST="model_list.json"
    cleanup_metadata
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
    python3 benchmark_wrapper.py -r benchmark -t 10 -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
fi

# 1P models 
if [ "$1" == "partner-models" ]
then
    MODEL_LIST="/home/hcsuser/perf/partner_model_list.json"
    cleanup_metadata
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
fi

