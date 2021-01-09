#!/bin/bash

# metadata
FAIL_MODEL_FILE=".fail_model_map"
LATENCY_FILE=".latency_map"
METRICS_FILE=".metrics_map"

# symbolic shape info
SYMBOLIC_SHAPE_INFER="symbolic_shape_infer.py"
SYMBOLIC_SHAPE_INFER_LINK="https://raw.githubusercontent.com/microsoft/onnxruntime/master/onnxruntime/python/tools/symbolic_shape_infer.py"

cleanup_files() {
    rm -f $FAIL_MODEL_FILE
    rm -f $LATENCY_FILE
    rm -f $METRICS_FILE
    rm -f $SYMBOLIC_SHAPE_INFER
}

download_symbolic_shape() {
    sudo wget -c $SYMBOLIC_SHAPE_INFER_LINK
}

update_files() {
    cleanup_files
    download_symbolic_shape
}

# many models 
if [ "$1" == "many-models" ]
then
    update_files
    python3 benchmark_wrapper.py -r validate -m /home/hcsuser/mount/many-models -o result/"$1"
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m /home/hcsuser/mount/many-models -o result/"$1"
fi

# ONNX model zoo
if [ "$1" == "onnx-zoo-models" ]
then
    MODEL_LIST="model_list.json"
    update_files
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -o result/"$1"
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m $MODEL_LIST -o result/"$1"
fi

# 1P models 
if [ "$1" == "partner-models" ]
then
    MODEL_LIST="/home/hcsuser/perf/partner_model_list.json"
    update_files
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -o result/"$1"
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m $MODEL_LIST -o result/"$1"
fi

