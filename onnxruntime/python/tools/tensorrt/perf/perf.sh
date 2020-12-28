#!/bin/bash

# metadata
FAIL_MODEL_FILE=".fail_model_map"
LATENCY_FILE=".latency_map"
METRICS_FILE=".metrics_map"

# files to download info
SYMBOLIC_SHAPE_INFER="symbolic_shape_infer.py"
SYMBOLIC_SHAPE_INFER_LINK="https://raw.githubusercontent.com/microsoft/onnxruntime/master/onnxruntime/python/tools/symbolic_shape_infer.py"
FLOAT_16="float16.py"
FLOAT_16_LINK="https://raw.githubusercontent.com/microsoft/onnxconverter-common/master/onnxconverter_common/float16.py"

# root working directory 
DEFAULT_DIR="./"

cleanup_files() {
    rm -f $FAIL_MODEL_FILE
    rm -f $LATENCY_FILE
    rm -f $METRICS_FILE
    rm -f $SYMBOLIC_SHAPE_INFER
    rm -f $FLOAT_16
}

download_files() {
    sudo wget -c $SYMBOLIC_SHAPE_INFER_LINK
    sudo wget -c $FLOAT_16_LINK
}

update_files() {
    cleanup_files
    download_files
}

# many models 
if [ "$1" == "many-models" ]
then
    update_files
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r validate -m /home/hcsuser/mount/many-models -o result/"$1"
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r benchmark -i random -t 10 -m /home/hcsuser/mount/many-models -o result/"$1"
fi

# ONNX model zoo
if [ "$1" == "onnx-zoo-models" ]
then
    MODEL_LIST="model_list.json"
    update_files
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r validate -m $MODEL_LIST -o result/"$1"
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r benchmark -i random -t 10 -m $MODEL_LIST -o result/"$1"
fi

# 1P models 
if [ "$1" == "partner-models" ]
then
    MODEL_LIST="partner_model_list.json"
    update_files
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r validate -m $MODEL_LIST -o result/"$1"
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r benchmark -i random -t 10 -m $MODEL_LIST -o result/"$1"
fi

# Test models 
if [ "$1" == "selected-models" ]
then
    MODEL_LIST="selected_models.json"
    update_files
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r validate -m $MODEL_LIST -o result/"$1"
    python3 benchmark_wrapper.py -d $DEFAULT_DIR -r benchmark -i random -t 1 -m $MODEL_LIST -o result/"$1"
fi
