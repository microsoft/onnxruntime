#!/bin/bash

while getopts d:o:m: parameter
do case "${parameter}"
in 
d) PERF_DIR=${OPTARG};;
o) OPTION=${OPTARG};;
m) MOUNT_PATH=${OPTARG};;
esac
done 

# metadata
FAIL_MODEL_FILE=".fail_model_map"
LATENCY_FILE=".latency_map"
METRICS_FILE=".metrics_map"

# files to download info
SYMBOLIC_SHAPE_INFER="symbolic_shape_infer.py"
SYMBOLIC_SHAPE_INFER_LINK="https://raw.githubusercontent.com/microsoft/onnxruntime/master/onnxruntime/python/tools/symbolic_shape_infer.py"
FLOAT_16="float16.py"
FLOAT_16_LINK="https://raw.githubusercontent.com/microsoft/onnxconverter-common/master/onnxconverter_common/float16.py"

cleanup_files() {
    rm -f $FAIL_MODEL_FILE
    rm -f $LATENCY_FILE
    rm -f $METRICS_FILE
    rm -f $SYMBOLIC_SHAPE_INFER
    rm -f $FLOAT_16
}

download_files() {
    wget --no-check-certificate -c $SYMBOLIC_SHAPE_INFER_LINK
    wget --no-check-certificate -c $FLOAT_16_LINK 
}

setup() {
    cd $PERF_DIR
    cleanup_files
    download_files
}

# many models 
if [ $OPTION == "many-models" ]
then
    setup
    python3 benchmark_wrapper.py -r validate -m "$3" -o result/"$2"
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m "$3" -o result/"$2"
fi

# ONNX model zoo
if [ $OPTION == "onnx-zoo-models" ]
then
    echo "in onnx-zoo-models"
    MODEL_LIST="model.json"
    setup
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -o result/$OPTION
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m $MODEL_LIST -o result/$OPTION
fi

# 1P models 
if [ "$2" == "partner-models" ]
then
    MODEL_LIST="partner_model_list.json"
    setup
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -o result/"$2"
    python3 benchmark_wrapper.py -r benchmark -i random -t 10 -m $MODEL_LIST -o result/"$2"
fi

# Test models 
if [ "$2" == "selected-models" ]
then
    MODEL_LIST="selected_models.json"
    setup
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -o result/"$2"
    python3 benchmark_wrapper.py -r benchmark -i random -t 1 -m $MODEL_LIST -o result/"$2"
fi
