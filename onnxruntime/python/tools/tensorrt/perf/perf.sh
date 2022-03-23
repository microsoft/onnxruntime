#!/bin/bash

while getopts d:o:m:e:a: parameter
do case "${parameter}"
in
d) PERF_DIR=${OPTARG};;
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
e) EP_LIST=${OPTARG};;
a) OPTIONAL_ARGS=${OPTARG};;
esac
done 

# add ep list
RUN_EPS=""
if [ ! -z "$EP_LIST" ]
then 
    RUN_EPS=" -e $EP_LIST"
    OPTIONAL_ARGS=$OPTIONAL_ARGS$RUN_EPS
fi

# change dir if docker
if [ ! -z $PERF_DIR ]
then 
    echo 'changing to '$PERF_DIR
    cd $PERF_DIR 
fi 

# metadata
FAIL_MODEL_FILE=".fail_model_map"
LATENCY_FILE=".latency_map"
METRICS_FILE=".metrics_map"
SESSION_FILE=".session_map"

# files to download info
FLOAT_16="float16.py"
FLOAT_16_LINK="https://raw.githubusercontent.com/microsoft/onnxconverter-common/master/onnxconverter_common/float16.py"

cleanup_files() {
    rm -f $FAIL_MODEL_FILE
    rm -f $LATENCY_FILE
    rm -f $METRICS_FILE
    rm -f $SESSION_FILE
    rm -f $FLOAT_16
    rm -rf result/$OPTION
}

download_files() {
    wget --no-check-certificate -c $FLOAT_16_LINK 
}

setup() {
    cleanup_files
    download_files
}

setup
python3 benchmark_wrapper.py -r validate -m $MODEL_PATH -o result/$OPTION $OPTIONAL_ARGS 
python3 benchmark_wrapper.py -r benchmark -t 1200 -m $MODEL_PATH -o result/$OPTION $OPTIONAL_ARGS
