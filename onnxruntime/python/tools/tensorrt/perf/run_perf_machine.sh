#!/bin/bash

# Parse Arguments
while getopts o:m:e:p: parameter
do case "${parameter}"
in 
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
e) EP_LIST=${OPTARG};;
p) PERF_DIR=${OPTARG};;
esac
done 

# Variables
WORKSPACE=/home/hcsuser/
HOME_PERF_DIR=/home/hcsuser/perf

# Select models to be tested or run selected-models 
if [ $OPTION == "onnx-zoo-models" ]
then 
    MODEL_PATH='model_list.json'
fi 

if [ $OPTION == "many-models" ]
then 
    MODEL_PATH=/home/hcsuser/mount/many-models
fi 

if [ $OPTION == "partner-models" ]
then 
    MODEL_PATH='/home/hcsuser/perf/partner/partner_model_list.json'
fi

./perf.sh -d $HOME_PERF_DIR -o $OPTION -m $MODEL_PATH -w $WORKSPACE -e "$EP_LIST" -v "machine"
