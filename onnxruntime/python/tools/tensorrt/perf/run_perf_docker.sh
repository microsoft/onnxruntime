#!/bin/bash

# Parse Arguments
while getopts d:o:m: parameter
do case "${parameter}"
in 
d) DOCKER_IMAGE=${OPTARG};;
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
esac
done 

# Variables
MACHINE_PERF_DIR=/home/hcsuser/perf/
DOCKER_PERF_DIR=/usr/share/perf/
PERF_SCRIPT=$DOCKER_PERF_DIR'perf.sh'
VOLUME=$MACHINE_PERF_DIR:$DOCKER_PERF_DIR

# Add Remaining Variables
if [ $OPTION == "onnx-zoo-models" ]
then 
    MODEL_PATH=model_list.json
fi 

if [ $OPTION == "many-models" ]
then 
    MODEL_PATH=/usr/share/mount/many-models
    VOLUME=$VOLUME' -v /home/hcsuser/mount/many-models:/usr/share/mount/many-models'
fi 

if [ $OPTION == "partner-models" ]
then 
   MODEL_PATH=partner_model_list.json
fi

sudo docker run --gpus all -v $VOLUME $DOCKER_IMAGE /bin/bash $PERF_SCRIPT -d $DOCKER_PERF_DIR -o $OPTION -m $MODEL_PATH
