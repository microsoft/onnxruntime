#!/bin/bash

# Parse Arguments
while getopts d:o:m:p:e: parameter
do case "${parameter}"
in 
d) DOCKER_IMAGE=${OPTARG};;
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
p) PERF_DIR=${OPTARG};;
e) EP_LIST=${OPTARG};;
esac
done 

# Variables
DOCKER_PERF_DIR=/usr/share/perf/
PERF_SCRIPT=$DOCKER_PERF_DIR'perf.sh'
VOLUME=$PERF_DIR:$DOCKER_PERF_DIR
ONNX_ZOO_VOLUME=' -v /home/hcsuser/perf/models:/usr/share/perf/models'
MANY_MODELS_VOLUME=' -v /home/hcsuser/mount/many-models:/usr/share/mount/many-models'
PARTNER_VOLUME=' -v /home/hcsuser/perf/partner:/usr/share/perf/partner'
WORKSPACE="/"

# Add Remaining Variables
if [ $OPTION == "onnx-zoo-models" ]
then 
    MODEL_PATH='model_list.json'
    VOLUME=$VOLUME$ONNX_ZOO_VOLUME
fi 

if [ $OPTION == "many-models" ]
then 
    MODEL_PATH=/usr/share/mount/many-models
    VOLUME=$VOLUME$MANY_MODELS_VOLUME
fi 

if [ $OPTION == "partner-models" ]
then 
   MODEL_PATH='partner/partner_model_list.json'
   VOLUME=$VOLUME$PARTNER_VOLUME
fi

if [ $OPTION == "selected-models" ]
then	
  VOLUME=$VOLUME$ONNX_ZOO_VOLUME$MANY_MODELS_VOLUME$PARTNER_VOLUME' -v /home/hcsuser/perf/subset_jsons/:/usr/share/perf/subset_jsons'
fi

sudo docker run --gpus all -v $VOLUME $DOCKER_IMAGE /bin/bash $PERF_SCRIPT -d $DOCKER_PERF_DIR -o $OPTION -m $MODEL_PATH -w $WORKSPACE -e "$EP_LIST"
