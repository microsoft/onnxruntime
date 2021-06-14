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
DOCKER_PERF_DIR='/perf/'
HOME_PERF_DIR='/home/hcsuser/perf/'
WORKSPACE='/'

# Volumes
VOLUME=$PERF_DIR:$DOCKER_PERF_DIR

if [ $OPTION == "many-models" ]
then 
    MANY_MODELS_VOLUME=' -v /home/hcsuser/mount/many-models:/mount/many-models'
    VOLUME=$VOLUME$MANY_MODELS_VOLUME
else 
    ADD_VOLUME=' -v '$HOME_PERF_DIR$OPTION':'$DOCKER_PERF_DIR$OPTION
    VOLUME=$VOLUME$ADD_VOLUME
fi 

MODEL_PATH=$WORKSPACE$MODEL_PATH
sudo docker run --gpus all -v $VOLUME $DOCKER_IMAGE /bin/bash $DOCKER_PERF_DIR'perf.sh' -d $DOCKER_PERF_DIR -o $OPTION -m $MODEL_PATH -w $WORKSPACE -e "$EP_LIST"
