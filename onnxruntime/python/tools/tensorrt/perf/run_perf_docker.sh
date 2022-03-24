#!/bin/bash

# Parse Arguments
while getopts d:o:m:p:e:v:a: parameter
do case "${parameter}"
in 
d) DOCKER_IMAGE=${OPTARG};;
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
p) PERF_DIR=${OPTARG};;
e) EP_LIST=${OPTARG};;
v) MODEL_VOLUME=${OPTARG};;
a) PERF_ARGUMENTS=${OPTARG};;
esac
done 

# Variables
DOCKER_PERF_DIR='/perf/'

docker run --gpus all -v $PERF_DIR:$DOCKER_PERF_DIR -v $MODEL_VOLUME/$OPTION:$DOCKER_PERF_DIR$OPTION $DOCKER_IMAGE /bin/bash $DOCKER_PERF_DIR'perf.sh' -d $DOCKER_PERF_DIR -o $OPTION -m $MODEL_PATH -e "$EP_LIST" "$PERF_ARGUMENTS"
