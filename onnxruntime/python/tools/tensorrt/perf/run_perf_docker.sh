#!/bin/bash

# Parse Arguments
while getopts d:o:m:p:e:v: parameter
do case "${parameter}"
in 
d) DOCKER_IMAGE=${OPTARG};;
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
p) PERF_DIR=${OPTARG};;
e) EP_LIST=${OPTARG};;
v) MODEL_VOLUME=${OPTARG};;
esac
done 

# Variables
DOCKER_PERF_DIR='/perf/'
HOME_PERF_DIR='/home/hcsuser/perf/'
WORKSPACE='/'
MODEL_PATH=$WORKSPACE$MODEL_PATH

docker run --gpus all -v $PERF_DIR:$DOCKER_PERF_DIR -v $MODEL_VOLUME/$OPTION:/perf/$OPTION $DOCKER_IMAGE /bin/bash $DOCKER_PERF_DIR'perf.sh' -d $DOCKER_PERF_DIR -o $OPTION -m $MODEL_PATH -w $WORKSPACE -e "$EP_LIST"