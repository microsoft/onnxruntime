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
MODEL_PATH=$WORKSPACE$MODEL_PATH

sudo docker run --gpus all -v $PERF_DIR:$DOCKER_PERF_DIR -v $HOME_PERF_DIR$OPTION:$DOCKER_PERF_DIR$OPTION $DOCKER_IMAGE /bin/bash $DOCKER_PERF_DIR'perf.sh' -d $DOCKER_PERF_DIR -o $OPTION -m $MODEL_PATH -w $WORKSPACE -e "$EP_LIST"
