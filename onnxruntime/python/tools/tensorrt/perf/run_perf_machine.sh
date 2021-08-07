#!/bin/bash

# Parse Arguments
while getopts o:m:e: parameter
do case "${parameter}"
in 
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
e) EP_LIST=${OPTARG};;
esac
done 

# Variables
WORKSPACE=/home/hcsuser/

./perf.sh -o $OPTION -m $WORKSPACE$MODEL_PATH -w $WORKSPACE -e "$EP_LIST" 
