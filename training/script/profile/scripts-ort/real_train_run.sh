#!/bin/bash

# Entry to do real phase1 + phase 2 training.

commitid=$1
custom_parameters=$2
bash $ORT_SCRIPT_PATH"_basics/_prepare_binary.sh" $commitid

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi

sleep $CONTAINTER_ZERO_WAIT_TIME # the first container wait for 2 minutes, in case other container is slower to downloading binary.

cd /code/
echo "########## Profile Section Seperator Real Train 16 GPUs - 128-16 -  Commit "$commitid" ################"
rm /tmp/results -rf
export CUSTOM_PARAM_STRING=$custom_parameters
bash $ORT_SCRIPT_PATH"_basics/_real_train.sh" 16 128 16 400 100 16384 16384 || true
# bash $ORT_SCRIPT_PATH"_basics/_real_train.sh" 16 128 16 2 2 16384 16384 || true
echo "Result above is for _real_train.sh 128-16"
export CUSTOM_PARAM_STRING=""
exit 0
