#!/bin/bash

commitid=$1
bash $ORT_SCRIPT_PATH"_basics/_prepare_binary.sh" $commitid
custom_parameters=$2

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi

sleep $CONTAINTER_ZERO_WAIT_TIME # the container wait for 2 minutes, in case other container is slower to downloading binary.
cd /code/
echo "########## Profile Section Seperator - Single GPU Trend  -  Commit "$commitid" ################"
rm /tmp/results -rf
export CUSTOM_PARAM_STRING=$custom_parameters
bash $ORT_SCRIPT_PATH"_basics/_perf_run.sh" \
   1 "1" "1" \
   100 50 \
   "2,4,8,16,32,64,80,128,144,150,154,158,160" "2,4,8,10,12,14,16,20,22,24" \
   "2,4,8,16,32,64,66,68,80"  "2,4,8,10,12" "trend" || true

# bash $ORT_SCRIPT_PATH"_basics/_perf_run.sh" \
#    1 "1" "1" \
#    100 50 \
#    "80" "16" \
#    "16"  "8" "trend" || true
echo "Result above is for trend with accumulation steps=1"
export CUSTOM_PARAM_STRING=""
exit 0
