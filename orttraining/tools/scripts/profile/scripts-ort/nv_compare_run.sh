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
echo "########## Profile Section Seperator - NV Compare -  Commit "$commitid" ################"
rm /tmp/results -rf
export CUSTOM_PARAM_STRING=$custom_parameters
bash $ORT_SCRIPT_PATH"_basics/_perf_run.sh" \
   1 "1,16" "1,16" \
   600 400 \
   "64,160" "8,24" \
   "32,80"  "4,12" "nv_compare" || true

# bash $ORT_SCRIPT_PATH"_basics/_perf_run.sh" \
#    1 "1" "1" \
#    10 10 \
#    "64" "8" \
#    "32" "4" "nv_compare" || true
echo "Result above is for nv_quick_compare.sh with accumulation steps=1"
export CUSTOM_PARAM_STRING=""
exit 0