#!/bin/bash

commitid=$1
bash $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/_basics/_env.sh
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
echo "########## Profile Section Seperator - Validation  -  Commit "$commitid" ################"
rm /tmp/results -rf
export CUSTOM_PARAM_STRING=$custom_parameters
bash $ORT_SCRIPT_PATH"_basics/_perf_run.sh" \
   1 "1,4,8,16" "1,4,8,16" \
   200 200 \
   "2,4,8,16" "2,4,8,16" \
   "2,32,64"  "4,8" "validation" || true
# bash $ORT_SCRIPT_PATH"_basics/_perf_run.sh" \
#    1 "16" "16" \
#    10 10 \
#    "8" "16" \
#    "32"  "8" "validation" || true
echo "Result above is for validation_run.sh with accumulation steps=1"
export CUSTOM_PARAM_STRING=""
exit 0