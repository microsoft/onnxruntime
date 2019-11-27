#!/bin/bash
if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi

commitid=$1
custom_parameters=$2

cd /code/

# clean up files that are generated in earlier runs.
rm binary -rf
rm ort_binary.zip
wget -O ort_binary.zip --no-verbose https://onnxtraining.blob.core.windows.net/philly/binary_${commitid}.tar.gz
tar -xzf ort_binary.zip
mv binary_${commitid} binary
chmod 777 binary -R

export CUSTOM_PARAMS_STRING=$custom_parameters

accumu_steps_type="fixed" # e.g. 1
mpikind="philly"
echo "########## Profile Section Seperator profile_nv_compare_github.sh -  Commit "$commitid" ################"
rm /tmp/results -rf
bash $ORT_SCRIPT_PATH"profile_nv_compare_github.sh" $accumu_steps_type $mpikind || true
echo "Result above is for profile_nv_compare_github.sh with fixed accumulation steps"

echo "########## Profile Section Seperator profile_trend.sh -  Commit "$commitid" ################"
rm /tmp/results -rf
bash $ORT_SCRIPT_PATH"profile_trend.sh" $accumu_steps_type $mpikind || true
echo "Result above is for profile_trend.sh"

#echo "########## Profile Section Seperator profile_real_training.sh -  Commit "$commitid" ################"
#rm /tmp/results -rf
#bash $ORT_SCRIPT_PATH"profile_real_training.sh" "dynamic" $mpikind || true
#echo "Result above is for profile_real_training.sh with dynamic accumulation steps"

export CUSTOM_PARAMS_STRING=""

exit 0
