#!/bin/bash
if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi

######################## PT #############################
export PT_SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-pt/
cd /workspace/bert
echo "################### Profile Section Seperator 1 Start - PT Without Accumulation ########################"
bash $PT_SCRIPT_PATH"profile-pt.sh" "fixed" || true
echo "################### Profile Section Seperator 1 End - PT Without Accumulation ##########################"

#echo "################### Profile Section Seperator 2 Start - PT With Accumulation ###########################"
#bash $PT_SCRIPT_PATH"profile_real_training-pt.sh" "nonfixed" || true
#echo "################### Profile Section Seperator 2 End - PT With Accumulation #############################"

echo "################### Profile Section Seperator 3 Start - PT ###############################"
bash $PT_SCRIPT_PATH"profile_trend-pt.sh" || true
echo "################### Profile Section Seperator 3 End - PT   ###############################"

exit 0
