#!/bin/bash
: '
Use this job template to submit job.

{
	"version": "2019-11-09",
	"metadata": {
		"name": "single_node_full_benchmarking_1109",
		"cluster": "rr3",
		"vc": "phillytest"
	},
	"resources": {
		"workers": {
			"type": "skuResource",
			"sku": "G16",
			"count": 1,
			"image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
			"commandLine": "$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/single_node_profile.sh",
			"constraints": [
				{
					"type": "uniqueConstraint",
					"tag": "connectivityDomain"
				}
			],
			"containerArgs": {
				"shmSize": "4G"
			}
		}
	}
}

'


if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi

if [ $PHILLY_VC == "msrhyperscl" ]; then
  export TEST_MODE=True
fi

######################## PT #############################
export PT_SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-pt/
cd /workspace/bert
echo "##################### Profile Section Seperator - PT ###############################"
bash $PT_SCRIPT_PATH"profile-pt.sh" "fixed" || true
echo "Result above is for profile_pt.sh"

echo "##################### Profile Section Seperator 2 - PT ###############################"
bash $PT_SCRIPT_PATH"profile_real_training-pt.sh" "nonfixed" || true
echo "Result above is for profile_real_training-pt.sh"

echo "##################### Profile Section Seperator 3 - PT ###############################"
bash $PT_SCRIPT_PATH"profile_trend-pt.sh" || true
echo "Result above is for profile_trend-pt.sh"

export ORT_SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/
export OMP_NUM_THREADS=1

######################## ORT Master #############################
commitid="a5b3f92b"  #current master
CUSTOM_PARAMS_STRING=""
bash $ORT_SCRIPT_PATH/single_node_profile.sh $commitid CUSTOM_PARAMS_STRING=""

exit 0
