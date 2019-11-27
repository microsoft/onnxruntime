#!/bin/bash
: '
Use this job template to submit job to validate gains for your commits.

{
	"version": "2019-11-09",
	"metadata": {
		"name": "single_node_validate_1109",
		"cluster": "rr3",
		"vc": "phillytest"
	},
	"resources": {
		"workers": {
			"type": "skuResource",
			"sku": "G16",
			"count": 1,
			"image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
			"commandLine": "$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/validate.sh",
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

echo "########## Get GPU Clock Rate/System Load Start ###############"
nvidia-smi  -q -i 0 -d CLOCK

nvidia-smi

echo "########## Get GPU Clock Rate/System Load End ###############"

if [ $PHILLY_VC == "msrhyperscl" ]; then
  export TEST_MODE=True
fi

#export TEST_MODE=True

######################## ORT #############################
export OMP_NUM_THREADS=1
export ORT_SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/

uptime
nvidia-smi  -q -i 0 -d CLOCK
commitid="a5b3f92b" 
bash $ORT_SCRIPT_PATH/validate.sh $commitid ""

uptime
nvidia-smi  -q -i 0 -d CLOCK
commitid="ff12625d" 
bash $ORT_SCRIPT_PATH/validate.sh $commitid ""

uptime
nvidia-smi  -q -i 0 -d CLOCK
commitid="1a5fe050" 
bash $ORT_SCRIPT_PATH/validate.sh $commitid ""

uptime
nvidia-smi  -q -i 0 -d CLOCK
commitid="19033e1a" 
bash $ORT_SCRIPT_PATH/validate.sh $commitid ""

uptime
nvidia-smi  -q -i 0 -d CLOCK
exit 0
