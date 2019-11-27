#!/bin/bash
: '
Use this job template to submit job.

{
	"version": "2019-11-25",
	"metadata": {
		"name": "upper_1125",
		"cluster": "rr3",
		"vc": "phillytest"
	},
	"resources": {
		"workers": {
			"type": "skuResource",
			"sku": "G16",
			"count": 4,
			"image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
			"commandLine": "$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/upper2.sh",
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

export ORT_SCRIPT_PATH=$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/
export OMP_NUM_THREADS=1

cd /code/
commitid="0e9cc6cf"
# clean up files that are generated in earlier runs.
rm binary -rf
rm ort_binary.zip
wget -O ort_binary.zip --no-verbose https://onnxtraining.blob.core.windows.net/philly/binary_${commitid}.tar.gz
tar -xzf ort_binary.zip
mv binary_${commitid} binary
chmod 777 binary -R

cp $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm.onnx /code/binary/bert.onnx 

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>"$PHILLY_CONTAINER_INDEX
nvidia-smi  -q -i 0 -d CLOCK

if [ $PHILLY_CONTAINER_INDEX -ne 0 ]
then
  echo "Not first container, skip by intention"
  sleep infinity
  exit 0
fi

sleep 2m
echo "########## Get GPU Clock Rate/System Load Start ###############"
nvidia-smi  -q -i 0 -d CLOCK

nvidia-smi

echo "########## Get GPU Clock Rate/System Load End ###############"

if [ $PHILLY_VC == "msrhyperscl" ]; then
  export TEST_MODE=True
fi
#export TEST_MODE=True


rm /tmp/results -rf
export CUSTOM_PARAMS_STRING=" --use_nccl=True --use_nccl_tensor_fusion=True --partition_optimizer=True"
bash $ORT_SCRIPT_PATH/profile_upper2.sh "fixed" "philly" 
nvidia-smi  -q -i 0 -d CLOCK
uptime

echo "YET ANOTHER RUN >>>>>>>>>>>>>>>>>>>>>>>>>"

exit 0
