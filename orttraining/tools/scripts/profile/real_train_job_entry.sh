#!/bin/bash
: '
Use this job template to submit job.

{
    "version": "2019-12-03",
    "metadata": {
        "name": "real_train_job",
        "cluster": "bj1",
        "vc": "phillytest"
    },
    "resources": {
        "workers": {
            "type": "skuResource",
            "sku": "G16",
            "count": 1,
            "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
            "commandLine": "$PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/real_train_job_entry.sh",
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
    },
    "volumes": {
        "myblob": {
            "_comment": "myblobcomment",
            "type": "blobfuseVolume",
            "storageAccount": "orttraining",
            "containerName": "bert",
            "path": "/bert_data"
        }
    },
    "credentials": {
        "storageAccounts": {
            "orttraining": {
                "_comment": "orttrainingcomment",
                "key": ""
            }
        }
    }
}
'
source $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/_basics/_env_variables.sh
cp $PHILLY_SCRIPT_ROOT"bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm.onnx" /code/bert.onnx 
bash $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/_basics/_machine_info.sh

echo "############## Real Training Job with commit : 844d9106 ##################"
commitid="844d9106"
bash $ORT_SCRIPT_PATH/real_train_run.sh $commitid " --use_nccl=True --cuda_mem_limit=30"

bash $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/_basics/_machine_info.sh

exit 0
