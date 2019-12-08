## Compare ORT with NV PT Benchmark

Submit bash job with following job template

### NV PT Job

    {
        "version": "2019-10-30",
        "metadata": {
            "name": "pt_nv-baseline",
            "cluster": "rr3",
            "vc": "msrhyperscl"
        },
        "resources": {
            "workers": {
                "type": "skuResource",
                "sku": "G4",
                "count": 1,
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-pt/profile-pt.sh \"fixed\"",
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

Note: 
> G4 means choose the machine having 4 GPUs.

## Comprehensive Batch Size Run

Submit bash job with following job template

### NV PT Job

    {
        "version": "2019-10-30",
        "metadata": {
            "name": "pt_nv-comprehensive_run",
            "cluster": "rr3",
            "vc": "msrhyperscl"
        },
        "resources": {
            "workers": {
                "type": "skuResource",
                "sku": "G4",
                "count": 1,
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-pt/profile_trend-pt.sh",
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


Note: 
> G4 means choose the machine having 4 GPUs.
