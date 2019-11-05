## Compare ORT with NV PT Benchmark

Submit bash job with following job template

### ORT Job

    {
        "version": "2019-10-30",
        "metadata": {
            "name": "ort-compare_with_nv",
            "cluster": "rr3",
            "vc": "msrhyperscl"
        },
        "resources": {
            "workers": {
                "type": "skuResource",
                "sku": "G4",
                "count": 1,
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:latest",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-ort/profile_nv_compare.sh \"msrhyperscl\" \"fixed\" \"philly\"",
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
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch-nvidia-bert-1907-py3",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-pt/profile-pt.sh \"msrhyperscl\"",
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

## Explore Maximum Batch Size

Submit bash job with following job template

### ORT Job

    {
        "version": "2019-10-30",
        "metadata": {
            "name": "nv-single-g-upper-exploration",
            "cluster": "rr3",
            "vc": "msrhyperscl"
        },
        "resources": {
            "workers": {
                "type": "skuResource",
                "sku": "G4",
                "count": 1,
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:latest",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-ort/profile_upper.sh \"msrhyperscl\" \"fixed\"",
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



### NV PT Job

TODO if needed later

## Comprehensive Batch Size Run

Submit bash job with following job template

### ORT Job

    {
        "version": "2019-10-30",
        "metadata": {
            "name": "ort-comprehensive_run",
            "cluster": "rr3",
            "vc": "msrhyperscl"
        },
        "resources": {
            "workers": {
                "type": "skuResource",
                "sku": "G4",
                "count": 1,
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:latest",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-ort/profile_trend.sh \"msrhyperscl\" \"fixed\"",
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
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch-nvidia-bert-1907-py3",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-pt/profile_trend-pt.sh \"msrhyperscl\"",
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