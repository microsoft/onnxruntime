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
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-ort/profile_nv_compare_github.sh \"fixed\" \"philly\"",
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
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-ort/profile_upper.sh \"fixed\"",
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
                "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1",
                "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-ort/profile_trend.sh \"fixed\"",
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

## Distributed Bert Pre-training Phase1 + Phase2

{
    "version": "2019-11-02",
    "metadata": {
        "name": "ort_4g_scale_train_phase1_and_phase2",
        "cluster": "rr3",
        "vc": "msrhyperscl"
    },
    "resources": {
        "workers": {
            "type": "skuResource",
            "sku": "G4",
            "count": 1,
            "image": "phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:latest",
            "commandLine": "$PHILLY_DATA_DIRECTORY/msrhyperscl/pengwa/profile/scripts-ort/train_scale_run.sh \"msrhyperscl\" 4",
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
                "key": "[Blob Key]"
            }
        }
    }
}