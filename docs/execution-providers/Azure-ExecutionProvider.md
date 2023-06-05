---
title: Cloud - Azure
description: Instructions to infer an ONNX model remotely with an Azure endpoint
parent: Execution Providers
nav_order: 11
---

# Azure Execution Provider (Preview)
{: .no_toc }


The Azure Execution Provider enables ONNX Runtime to invoke a remote Azure endpoint for inference. The endpoint must be deployed beforehand.
To consume the endpoint, a model with same inputs and outputs must be first loaded locally.

One use case for Azure Execution Provider is for small-big models. E.g. A smaller model can be deployed on edge devices for faster inference,
while a bigger model can be deployed on Azure for higher precision. Using the Azure Execution Provider, switching between the two can be easily achieved (assuming same inputs and outputs). 

Azure Execution Provider is in preview stage, and all API(s) and usage are subject to change.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install
Pre-built Python binaries of ONNX Runtime with Azure EP are published on Pypi: [onnxruntime-azure](https://pypi.org/project/onnxruntime-azure/)

## Requirements

For Linux, please make sure openssl is installed.

## Build

For build instructions, please see the [BUILD page](../build/eps.md#azure).

## Usage

### Python

```python
from onnxruntime import *
import numpy as np
import os

sess_opt = SessionOptions()
sess_opt.add_session_config_entry('azure.endpoint_type', 'triton'); # only support triton server for now
sess_opt.add_session_config_entry('azure.uri', 'https://...')
sess_opt.add_session_config_entry('azure.model_name', 'a_simple_model');
sess_opt.add_session_config_entry('azure.model_version', '1'); # optional, default 1
sess_opt.add_session_config_entry('azure.verbose', 'true'); # optional, default false

sess = InferenceSession('a_simple_model.onnx', sess_opt, providers=['CPUExecutionProvider','azureExecutionProvider'])

run_opt = RunOptions()
run_opt.add_run_config_entry('use_azure', '1') # optional, default '0' to run inference locally.
run_opt.add_run_config_entry('azure.auth_key', '...') # optional, required only when use_azure set to 1

x = np.array([1,2,3,4]).astype(np.float32)
y = np.array([4,3,2,1]).astype(np.float32)

z = sess.run(None, {'X':x, 'Y':y}, run_opt)[0]
```

### Current Limitations

* Only supports [Triton Inference Server](https://github.com/triton-inference-server) on [AML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?tabs=python%2Cendpoint).
* Only builds and run on Windows and Linux.
* Available only as Python package, but can be built from source and used via C/C++ API(s).
* **Known Issue:** For certain ubuntu versions, https call made by AzureEP might report error - "error setting certificate verify location ...".
To silence it, please create file "/etc/pki/tls/certs/ca-bundles.crt" that link to "/etc/ssl/certs/ca-certificates.crt".