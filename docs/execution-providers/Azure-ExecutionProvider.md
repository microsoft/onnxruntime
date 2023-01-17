---
title: Azure
description: Instructions to infer an ONNX model remotely with an Azure endpoint
parent: Execution Providers
nav_order: 12
redirect_from: /docs/reference/execution-providers/Cloud-ExecutionProvider
---

# Azure Execution Provider (Preview)

The Azure Execution Provider enables ONNX Runtime to visit an remote endpoint for inferenece, the endpoint must be deployed beforehand.
Azure Execution Provider is in preview stage, all API(s) and usage are subjuct to change.

One use case for Azure Execution Provider is small-big models. E.g. A smaller model deployed on edge device for faster inference,
while a bigger model deployed on Azure for higher precision, with Azure Execution Provider, a switch between the two could be easily achieved.
Note that the two models are expected to have same inputs and outputs.

## Limitations

To consume the endpoint, ONNX Runtime must load the model locally, it could be the same model deployed to Azure, or another model that has exactly same inputs and outputs.
Customer could then use configured run options to switch between local and remote inferencing.
So far, Azure Execution Provider is limited to:
* only support [triton](https://github.com/triton-inference-server) server on [AML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?tabs=python%2Cendpoint).
* only build and run on Windows and Linux.
* available only as python package, but user could also build from source and consume the feature by C/C++ API(s).

## Requirements

For Windows, please install [zlib](https://zlib.net/) and [re2](https://github.com/google/re2), and add their binaries into the system path.
If built from source, zlib and re2 binaries could be easily located with:

```dos
cd <build_output_path>
dir /s zlib1.dll re2.dll
```

For Linux, please make sure openssl is installed.

## Known Issue

For certain ubuntu versions, https call made by azureEP might report error like - "error setting certificate verify location ...".
To silence it, please create file "/etc/pki/tls/certs/ca-bundles.crt" that links to "/etc/ssl/certs/ca-certificates.crt".

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