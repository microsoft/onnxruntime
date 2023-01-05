---
title: Cloud
description: Instructions to infer an ONNX model remotely with a cloud endpoint
parent: Execution Providers
nav_order: 12
redirect_from: /docs/reference/execution-providers/Cloud-ExecutionProvider
---

# Cloud Execution Provider

The Cloud Execution Provider enables ONNX Runtime to invoke a cloud endpoint for inferenece, the endpoint must be deployed beforehand.
Cloud Execution Provider is in preview stage, all API(s) and usage are subjuct to change.

## Contents

To consume the endpoint, ONNX Runtime must load the same model locally, customer could then use configured run options to choose to go remote.
By far, Cloud Execution Provider is limited to:
* only support [triton](https://github.com/triton-inference-server) server on [AML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?tabs=python%2Cendpoint).
* only build and run on Windows and Linux.

## Requirements

For Windows, please install [zlib](https://zlib.net/) and [re2](https://github.com/google/re2), and add their binaries into the system path.
If built from source, zlib and re2 binaries could be easily located with:

```dos
cd <build_output_path>
dir /s zlib1.dll re2.dll
```

For Linux, please make sure openssl is installed.

## Known Issue

For certain ubuntu versions, https call made by CloudEP might report error like - "error setting certificate verify location ...".
To silence it, please create file "/etc/pki/tls/certs/ca-bundles.crt" that links to "/etc/ssl/certs/ca-certificates.crt".

## Build

For build instructions, please see the [BUILD page](../build/eps.md#Cloud).

## Usage

### Python

```python
from onnxruntime import *
import numpy as np
import os

sess_opt = SessionOptions()
sess_opt.add_session_config_entry('cloud.endpoint_type', 'triton'); # only support triton server for now
sess_opt.add_session_config_entry('cloud.uri', 'https://...')
sess_opt.add_session_config_entry('cloud.model_name', 'a_simple_model');
sess_opt.add_session_config_entry('cloud.model_version', '1'); # optional, default 1
sess_opt.add_session_config_entry('cloud.verbose', 'true'); # optional, default false

sess = InferenceSession('a_simple_model.onnx', sess_opt, providers=['CPUExecutionProvider','CloudExecutionProvider'])

run_opt = RunOptions()
run_opt.add_run_config_entry('use_cloud', '1') # optional, default '0' to run inference locally.
run_opt.add_run_config_entry('cloud.auth_key', '...') # optional, required only when use_cloud set to 1

x = np.array([1,2,3,4]).astype(np.float32)
y = np.array([4,3,2,1]).astype(np.float32)

z = sess.run(None, {'X':x, 'Y':y}, run_opt)[0]
```