---
title: Cloud
description: Instructions to execute ONNX Runtime with a Cloud endpoint
parent: Execution Providers
nav_order: 12
redirect_from: /docs/reference/execution-providers/Cloud-ExecutionProvider
---

# ROCm Execution Provider
{: .no_toc }

The Cloud Execution Provider enables ONNX Runtime to invoke a cloud endpoint for inferenece.

## Contents
By far Cloud Execution Provider only support Windows and Linux.

* TOC placeholder
{:toc}

## Install
{:toc}

## Requirements
For Windows, please install [zlib](https://zlib.net/) and [re2](https://github.com/google/re2), and add their binareis into system path;
If building from source, zlib and re2 binaries could be simply found under build output folder, which can be located with "dir /s zlib1.dll re2.dll".
For Linux, please make openssl is installed.

## Known issues
For certain ubuntu versions, https call made by CloudEP might report error like - "error setting certificate verify location ...",
please create the link file "/etc/pki/tls/certs/ca-bundles.crt" points to "/etc/ssl/certs/ca-certificates.crt" to silence it.

## Build
For build instructions, please see the [BUILD page](../build/eps.md#Cloud).

## Usage

### Python

```python
from onnxruntime import *
import numpy as np
import os

sess_opt = SessionOptions()
sess_opt.add_session_config_entry('cloud.endpoint_type', 'triton');
sess_opt.add_session_config_entry('cloud.uri', 'https://...')
sess_opt.add_session_config_entry('cloud.model_name', 'model_name');
sess_opt.add_session_config_entry('cloud.model_version', '1'); #optional, default 1
sess_opt.add_session_config_entry('cloud.verbose', 'true'); #optional, default false

sess = InferenceSession('addf.onnx', sess_opt, providers=['CPUExecutionProvider','CloudExecutionProvider'])

run_opt = RunOptions()
run_opt.add_run_config_entry('use_cloud', '1')
run_opt.add_run_config_entry('cloud.auth_key', '...')

x = np.array([1,2,3,4]).astype(np.float32)
y = np.array([4,3,2,1]).astype(np.float32)

z = sess.run(None, {'X':x, 'Y':y}, run_opt)[0]
```

## Performance Tuning
{:toc}

##