---
title: Cloud - Azure
description: Instructions to infer an ONNX model remotely with an Azure endpoint
parent: Execution Providers
nav_order: 13
redirect_from: /docs/reference/execution-providers/Azure-ExecutionProvider
---

# Azure Execution Provider (Preview)
{: .no_toc }

The Azure Execution Provider enables ONNX Runtime to invoke a remote Azure endpoint for inference, the endpoint must be deployed or available beforehand.


Since 1.16, below pluggable operators are available from [onnxruntime-extensions](https://github.com/microsoft/onnxruntime-extensions): 

- [OpenAIAudioToText](https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/custom_ops.md#openaiaudiototext)
- [AzureTextToText](https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/custom_ops.md#azuretexttotext)
- [AzureTritonInvoker](https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/custom_ops.md#azuretritoninvoker)

With the operators, Azure Execution Provider supports two mode of usage:

- [Edge and azure side by side](#edge-and-azure-side-by-side)
- [Merge and run the hybrid](#merge-and-run-the-hybrid)

Azure Execution Provider is in preview stage, and all API(s) and usage are subject to change.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install
Since 1.16, Azure Execution Provider is shipped by default in both python and nuget packages.

## Requirements
Since 1.16, all Azure Execution Provider operators are shipped with [onnxruntime-extensions](https://github.com/microsoft/onnxruntime-extensions) (>=v0.9.0) python and nuget packages. Please ensure the installation of correct onnxruntime-extension packages before using Azure Execution Provider.

## Build

For build instructions, please see the [BUILD page](../build/eps.md#azure).

## Usage

### Edge and azure side by side
In this mode, there are two models running simultaneously. The azure model runs asynchronously by [RunAsync](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h#L4341) API, which is also available through [python](https://github.com/microsoft/onnxruntime/blob/873ef8b8f0b09b49c0a7b7e2f03f3639d7418c22/onnxruntime/python/onnxruntime_pybind_state.cc#L1759) and [csharp](https://github.com/microsoft/onnxruntime/blob/873ef8b8f0b09b49c0a7b7e2f03f3639d7418c22/csharp/src/Microsoft.ML.OnnxRuntime/InferenceSession.shared.cs#L1147).
```python
import os
import onnx
from onnx import helper, TensorProto
from onnxruntime_extensions import get_library_path
from onnxruntime import SessionOptions, InferenceSession
import numpy as np
import threading


# Generate the local model by:
# https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/whisper_e2e.py
def get_whiper_tiny():
    return '/onnxruntime-extensions/tutorials/whisper_onnx_tiny_en_fp32_e2e.onnx'


# Generate the azure model
def get_openai_audio_azure_model():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [1])
    model = helper.make_tensor_value_info('model_name', TensorProto.STRING, [1])
    response_format = helper.make_tensor_value_info('response_format', TensorProto.STRING, [-1])
    file = helper.make_tensor_value_info('file', TensorProto.UINT8, [-1])

    transcriptions = helper.make_tensor_value_info('transcriptions', TensorProto.STRING, [-1])

    invoker = helper.make_node('OpenAIAudioToText',
                               ['auth_token', 'model_name', 'response_format', 'file'],
                               ['transcriptions'],
                               domain='com.microsoft.extensions',
                               name='audio_invoker',
                               model_uri='https://api.openai.com/v1/audio/transcriptions',
                               audio_format='wav',
                               verbose=False)

    graph = helper.make_graph([invoker], 'graph', [auth_token, model, response_format, file], [transcriptions])
    model = helper.make_model(graph, ir_version=8,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    model_name = 'openai_whisper_azure.onnx'
    onnx.save(model, model_name)
    return model_name


if __name__ == '__main__':
    sess_opt = SessionOptions()
    sess_opt.register_custom_ops_library(get_library_path())

    azure_model_path = get_openai_audio_azure_model()
    azure_model_sess = InferenceSession(azure_model_path,
        sess_opt, providers=['CPUExecutionProvider', 'AzureExecutionProvider'])  # load AzureEP

    with open('test16.wav', "rb") as _f:  # read raw audio data from a local wav file
        audio_stream = np.asarray(list(_f.read()), dtype=np.uint8)

    azure_model_inputs = {
        "auth_token": np.array([os.getenv('AUDIO', '')]),  # read auth from env variable
        "model_name": np.array(['whisper-1']),
        "response_format":  np.array(['text']),
        "file": audio_stream
    }


    class RunAsyncState:
        def __init__(self):
            self.__event = threading.Event()
            self.__outputs = None
            self.__err = ''

        def fill_outputs(self, outputs, err):
            self.__outputs = outputs
            self.__err = err
            self.__event.set()

        def get_outputs(self):
            if self.__err != '':
                raise Exception(self.__err)
            return self.__outputs;

        def wait(self, sec):
            self.__event.wait(sec)


    def azureRunCallback(outputs: np.ndarray, state: RunAsyncState, err: str) -> None:
        state.fill_outputs(outputs, err)


    run_async_state = RunAsyncState();
    # infer azure model asynchronously
    azure_model_sess.run_async(None, azure_model_inputs, azureRunCallback, run_async_state)

    # in the same time, run the edge
    edge_model_path = get_whiper_tiny()
    edge_model_sess = InferenceSession(edge_model_path,
        sess_opt, providers=['CPUExecutionProvider'])

    edge_model_outputs = edge_model_sess.run(None, {
        'audio_stream': np.expand_dims(audio_stream, 0),
        'max_length': np.asarray([200], dtype=np.int32),
        'min_length': np.asarray([0], dtype=np.int32),
        'num_beams': np.asarray([2], dtype=np.int32),
        'num_return_sequences': np.asarray([1], dtype=np.int32),
        'length_penalty': np.asarray([1.0], dtype=np.float32),
        'repetition_penalty': np.asarray([1.0], dtype=np.float32)
    })

    print("\noutput from whisper tiny: ", edge_model_outputs)
    run_async_state.wait(10)
    print("\nresponse from openAI: ", run_async_state.get_outputs())
    # compare results and pick the better
```

### Merge and run the hybrid

Alternatively, one could also merge local and azure models into a hybrid, then infer as an ordinary onnx model.
Sample scripts could be found [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/AzureEP).

## Current Limitations

* Only builds and run on Windows, Linux and Android.
* For Android, AzureTritonInvoker is not supported.