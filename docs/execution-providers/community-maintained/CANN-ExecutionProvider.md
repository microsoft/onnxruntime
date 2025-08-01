---
title: Huawei - CANN
description: Instructions to execute ONNX Runtime with the Huawei CANN execution provider
grand_parent: Execution Providers
parent: Community-maintained
nav_order: 7
redirect_from: /docs/reference/execution-providers/CANN-ExecutionProvider
---

# CANN Execution Provider
{: .no_toc }

Huawei Compute Architecture for Neural Networks (CANN) is a heterogeneous computing architecture for AI scenarios and provides multi-layer programming interfaces to help users quickly build AI applications and services based on the Ascend platform.

Using CANN Excution Provider for ONNX Runtime can help you accelerate ONNX models on Huawei Ascend hardware.

The CANN Execution Provider (EP) for ONNX Runtime is developed by Huawei.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Pre-built binaries of ONNX Runtime with CANN EP are published, but only for python currently, please refer to [onnxruntime-cann](https://pypi.org/project/onnxruntime-cann/).

## Requirements

Please reference table below for official CANN packages dependencies for the ONNX Runtime inferencing package.

|ONNX Runtime|CANN|
|---|---|
|v1.20.0|8.2.0|
|v1.21.0|8.2.0|
|v1.22.1|8.2.0|

## Build

For build instructions, please see the [BUILD page](../../build/eps.md#cann).

## Configuration Options

The CANN Execution Provider supports the following configuration options.

### device_id

The device ID.

Default value: 0

### npu_mem_limit

The size limit of the device memory arena in bytes. This size limit is only for the execution provider's arena. The total device memory usage may be higher.

### arena_extend_strategy

The strategy for extending the device memory arena.

Value                   | Description
-|-
kNextPowerOfTwo     | subsequent extensions extend by larger amounts (multiplied by powers of two)
kSameAsRequested    | extend by the requested amount

Default value: kNextPowerOfTwo

### enable_cann_graph

Whether to use the graph inference engine to speed up performance. The recommended setting is true. If false, it will fall back to the single-operator inference engine.

Default value: true

### dump_graphs

Whether to dump the subgraph into onnx format for analysis of subgraph segmentation.

Default value: false

### dump_om_model

Whether to dump the offline model for Ascend AI Processor to an .om file.

Default value: true

### precision_mode

The precision mode of the operator.

Value                   | Description
-|-
force_fp32/cube_fp16in_fp32out | convert to float32 first according to operator implementation
force_fp16 | convert to float16 when float16 and float32 are both supported
allow_fp32_to_fp16 | convert to float16 when float32 is not supported
must_keep_origin_dtype | keep it as it is
allow_mix_precision/allow_mix_precision_fp16 | mix precision mode

Default value: force_fp16

### op_select_impl_mode

Some built-in operators in CANN have high-precision and high-performance implementation.

Value                   | Description
-|-
high_precision | aim for high precision
high_performance | aim for high preformance

Default value: high_performance

### optypelist_for_implmode

Enumerate the list of operators which use the mode specified by the op_select_impl_mode parameter.

The supported operators are as follows:

* Pooling
* SoftmaxV2
* LRN
* ROIAlign

Default value: None

## Performance tuning

### IO Binding

The [I/O Binding feature](../../performance/tune-performance/iobinding.html) should be utilized to avoid overhead resulting from copies on inputs and outputs.

* Python

```python
import numpy as np
import onnxruntime as ort

providers = [
    (
        "CANNExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "npu_mem_limit": 2 * 1024 * 1024 * 1024,
            "enable_cann_graph": True,
        },
    ),
    "CPUExecutionProvider",
]

model_path = '<path to model>'

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.int64)
x_ortvalue = ort.OrtValue.ortvalue_from_numpy(x, "cann", 0)

io_binding = sess.io_binding()
io_binding.bind_ortvalue_input(name="input", ortvalue=x_ortvalue)
io_binding.bind_output("output", "cann")

sess.run_with_iobinding(io_binding)

return io_binding.get_outputs()[0].numpy()
```

* C/C++(future)

## Samples

Currently, users can use C/C++ and Python API on CANN EP.

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

options = ort.SessionOptions()

providers = [
    (
        "CANNExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "npu_mem_limit": 2 * 1024 * 1024 * 1024,
            "op_select_impl_mode": "high_performance",
            "optypelist_for_implmode": "Gelu",
            "enable_cann_graph": True
        },
    ),
    "CPUExecutionProvider",
]

session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
```

### C/C++
Note: This sample shows model inference using [resnet50_Opset16.onnx](https://github.com/onnx/models/tree/main/Computer_Vision/resnet50_Opset16_timm) as an example. 
      You need to modify the model_path, and the input_prepare() and output_postprocess() functions according to your needs.


```c++
#include <iostream>
#include <vector>

#include "onnxruntime_cxx_api.h"

// path of model, Change to user's own model path
const char* model_path = "./onnx/resnet50_Opset16.onnx";

/**
 * @brief Input data preparation provided by user.
 *
 * @param num_input_nodes The number of model input nodes.
 * @return  A collection of input data.
 */
std::vector<std::vector<float>> input_prepare(size_t num_input_nodes) {
  std::vector<std::vector<float>> input_datas;
  input_datas.reserve(num_input_nodes);

  constexpr size_t input_data_size = 3 * 224 * 224;
  std::vector<float> input_data(input_data_size);
  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_data_size; i++)
    input_data[i] = (float)i / (input_data_size + 1);
  input_datas.push_back(input_data);

  return input_datas;
}

/**
 * @brief Model output data processing logic(For User updates).
 *
 * @param output_tensors The results of the model output.
 */
void output_postprocess(std::vector<Ort::Value>& output_tensors) {
  auto floatarr = output_tensors.front().GetTensorMutableData<float>();

  for (int i = 0; i < 5; i++) {
    std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
  }
  
  std::cout << "Done!" << std::endl;
}

/**
 * @brief The main functions for model inference.
 *
 *  The complete model inference process, which generally does not need to be
 * changed here
 */
void inference() {
  const auto& api = Ort::GetApi();
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

  // Enable cann graph in cann provider option.
  OrtCANNProviderOptions* cann_options = nullptr;
  api.CreateCANNProviderOptions(&cann_options);

  // Configurations of EP
  std::vector<const char*> keys{
      "device_id",
      "npu_mem_limit",
      "arena_extend_strategy",
      "enable_cann_graph"};
  std::vector<const char*> values{"0", "4294967296", "kNextPowerOfTwo", "1"};
  api.UpdateCANNProviderOptions(
      cann_options, keys.data(), values.data(), keys.size());

  // Convert to general session options
  Ort::SessionOptions session_options;
  api.SessionOptionsAppendExecutionProvider_CANN(
      static_cast<OrtSessionOptions*>(session_options), cann_options);

  Ort::Session session(env, model_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;

  // Input Process
  const size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names;
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  input_node_names.reserve(num_input_nodes);
  input_names_ptr.reserve(num_input_nodes);
  std::vector<std::vector<int64_t>> input_node_shapes;
  std::cout << num_input_nodes << std::endl;
  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_node_shapes.push_back(tensor_info.GetShape());
  }

  // Output Process
  const size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names;
  std::vector<Ort::AllocatedStringPtr> output_names_ptr;
  output_names_ptr.reserve(num_input_nodes);
  output_node_names.reserve(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    output_node_names.push_back(output_name.get());
    output_names_ptr.push_back(std::move(output_name));
  }

  //  User need to generate input date according to real situation.
  std::vector<std::vector<float>> input_datas = input_prepare(num_input_nodes);

  auto memory_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> input_tensors;
  input_tensors.reserve(num_input_nodes);
  for (size_t i = 0; i < input_node_shapes.size(); i++) {
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_datas[i].data(),
        input_datas[i].size(),
        input_node_shapes[i].data(),
        input_node_shapes[i].size());
    input_tensors.push_back(std::move(input_tensor));
  }

  auto output_tensors = session.Run(
      Ort::RunOptions{nullptr},
      input_node_names.data(),
      input_tensors.data(),
      num_input_nodes,
      output_node_names.data(),
      output_node_names.size());

  // Processing of out_tensor
  output_postprocess(output_tensors);
}

int main(int argc, char* argv[]) {
  inference();
  return 0;
}
```

## Supported ops

Following ops are supported by the CANN Execution Provider in single-operator Inference mode.

|Operator|Note|
|--------|------|
|ai.onnx:Abs||
|ai.onnx:Add||
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:BatchNormalization||
|ai.onnx:Cast||
|ai.onnx:Ceil||
|ai.onnx:Conv|Only 2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:Cos||
|ai.onnx:Div||
|ai.onnx:Dropout||
|ai.onnx:Exp||
|ai.onnx:Erf||
|ai.onnx:Flatten||
|ai.onnx:Floor||
|ai.onnx:Gemm||
|ai.onnx:GlobalAveragePool||
|ai.onnx:GlobalMaxPool||
|ai.onnx:Identity||
|ai.onnx:Log||
|ai.onnx:MatMul||
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:Mul||
|ai.onnx:Neg||
|ai.onnx:Reciprocal||
|ai.onnx:Relu||
|ai.onnx:Reshape||
|ai.onnx:Round||
|ai.onnx:Sin||
|ai.onnx:Sqrt||
|ai.onnx:Sub||
|ai.onnx:Transpose||

## Additional Resources

Additional operator support and performance tuning will be added soon.

* [Ascend](https://www.hiascend.com/en/)
* [CANN](https://www.hiascend.com/en/software/cann)
