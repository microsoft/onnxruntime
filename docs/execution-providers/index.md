---
title: Execution Providers
has_children: true
nav_order: 6
redirect_from: /docs/reference/execution-provider
---

# ONNX Runtime Execution Providers
{: .no_toc }

ONNX Runtime works with different hardware acceleration libraries through its extensible **Execution Providers** (EP) framework to optimally execute the ONNX models on the hardware platform. This interface enables flexibility for the AP application developer to deploy their ONNX models in different environments in the cloud and the edge and optimize the execution by taking advantage of the compute capabilities of the platform.

<p align="center"><img width="50%" src="https://www.onnxruntime.ai/images/ONNX_Runtime_EP1.png" alt="Executing ONNX models across different HW environments"/></p>

ONNX Runtime works with the execution provider(s) using the `GetCapability()` interface to allocate specific nodes or sub-graphs for execution by the EP library in supported hardware. The EP libraries that are pre-installed in the execution environment process and execute the ONNX sub-graph on the hardware. This architecture abstracts out the details of the hardware specific libraries that are essential to optimize the execution of deep neural networks across hardware platforms like CPU, GPU, FPGA or specialized NPUs.

<p align="center"><img width="50%" src="https://www.onnxruntime.ai/images/ONNX_Runtime_EP3.png" alt="ONNX Runtime GetCapability()"/></p>

ONNX Runtime supports many different execution providers today. Some of the EPs are in production for live service, while others are released in preview to enable developers to develop and customize their application using the different options.


## Summary of supported Execution Providers 

|CPU|GPU|IoT/Edge/Mobile|Other|
---|---|---|---
|Default CPU|[NVIDIA CUDA](../execution-providers/CUDA-ExecutionProvider.md)|[Intel OpenVINO](../execution-providers/OpenVINO-ExecutionProvider.md)|[Rockchip NPU](../execution-providers/community-maintained/RKNPU-ExecutionProvider.md) (*preview*)|
|[Intel DNNL](../execution-providers/oneDNN-ExecutionProvider.md)|[NVIDIA TensorRT](../execution-providers/TensorRT-ExecutionProvider.md)|[Arm Compute Library](../execution-providers/community-maintained/ACL-ExecutionProvider.md) (*preview*)|[AMD Vitis-AI](../execution-providers/Vitis-AI-ExecutionProvider.md)|
|[TVM](../execution-providers/community-maintained/TVM-ExecutionProvider.md) (*preview*)|[DirectML](../execution-providers/DirectML-ExecutionProvider.md)|[Android Neural Networks API](../execution-providers/NNAPI-ExecutionProvider.md)|[Huawei CANN](../execution-providers/community-maintained/CANN-ExecutionProvider.md) (*preview*)|
|[Intel OpenVINO](../execution-providers/OpenVINO-ExecutionProvider.md)|[AMD MIGraphX](../execution-providers/MIGraphX-ExecutionProvider.md)|[Arm NN](../execution-providers/community-maintained/ArmNN-ExecutionProvider.md) (*preview*)|[AZURE](../execution-providers/Azure-ExecutionProvider.md) (*preview*)|
|[XNNPACK](../execution-providers/Xnnpack-ExecutionProvider.md)|[Intel OpenVINO](../execution-providers/OpenVINO-ExecutionProvider.md)|[CoreML](../execution-providers/CoreML-ExecutionProvider.md) (*preview*)|
||[AMD ROCm](../execution-providers/ROCm-ExecutionProvider.md)|[TVM](../execution-providers/community-maintained/TVM-ExecutionProvider.md) (*preview*)|
||[TVM](../execution-providers/community-maintained/TVM-ExecutionProvider.md) (*preview*)|[Qualcomm QNN](../execution-providers/QNN-ExecutionProvider.md)|
|||[XNNPACK](../execution-providers/Xnnpack-ExecutionProvider.md)|
|||[AMD Vitis-AI](../execution-providers/Vitis-AI-ExecutionProvider.md)|

## Add an Execution Provider

Developers of specialized HW acceleration solutions can integrate with ONNX Runtime to execute ONNX models on their stack. To create an EP to interface with ONNX Runtime you must first identify a unique name for the EP. See: [Add a new execution provider](add-execution-provider.md) for detailed instructions.

## Build ONNX Runtime package with EPs

The ONNX Runtime package can be built with any combination of the EPs along with the default CPU execution provider. **Note** that if multiple EPs are combined into the same ONNX Runtime package then all the dependent libraries must be present in the execution environment. The steps for producing the ONNX Runtime package with different EPs is documented [here](../build/inferencing.md).

## APIs for Execution Provider

The same ONNX Runtime API is used across all EPs. This provides the consistent interface for applications to run with different HW acceleration platforms. The APIs to set EP options are available across Python, C/C++/C#, Java and node.js.

**Note** we are updating our API support to get parity across all language binding and will update specifics here.

    `get_providers`: Return list of registered execution providers.
    `get_provider_options`: Return the registered execution providers' configurations.
    `set_providers`: Register the given list of execution providers. The underlying session is re-created. 
        The list of providers is ordered by Priority. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.

## Use Execution Providers

``` python
import onnxruntime as rt

#define the priority order for the execution providers
# prefer CUDA Execution Provider over CPU Execution Provider
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# initialize the model.onnx
sess = rt.InferenceSession("model.onnx", providers=EP_list)

# get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
output_name = sess.get_outputs()[0].name

# get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
input_name = sess.get_inputs()[0].name

# inference run using image_data as the input to the model 
detections = sess.run([output_name], {input_name: image_data})[0]

print("Output shape:", detections.shape)

# Process the image to mark the inference points 
image = post.image_postprocess(original_image, input_size, detections)
image = Image.fromarray(image)
image.save("kite-with-objects.jpg")

# Update EP priority to only CPUExecutionProvider
sess.set_providers(['CPUExecutionProvider'])

cpu_detection = sess.run(...)

```


