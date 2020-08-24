# ONNX Runtime Samples and Tutorials

Here you will find various samples, tutorials, and reference implementations for using ONNX Runtime. 
For a list of available dockerfiles and published images to help with getting started, see [this page](../dockerfiles/README.md).

**General**
* [Python](#Python)
* [C#](#C)
* [C/C++](#CC)
* [Java](#Java)
* [Node.js](#Nodejs)

**Integrations**
* [Azure Machine Learning](#azure-machine-learning)
* [Azure IoT Edge](#azure-iot-edge)
* [Azure Media Services](#azure-media-services)
* [Azure SQL Edge and Managed Instance](#azure-sql)
* [Windows Machine Learning](#windows-machine-learning)
* [ML.NET](#mlnet)

***
 
## Python
**Inference only**
* [CPU: Basic](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/simple_onnxruntime_inference.ipynb)
* [CPU: Resnet50](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb)
* [ONNX-Ecosystem Docker image](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem/inference_demos)
* [ONNX Runtime Server: SSD Single Shot MultiBox Detector](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb)
* [NUPHAR EP samples](../docs/python/notebooks/onnxruntime-nuphar-tutorial.ipynb)

**Inference with model conversion**
* [SKL Pipeline: Train, Convert, and Inference](https://microsoft.github.io/onnxruntime/python/tutorial.html)
* [Keras: Convert and Inference](https://microsoft.github.io/onnxruntime/python/auto_examples/plot_dl_keras.html#sphx-glr-auto-examples-plot-dl-keras-py)

**Other**
* [Running ONNX model tests](../docs/Model_Test.md)
* [Common Errors with explanations](https://microsoft.github.io/onnxruntime/python/auto_examples/plot_common_errors.html#sphx-glr-auto-examples-plot-common-errors-py)

## C#
* [Inference Tutorial](../docs/CSharp_API.md#getting-started)
* [ResNet50 v2 Tutorial](../csharp/sample/Microsoft.ML.OnnxRuntime.ResNet50v2Sample)
* [Faster R-CNN Tutorial](../csharp/sample/Microsoft.ML.OnnxRuntime.FasterRcnnSample)

## C/C++
* [C: SqueezeNet](../csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp)
* [C++: model-explorer](./c_cxx/model-explorer) - single and batch processing
* [C++: SqueezeNet](../csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp)
* [C++: MNIST](./c_cxx/MNIST)

## Java
* [Inference Tutorial](../docs/Java_API.md#getting-started)
* [MNIST inference](../java/src/test/java/sample/ScoreMNIST.java)

## Node.js

### Samples
In each sample's implementation subdirectory, run
```
npm install
node ./
```
* [Basic Usage](./nodejs/01_basic-usage/) - a demonstration of basic usage of ONNX Runtime Node.js binding.

* [Create Tensor](./nodejs/02_create-tensor/) - a demonstration of basic usage of creating tensors.
<!--
* [Create Tensor (Advanced)](./nodejs/03_create-tensor-advanced/) - a demonstration of advanced usage of creating tensors.
-->

* [Create InferenceSession](./nodejs/04_create-inference-session/) - shows how to create `InferenceSession` in different ways.


---

## Azure Machine Learning

**Inference and deploy through AzureML**

*For aditional information on training in AzureML, please see [AzureML Training Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training)*
* Inferencing on **CPU** using [ONNX Model Zoo](https://github.com/onnx/models) models: 
  * [Facial Expression Recognition](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-facial-expression-recognition-deploy.ipynb) 
  * [MNIST Handwritten Digits](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-mnist-deploy.ipynb)
  * [Resnet50 Image Classification](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb)
* Inferencing on **CPU** with **PyTorch** model training:
  * [MNIST](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-train-pytorch-aml-deploy-mnist.ipynb)
* Inferencing on **CPU** with model conversion for existing (CoreML) model:
  * [TinyYolo](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb)
* Inferencing on **GPU** with **TensorRT** Execution Provider (AKS):
  * [FER+](../docs/python/notebooks/onnx-inference-byoc-gpu-cpu-aks.ipynb)
  
## Azure IoT Edge
**Inference and Deploy with Azure IoT Edge**
  * [Intel OpenVINO](http://aka.ms/onnxruntime-openvino)
  * [NVIDIA TensorRT on Jetson Nano (ARM64)](http://aka.ms/onnxruntime-arm64)
  * [ONNX Runtime with Azure ML](https://github.com/Azure-Samples/onnxruntime-iot-edge/blob/master/AzureML-OpenVINO/README.md)
  
## Azure Media Services
[Video Analysis through Azure Media Services using using Yolov3 to build an IoT Edge module for object detection](https://github.com/Azure/live-video-analytics/tree/master/utilities/video-analysis/yolov3-onnx)
 
## Azure SQL
[Deploy ONNX model in Azure SQL Edge](https://docs.microsoft.com/en-us/azure/azure-sql-edge/deploy-onnx)

## Windows Machine Learning
[Examples of inferencing with ONNX Runtime through Windows Machine Learning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/tools-and-samples#samples)
  
## ML.NET
[Object Detection with ONNX Runtime in ML.NET](https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/object-detection-onnx)
  
  
