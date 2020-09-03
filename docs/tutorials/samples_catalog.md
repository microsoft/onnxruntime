---
parent: Tutorials
toc: true
nav_order: 2
---

# Samples catalog
{: .no_toc }

This page catalogs code samples for ONNX Runtime, running locally, and on Azure, both cloud and edge.  

## Python

* [CPU: Basic Inference](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/simple_onnxruntime_inference.ipynb)
* [CPU: Resnet50 Inference](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb)
* [Inference with ONNX-Ecosystem Docker image](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem/inference_demos)
* [ONNX Runtime Server: SSD Single Shot MultiBox Detector](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb)
* [NUPHAR Execution Provider samples](https://github.com/microsoft/onnxruntime/tree/master/docs/python/notebooks/onnxruntime-nuphar-tutorial.ipynb)
* [SKL Pipeline: Train, Convert, and Inference](https://microsoft.github.io/onnxruntime/python/tutorial.html)
* [Keras: Convert and Inference](https://microsoft.github.io/onnxruntime/python/auto_examples/plot_dl_keras.html#sphx-glr-auto-examples-plot-dl-keras-py)
* [Train NVIDIA BERT transformer model](https://github.com/microsoft/onnxruntime-training-examples/tree/master/nvidia-bert)
* [Train HuggingFace GPT-2 model](https://github.com/microsoft/onnxruntime-training-examples/tree/master/huggingface-gpt2)

## C/C++

* [C: SqueezeNet](https://github.com/microsoft/onnxruntime/tree/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp)
* [C++: model-explorer](https://github.com/microsoft/onnxruntime/tree/master/c_cxx/model-explorer) - single and batch processing
* [C++: SqueezeNet](https://github.com/microsoft/onnxruntime/tree/mastercsharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp)

## Java

* [MNIST inference](https://github.com/microsoft/onnxruntime/tree/master/java/src/test/java/sample/ScoreMNIST.java)

## Node.js

* [Inference with Nodejs](https://github.com/microsoft/onnxruntime/tree/master/samples/nodejs)


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
  * [FER+](.https://github.com/microsoft/onnxruntime/tree/master/docs/python/notebooks/onnx-inference-byoc-gpu-cpu-aks.ipynb)
  
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
  
--- 

## TABLE OF CONTENTS
{: .no_toc }

* TOC placeholder
{:toc}
