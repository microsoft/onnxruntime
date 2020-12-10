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
* [Huggingface](#huggingface)

***
# General

## Python
**Inference only**
* [Basic](https://microsoft.github.io/onnxruntime/python/tutorial.html)
* [Resnet50](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb)
* [ONNX-Ecosystem Docker image samples](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem/inference_demos)
* [ONNX Runtime Server: SSD Single Shot MultiBox Detector](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb)
* [NUPHAR EP samples](../docs/python/notebooks/onnxruntime-nuphar-tutorial.ipynb)

**Inference with model conversion**
* [SKL tutorials](http://onnx.ai/sklearn-onnx/index_tutorial.html)
* [Keras - Basic](https://microsoft.github.io/onnxruntime/python/auto_examples/plot_dl_keras.html#sphx-glr-auto-examples-plot-dl-keras-py)
* [SSD Mobilenet (Tensorflow)](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb)
* [BERT-SQuAD (PyTorch) on CPU](../onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb)
* [BERT-SQuAD (PyTorch) on GPU](../onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb)
* [BERT-SQuAD (Keras)](../onnxruntime/python/tools/transformers/notebooks/Tensorflow_Keras_Bert-Squad_OnnxRuntime_CPU.ipynb)
* [BERT-SQuAD (Tensorflow)](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb)
* [GPT2 (PyTorch)](../onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb)
* [EfficientDet (Tensorflow)](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientdet.ipynb)
* [EfficientNet-Edge (Tensorflow)](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientnet-edge.ipynb)
* [EfficientNet-Lite (Tensorflow)](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientnet-lite.ipynb)
* [EfficientNet(Keras)](https://github.com/onnx/keras-onnx/blob/master/tutorial/TensorFlow_Keras_EfficientNet.ipynb)
* [MNIST (Keras)](https://github.com/onnx/keras-onnx/blob/master/tutorial/TensorFlow_Keras_MNIST.ipynb)

**Quantization**
* [BERT Quantization on CPU](../onnxruntime/python/tools/quantization/notebooks/Bert-GLUE_OnnxRuntime_quantization.ipynb)

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

* [Inference with Nodejs](./nodejs)

---
# Integrations

## Azure Machine Learning

**Inference and deploy through AzureML**

*For aditional information on training in AzureML, please see [AzureML Training Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training)*
* Inferencing on **CPU** using [ONNX Model Zoo](https://github.com/onnx/models) models: 
  * [Facial Expression Recognition](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-facial-expression-recognition-deploy.ipynb) 
  * [MNIST Handwritten Digits](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-mnist-deploy.ipynb)
  * [Resnet50 Image Classification](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb)
* Inferencing on **CPU** with **PyTorch** model training:
  * [MNIST](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-train-pytorch-aml-deploy-mnist.ipynb)
  * [BERT](../onnxruntime/python/tools/transformers/notebooks/Inference_Bert_with_OnnxRuntime_on_AzureML.ipynb)
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
  
## Huggingface
[Export Tranformer models](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb)
