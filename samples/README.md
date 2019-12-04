# ONNX Runtime Samples and Tutorials

Here you will find various samples, tutorials, and reference implementations for using ONNX Runtime. 
For a list of available dockerfiles and published images to help with getting started, see [this page](../dockerfiles/README.md).

* [Python](#Python)
* [C#](#C)
* [C/C++](#CC)
***
 
## Python
**Inference only**
* [Basic Model Inferencing (single node Sigmoid) on CPU](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/simple_onnxruntime_inference.ipynb)
* [Model Inferencing (Resnet50) on CPU](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb)
* [Model Inferencing on CPU](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem/inference_demos) using [ONNX-Ecosystem Docker image](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem)
* [Model Inferencing on CPU using ONNX Runtime Server (SSD Single Shot MultiBox Detector)](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxRuntimeServerSSDModel.ipynb)
* [Model Inferencing using NUPHAR Execution Provider](../docs/python/notebooks/onnxruntime-nuphar-tutorial.ipynb)

**Inference with model conversion**
* [SKL Pipeline: Train, Convert, and Inference](https://microsoft.github.io/onnxruntime/tutorial.html)
* [Keras: Convert and Inference](https://microsoft.github.io/onnxruntime/auto_examples/plot_dl_keras.html#sphx-glr-auto-examples-plot-dl-keras-py)

**Inference and deploy through AzureML**
* Inferencing on CPU using [ONNX Model Zoo](https://github.com/onnx/models) models: 
  * [Facial Expression Recognition](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-facial-expression-recognition-deploy.ipynb) 
  * [MNIST Handwritten Digits](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-mnist-deploy.ipynb)
  * [Resnet50 Image Classification](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb)
* Inferencing on CPU with model conversion step for existing models:
  * [TinyYolo](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb)
* Inferencing on CPU with PyTorch model training:
  * [MNIST](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-train-pytorch-aml-deploy-mnist.ipynb)
  
  *For aditional information on training in AzureML, please see [AzureML Training Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training)*

* Inferencing on GPU with TensorRT Execution Provider (AKS)
  * [FER+](../docs/python/notebooks/onnx-inference-byoc-gpu-cpu-aks.ipynb)

**Inference and Deploy wtih Azure IoT Edge**
  * [Intel OpenVINO](http://aka.ms/onnxruntime-openvino)
  * [NVIDIA TensorRT on Jetson Nano (ARM64)](http://aka.ms/onnxruntime-arm64)
  * [ONNX Runtime with Azure ML](https://github.com/Azure-Samples/onnxruntime-iot-edge/blob/master/AzureML-OpenVINO/README.md)

**Other**
* [Running ONNX model tests](./docs/Model_Test.md)
* [Common Errors with explanations](https://microsoft.github.io/onnxruntime/auto_examples/plot_common_errors.html#sphx-glr-auto-examples-plot-common-errors-py)

## C#
* [Inferencing Tutorial](../docs/CSharp_API.md#getting-started)

## C/C++
* [C - Inferencing (SqueezeNet)](../csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp)
* [C++ - Inferencing (SqueezeNet)](../csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp)
* [C++ - Inferencing (MNIST)](../samples/c_cxx/MNIST)
