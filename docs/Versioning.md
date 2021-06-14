# Versioning

## API
ONNX Runtime follows [Semantic Versioning 2.0](https://semver.org/) for its public API.
Each release has the form MAJOR.MINOR.PATCH, adhering to the definitions from the linked semantic versioning doc.

## Current stable release version
The version number of the current stable release can be found
[here](../VERSION_NUMBER).

## Release cadence
See [Release Management](ReleaseManagement.md)

# Compatibility

## Backwards compatibility
All versions of ONNX Runtime will support ONNX opsets all the way back to (and including) opset version 7.
In other words, if an ONNX Runtime release implements ONNX opset ver 9, it'll be able to run all
models that are stamped with ONNX opset versions in the range [7-9].


### Version matrix
The following table summarizes the relationship between the ONNX Runtime version and the ONNX opset version implemented in that release.
Please note the backward compatibility notes above.
For more details on ONNX Release versions, see [this page](https://github.com/onnx/onnx/blob/master/docs/Versioning.md).

| ONNX Runtime release version | ONNX release version | ONNX opset version | ONNX ML opset version | Supported ONNX IR version | [Windows ML Availability](https://docs.microsoft.com/en-us/windows/ai/windows-ml/release-notes/)|
|------------------------------|--------------------|--------------------|----------------------|------------------|------------------|
| 1.8.1 | **1.8** down to 1.2 | 13 | 2 | 7 | Windows AI 1.8+ |
| 1.8.0 | **1.8** down to 1.2 | 13 | 2 | 7 | Windows AI 1.8+ |
| 1.7.0 | **1.8** down to 1.2 | 13 | 2 | 7 | Windows AI 1.7+ |
| 1.6.0 | **1.8** down to 1.2 | 13 | 2 | 7 | Windows AI 1.6+ |
| 1.5.3 | **1.7** down to 1.2 | 12 | 2 | 7 | Windows AI 1.5+ |
| 1.5.2 | **1.7** down to 1.2 | 12 | 2 | 7 | Windows AI 1.5+ |
| 1.5.1 | **1.7** down to 1.2 | 12 | 2 | 7 | Windows AI 1.5+ |
| 1.4.0 | **1.7** down to 1.2 | 12 | 2 | 7 | Windows AI 1.4+ |
| 1.3.1 | **1.7** down to 1.2 | 12 | 2 | 7 | Windows AI 1.4+ |
| 1.3.0 | **1.7** down to 1.2 | 12 | 2 | 7 | Windows AI 1.3+ |
| 1.2.0<br>1.1.2<br>1.1.1<br>1.1.0 | **1.6** down to 1.2 | 11 | 2 | 6 | Windows AI 1.3+ |
| 1.0.0 | **1.6** down to 1.2 | 11 | 2 | 6 | Windows AI 1.3+ |
| 0.5.0 | **1.5** down to 1.2 | 10 | 1 | 5 | Windows AI 1.3+ |
| 0.4.0 | **1.5** down to 1.2 | 10 | 1 | 5 | Windows AI 1.3+ |
| 0.3.1<br>0.3.0 | **1.4** down to 1.2 | 9 | 1 | 3 | Windows 10 2004+ |
| 0.2.1<br>0.2.0 | **1.3** down to 1.2 | 8 | 1 | 3 | Windows 10 1903+ |
| 0.1.5<br>0.1.4 | **1.3** down to 1.2 | 8 | 1 | 3 | Windows 10 1809+ |


## Tool Compatibility
A variety of tools can be used to create ONNX models. Unless otherwise noted, please use the latest released version of the tools to convert/export the ONNX model. Most tools are backwards compatible and support multiple ONNX versions. Join this with the table above to evaluate ONNX Runtime compatibility.


|Tool|Recommended Version|Supported ONNX version(s)|
|---|---|---|
|[PyTorch](https://pytorch.org/)|[Latest stable](https://pytorch.org/get-started/locally/)|1.2-1.6|
|[ONNXMLTools](https://pypi.org/project/onnxmltools/)<br>CoreML, LightGBM, XGBoost, LibSVM|[Latest stable](https://github.com/onnx/onnxmltools/releases)|1.2-1.6|
|[ONNXMLTools](https://pypi.org/project/onnxmltools/)<br> SparkML|[Latest stable](https://github.com/onnx/onnxmltools/releases)|1.4-1.5|
|[SKLearn-ONNX](https://pypi.org/project/skl2onnx/)|[Latest stable](https://github.com/onnx/sklearn-onnx/releases)|1.2-1.6|
|[Keras-ONNX](https://pypi.org/project/keras2onnx/)|[Latest stable](https://github.com/onnx/keras-onnx/releases)|1.2-1.6|
|[Tensorflow-ONNX](https://pypi.org/project/tf2onnx/)|[Latest stable](https://github.com/onnx/tensorflow-onnx/releases)|1.2-1.6|
|[WinMLTools](https://docs.microsoft.com/en-us/windows/ai/windows-ml/convert-model-winmltools)|[Latest stable](https://pypi.org/project/winmltools/)|1.2-1.6|
|[AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml)|[1.0.39+](https://pypi.org/project/azureml-automl-core)|1.5|
| |[1.0.33](https://pypi.org/project/azureml-automl-core/1.0.33/)|1.4|

