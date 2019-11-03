# Versioning

## API
ONNX Runtime follows [Semantic Versioning 2.0](https://semver.org/) for its public API.
Each release has the form MAJOR.MINOR.PATCH. The meanings of MAJOR, MINOR and PATCH are
same as what is described in the semantic versioning doc linked above.

## Current stable release version
The version number of the current stable release can be found
[here](../VERSION_NUMBER).

## Release cadence
See [Release Management](ReleaseManagement.md)

# Compatibility
## ONNX Compatibility
ONNX Runtime supports both backwards and forward compatibility.

### Backwards compatibility
All versions of ONNX Runtime will support ONNX opsets all the way back to (and including) opset version 7.
In other words if an ONNX Runtime release implements ONNX opset ver 9, it'll be able to run all
models that are stamped with ONNX opset verions in the range [7-9].

### Forward compatibility
A release version that supports opset ver 8 will be able to run all models that are stamped with opset ver 9 provided
the model doesn't use ops that were newly introduced in opset ver 9.

### Version matrix
Following table summarizes the relationship between the ONNX Runtime version and the ONNX
opset version implemented in that release. Please note the Backwards and Forward compatiblity notes above.
For more details on ONNX Release versions, see [this page](https://github.com/onnx/onnx/blob/master/docs/Versioning.md).

| ONNX Runtime release version | ONNX release version | ONNX opset version | ONNX ML opset version | Supported ONNX IR version | [WinML compatibility](https://docs.microsoft.com/en-us/windows/ai/windows-ml/)|
|------------------------------|--------------------|--------------------|----------------------|------------------|------------------|
| 1.0.0 | **1.6** down to 1.2 | 11 | 2 | 6 | -- |
| 0.5.0 | **1.5** down to 1.2 | 10 | 1 | 5 | -- |
| 0.4.0 | **1.5** down to 1.2 | 10 | 1 | 5 | -- |
| 0.3.1<br>0.3.0 | **1.4** down to 1.2 | 9 | 1 | 3 | -- |
| 0.2.1<br>0.2.0 | **1.3** down to 1.2 | 8 | 1 | 3 | 1903 (19H1)+ |
| 0.1.5<br>0.1.4 | **1.3** down to 1.2 | 8 | 1 | 3 | 1809 (RS5)+ |


## Tool Compatibility
A variety of tools can be used to create ONNX models. Unless otherwise noted, please use the latest released version of the tools to convert/export the ONNX model. Many tools are backwards compatible and support multiple ONNX versions. Join this with the table above to evaluate ONNX Runtime compatibility.


|Tool|Recommended Version|Supported ONNX version(s)|
|---|---|---|
|[PyTorch](https://pytorch.org/)|[Latest stable](https://pytorch.org/get-started/locally/)|1.2-1.5|
|[ONNXMLTools](https://pypi.org/project/onnxmltools/)<br>CoreML, LightGBM, XGBoost, LibSVM|[Latest stable](https://github.com/onnx/onnxmltools/releases)|1.2-1.5|
|[ONNXMLTools](https://pypi.org/project/onnxmltools/)<br> SparkML|[Latest stable](https://github.com/onnx/onnxmltools/releases)|1.4-1.5|
|[SKLearn-ONNX](https://pypi.org/project/skl2onnx/)|[Latest stable](https://github.com/onnx/sklearn-onnx/releases)|1.2-1.5|
|[Keras-ONNX](https://pypi.org/project/keras2onnx/)|[Latest stable](https://github.com/onnx/keras-onnx/releases)|1.2-1.5|
|[Tensorflow-ONNX](https://pypi.org/project/tf2onnx/)|[Latest stable](https://github.com/onnx/tensorflow-onnx/releases)|1.2-1.5|
|[WinMLTools](https://docs.microsoft.com/en-us/windows/ai/windows-ml/convert-model-winmltools)|[Latest stable](https://pypi.org/project/winmltools/)|1.2-1.4|
|[AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml)|[1.0.39+](https://pypi.org/project/azureml-automl-core)|1.5|
| |[1.0.33](https://pypi.org/project/azureml-automl-core/1.0.33/)|1.4|

