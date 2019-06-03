# Versioning

## API
ONNX Runtime follows [Semantic Versioning 2.0](https://semver.org/) for its public API.
Each release has the form MAJOR.MINOR.PATCH. The meanings of MAJOR, MINOR and PATCH are
same as what is described in the semantic versioning doc linked above.

## Current stable release version
The version number of the current stable release can be found
[here](../VERSION_NUMBER)

## Release cadence
See [Release Management](ReleaseManagement.md)

## Compatibility with ONNX opsets
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
opset version implemented in that release.

| ONNX Runtime release version | ONNX opset version <br> implemented in this release | ONNX ML opset version <br> implemented in this release | Supported ONNX IR version |
|------------------------------|--------------------|----------------------|------------------|
| 0.4.0 | 10 | 1 | 5 |
| 0.3.1 | 9 | 1 | 3 |
| 0.3.0 | 9 | 1 | 3 |
| 0.2.1 | 8 | 1 | 3 |
| 0.2.0 | 8 | 1 | 3 |
| 0.1.5 | 8 | 1 | 3 |
| 0.1.4 | 8 | 1 | 3 |
