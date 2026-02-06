# Versioning

## API
ONNX Runtime follows [Semantic Versioning 2.0](https://semver.org/) for its public API.
Each release has the form MAJOR.MINOR.PATCH, adhering to the definitions from the linked semantic versioning doc.

## Current stable release version
The version number of the current stable release can be found
[here](../VERSION_NUMBER).

## Release cadence
See [Release Management](ReleaseManagement.md)

## Updating the Version for a Release

When preparing a release, follow these steps to update the version number across the codebase. This applies both when creating an initial release branch (updating `main`) and when preparing patch releases on release branches:

### Prerequisites
- Node.js (check [js/.nvmrc](../js/.nvmrc) for the required version)
- npm (comes with Node.js)
- Python 3

Verify your setup:
```bash
node --version   # Should match the version in js/.nvmrc
npm --version    # Should be v8.0 or newer
```

### Steps

1. **Update the VERSION_NUMBER file**

   Edit [VERSION_NUMBER](../VERSION_NUMBER) in the repository root to reflect the new version (e.g., `1.23.3`).

2. **Run the version update script**

   From the repository root, run:
   ```bash
   python tools/python/update_version.py
   ```

   This script automatically updates version numbers in:
   - `docs/Versioning.md` - Adds a new row to the version table
   - `docs/python/README.rst` - Adds release notes entry
   - `onnxruntime/__init__.py` - Python package version
   - `js/` packages - All NPM package versions and lock files

3. **Update the C API static_assert (Manual Step)**

   The script does **not** update the version check in the C API. You must manually update the `static_assert` in [onnxruntime/core/session/onnxruntime_c_api.cc](../onnxruntime/core/session/onnxruntime_c_api.cc).

   Search for `static_assert(std::string_view(ORT_VERSION)` and update the version string:
   ```cpp
   static_assert(std::string_view(ORT_VERSION) == "X.Y.Z",
                 "ORT_Version change detected, please follow below steps to ensure OrtApi is updated properly");
   ```

   Replace `X.Y.Z` with your new version number. The comments following this assert explain additional steps if new APIs were added to this release.

4. **Review all changes**

   Review all modified files. Verify:
   - Version numbers are correct in all updated files
   - The release notes URL format is correct (e.g., `https://github.com/Microsoft/onnxruntime/releases/tag/vX.Y.Z`)

5. **Commit and create PR**

   Commit all changes and create a PR targeting `main` or a release branch as appropriate.

### Notes

- The version table in this file and the ONNX opset compatibility information on [onnxruntime.ai](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support) are the canonical sources for version compatibility information.
- For ONNX version/opset/IR reference numbers, see the [ONNX Versioning documentation](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions).

# Compatibility

## Backwards compatibility
All versions of ONNX Runtime will support ONNX opsets all the way back to (and including) opset version 7.
In other words, if an ONNX Runtime release implements ONNX opset ver 9, it'll be able to run all
models that are stamped with ONNX opset versions in the range [7-9].


### Version matrix
The [table](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support) summarizes the relationship between the ONNX Runtime version and the ONNX opset version implemented in that release.
Please note the backward compatibility notes above.
For more details on ONNX Release versions, see [this page](https://github.com/onnx/onnx/blob/main/docs/Versioning.md).

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
|[Paddle2ONNX](https://pypi.org/project/paddle2onnx/)| [Latest stable](https://github.com/PaddlePaddle/Paddle2ONNX/releases) | 1.6-1.9 |
|[AutoML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml)|[1.0.39+](https://pypi.org/project/azureml-automl-core)|1.5|
| |[1.0.33](https://pypi.org/project/azureml-automl-core/1.0.33/)|1.4|
