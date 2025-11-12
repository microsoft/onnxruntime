# ps_onnxruntime_test

A tool inspired by onnxruntme_perf_test, but lighter and with some additional features.
It's primary purpose is to simplify evaluation of NPU driver, ORT EP and/or model versions.
This tool works with both PSORT and WinML.

The tool supports 3 stages of evaluation:
1. **Data Generation** `--stage -1`: Performs CPU inferences, calculates the L2Norm(CPU), and stores all outputs in npy format.
1. **Model compilation** `--stage 0`: Performs model compilation check and reports compilation time and compilation resource usage.
1. **Model performance** `--stage 1`: Keeps track of the model performance by recording inference time, memory usage and CPU usage. Then, checks the performance by comparing average inference time with a predefined threshold.
1. **Model accuracy** `--stage 2`: Records output tensors from quantized and reference models. Then calculate the L2 norm between tensors and check the accuracy by comparing the L2 norm with a predefined threshold.

## Setup

- Install CMake (for PSORT build)
- Install latest version of [Visual Studio](https://visualstudio.microsoft.com/downloads/) (any edition)(for WinML build).
- Provide read access to [OrtEpValidation](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/OrtEpValidation) Azure Artifact Feed.
    - For access, please contact one of the feed owners: mihailogrbic or micob.

## Building
For a simplified way to build the tool, you can use the provided [`build.bat`](build.bat) script. This script will automatically set up the necessary environment variables and build the tool for you.

### Build Types and Feature Flag
There are two supported build configurations using a single feature flag:

1. **PSORT Build** (CMake): Default when `USE_WINML_FEATURES=OFF` or not set
   - Uses CMake build system
   - Includes PSORT-specific functionality (execution provider parameter, encryption support)
   
2. **WinML Build** (MSBuild): When `USE_WINML_FEATURES=ON`
   - Uses MSBuild/Visual Studio project files 
   - Includes WinML-specific functionality (EP policy, compilation features)
   
**Note**: Common code that's not specific to either PSORT or WinML is not put under any compilation flag.

### Quick Build
You can run the script with the following command:
```
.\build.bat {psort|winml} {qnn|ov|vitisai} {Release|Debug}
```

### Manual PSORT Build (CMake) 
If more control is needed, you can build the tool manually using CMake. The following command will configure the build for the specified architecture and dependencies:
```
cmake -S . -A {x64|ARM64} -B <build dir>
    [-DONNXRUNTIME_NUGET_DIR=<ps-onnxruntime nuget local dir>]
    [-DONNXRUNTIME_NUGET_VERSION=<version>]
    [-DEP={qnn|ov|vitisai}]
    [-DUSE_WINML_FEATURES=OFF]  # Default for PSORT build

cmake --build <build dir> --config {Release|Debug}
```

### Manual WinML Build (MSBuild)
For WinML builds, use MSBuild with the Visual Studio project:
```
cd winml\EpValidationToolWinML
nuget restore EpValidationToolWinML.sln
msbuild EpValidationToolWinML.vcxproj /t:Clean;Build /p:Configuration={Release|Debug} /p:Platform={x64|ARM64}
```
The `USE_WINML_FEATURES` flag is already defined in the vcxproj file.

### Build Options
If `-DONNXRUNTIME_NUGET_DIR` is not specified, the nuget will be downloaded and installed.

If `-DONNXRUNTIME_NUGET_VERSION` is not specified, the default version will be used. The version number will be printed in the log.

If `-DEP` is not specified, the tool will be built for CPU.

## Running the tests
1. Download quantized and reference models and place them together inside models directory:
    1. [PSJ quantized model](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/ModelsProcessed/NuGet/embtxt_transformer_c-onnx-all-onnx-1.17-int-encrypted/overview/1.0.280-main)

    1. [PSJ reference model](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/ModelsIngested/UPack/florence_v1_6_2_d3_tulrv6_multi_text_transformer)

1. Download PSJ model's sample [dataset](https://perceptiveshelldata.blob.core.windows.net/ep-validation-data/psj).

1. Copy [dataset_configs/psj.json](dataset_configs/psj.json) to the downloaded datset dir, next to input node folders _embedding_ and _float_attention_mask_, and rename it to `dataset_config.json`.

1. Example command for running the tool (PSORT version)
    ```
    .\install_ov_Release\bin\ps_onnxruntime_test.exe --stage 2 --sessionOptions "session.disable_cpu_ep_fallback|1" --epOptions "htp_performance_mode|high_performance htp_graph_finalization_optimization_mode|3" --executionProvider ov --perfThreshold "15" --accThreshold "output|0.05" --modelPath "models\embtxt_transformer_c.quant.onnxe" --refModelPath "models\florence_v1_6_2_d3_tulrv6_multi_text_transformer.onnx" --datasetDir "dataset_configs\psj" --outputDir "outputs" --modelKey "long_key"
    ```
1. Example command for running the tool (WinML version)
    ```
    EpValidationToolWinML.exe --stage 2  --perfThreshold "68" --accThreshold "output|0.1" --modelPath "PSJ.quant.onnx"  --datasetDir "psj\psj" --outputDir "outputs" --wasdkMajorMinor 0x00010008 --wasdkVersionTag "" --wasdkPackageVersion 0x1F400271014A0000
    ```
    ```
    0x00010008 represents 1.8
    0x1F400271014A0000 is hex code of 8000.625.330.0(Major.Minor.Build.Revision) runtime version
    wasdkVersionTag can be ""/"experimental" /etc
 
    Passing these params are optional
    ```
2. Passing Custom EP/Device details is supported in Winml based exe ( for filtering specific EP/device for inference)
    ```
    EpValidationToolWinML.exe --stage 2  --perfThreshold "68" --accThreshold "output|0.1" --modelPath "PSJ.quant.onnx"  --datasetDir "psj\psj" --outputDir "outputs" --epName OpenVINOExecutionProvider --epDeviceType NPU --epDeviceId 25662 --epVendorId 32902
	```
	
    ```
    All these 4 params are optional ( epName , epDeviceType ,epDeviceId ,epVendorId )
    ```

The `ps_onnxruntime_test` also comes with a wrapper PowerShell script `.\scripts\run_model_tests.ps1` that is used for a more robust interaction with the tool. It is written with CI automation in mind, as it simplifies the process of running model tests. It loads a JSON configuration, downloads necessary files (not yet enabled), converts the configuration to command-line arguments, and executes the test with the specified parameters.

An example command for invoking the script:
```
.\scripts\run_model_tests.ps1 -modelDir "models" -datasetDir "dataset_configs\psj" -configPath "scripts\test_configs\psj_test_config.json" -executionProvider "ov" -stage "2" -exePath "install_ov_Release\bin\ps_onnxruntime_test.exe" -modelKey "long_key"
```

***Note**: Both `ps_onnxruntime_test.exe` and `run_model_tests.ps1` also support decrypted qdq models (with `.quant.onnxe` extension). In such case, remove `--modelKey`/`-modelKey` argument.*

## Adding support for a new model

### Dataset config

Provide a new dataset config in `dataset_configs` folder. The config should contain the following fields:
- `input_to_dir`: A mapping of input node names to their corresponding directories in the dataset. Each dataset directory should contain the input samples for the corresponding node.
- `output_to_dir`: A mapping of output node names to the directories that contains the output dataset. All of them should have same number for samples for all the nodes.

For example dataset config refer to [dataset_configs/psj.json](dataset_configs/psj.json).

### Dataset

Prepare the dataset that will be used to validate the model.
The dataset should be organized with one directory per input layer.
Each directory must contain the same number of .npy files, where each file represents a tensor of the appropriate shape.
File names must be consistent across all directories.

Once dataset is tested locally, contact someone with owner or contributer role to upload the dataset to [ep-validation-data](https://perceptiveshelldata.blob.core.windows.net/ep-validation-data) blob container.

### Model validation config

Provide a new validation config in [validation_configs](validation_configs) folder. The config should contain the following fields in form of a json file:
- `sessionOptions`: Set options such as `disable_cpu_ep_fallback` (you can find the full list of session options [here](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)).
- `executionProvider`: Name of the execution provider. Can be `qnn`, `ov` or `vitisai`.
- `epOptions`: Specified for each execution provider (as an example, the list of QNN EP options can be found [here](https://github.com/microsoft/onnxruntime/blob/788ca51b044bf1c7379a065213ec1b56c978c55f/include/onnxruntime/core/session/onnxruntime_c_api.h#L3648-L3694)).
- `perf_threshold`: Threshold for average inference time provided in milliseconds.
- `accuracy_threshold`: L2 norm thresholds for each output tensor.

For example test config refer to [validation_configs/psj.json](validation_configs/psj.json).

### Model map configuration

Add model information to [model_map.json](model_map.json). This file centralizes all model-related configurations. The config should contain the following structure:

Where:
- `model_name`: Identifier for the model (e.g., `psa`, `psf`, `psi`, `psj`)
- `stage`: `ps_onnnxruntime_test` tool evaluation stage (see [here](#ps_onnxruntime_test))
- `dataset_package`: Configuration for the model's dataset package
  - `nuget_feed`: Name of the NuGet feed containing the dataset
  - `package_name`: Dataset package name
  - `package_version`: Dataset package version
- `quantized_model`: Configuration for the quantized model package
  - `nuget_feed`: Name of the NuGet feed containing the model
  - `package_name`: Model package name
  - `package_version`: Model package version
  - `encryption_key`: Encryption key for the model


## ORT EP validation pipeline

The pipeline is defined in [pipeline/ep-validation.yml](pipeline/ep-validation.yml) and includes two stages:
- Build: ps_onnxruntime_test is compiled for all supported platforms: OV, VitisAI, and QNN.
- Package: A directory is created containing all necessary scripts and configuration files for local EP validation. Once complete, a NuGet package is generated from this directory and published to the [OrtEpValidation](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/OrtEpValidation) feed.

### ep-validation NuGet package content
```
│   common_utils.ps1
│   ep-validation-ov.nuspec
│   run_ep_validation.ps1
│   setup_ep_validation.ps1
│   model_map.json
│   raw_to_npy.py
│
├───bin/
│       ps_onnxruntime_test.exe
│       ... other dependencies
│
├───dataset_configs
│       psj.json
|       ... other dataset configs
│
└───validation_configs
        psj.json
        ... other model validation configs
```

### Local EP validation

Prerequisties:
- python should be installed.
    - required packages (listed in [requirements.txt](requirements.txt)) will be installed with [setup_ep_validation.ps1](scripts/setup_ep_validation.ps1) 
- `nuget.exe` should be added to system path.
- `azcopy.exe` should be added to system path.
- [OrtEpValidation](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/OrtEpValidation) feed access.
- [ModelCollaterals](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/ModelCollaterals) feed access.
- [ps-packaging](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/ps-packaging) feed access.


***Warning**: Please note that placing the validation package deeper within the system directory structure may lead to errors due to excessively long paths. This is especially likely because model NuGet packages often have long names.*

To run EP validation on local NPU machine follow these steps:

1. Download ep-validation NuGet package for the specific platform from [OrtEpValidation](https://devicesasg.visualstudio.com/PerceptiveShell/_artifacts/feed/OrtEpValidation).
1. Modify validation configs per model if needed
1. Modify the [model_map.json](model_map.json) (if needed)
1. Run `setup_ep_validation.ps1`
    - Downloads quantized models from NuGet feeds.
    - Downloads dataset packages from Azure Artifacts Universal Packages.
    - Organizes dataset files into appropriate folder structure.
    - Converts raw binary data files to numpy (.npy) format if needed.
    - Copies dataset configs to appropriate dataset directories.
1. Run `run_ep_validation.ps1`
    - Calls `ps_onnxruntime_test.exe` for each configuration and dumps output in `outputs/<validation_config_filename>` directory.
