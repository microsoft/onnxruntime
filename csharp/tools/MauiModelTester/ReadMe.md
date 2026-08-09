# MAUI Model Tester
## Usage

Run create_test_data.py to specify
  - path to the model you wish to use
  - symbolic dimension values to use if needed
  - any specific input if the randomly generated input will not be good enough
	  - you can create specific input with /tools/python/onnx_test_data_utils.py
      - see the comments in create_test_data.py for more details
  - expected output if saving the output from running the model locally is not good enough
    - the model will be executed when creating the test data to validate the input

This will copy the model to Resources\Raw\test_data\model.onnx and the test data files to
Resources\Raw\test_data\test_data_set_0

The MAUI application will read the model and test data from those locations and should need no other changes to be able
to execute the model.

The project uses builds from the nightly feed by default to keep things simple.

If it was part of the main ONNX Runtime C# solution we'd have to
  - add the ORT nightly feed to the top level nuget.config
    - this potentially adds confusion about where nuget packages come from in unit tests
  - keep updating the referenced nightly packages so they remain valid so the complete solution builds in the CI

You will need to manually add the ORT-Nightly feed to the packageSources section of the nuget.config in this directory.
  - `<add key="ORT-Nightly" value="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/index.json" />`
  - This feed isn't allowed in the checked in nuget.config

If you need to update the ORT packages used by the app to the latest nightly:
- In Visual Studio, Tools -> Nuget Package Manager -> Manage NuGet Packages for Solution...
- Make sure 'Include prerelease' is checked
- Set Package Source to ORT-Nightly
- Update Microsoft.ML.OnnxRuntime, Microsoft.ML.OnnxRuntime.Managed and Microsoft.ML.OnnxRuntime.Extensions to the
latest build.

## Testing C# or native code changes

If you have new code to test the easiest way is to run the nuget packaging pipeline on
https://aiinfra.visualstudio.com/Lotus/_build against your branch. Download the native and managed nuget packages from
the CI artifacts and update the nuget.config to point to the directory they are in.
This can be used to test both native and C# code changes.

If you wish to test local changes to the C# code, you can create a local nuget package and add the directory it's in to
nuget.config. With the current setup you'd first need to build ORT with the `--build_csharp` param to update the C#
Directory.build.props and create the native library for the current platform, as the current packaging infrastructure
creates the native and managed packages at the same time and requires a native library to exist.

Alternatively, you can open /csharp/OnnxRuntime.CSharp.sln and temporarily add
/csharp/tools/MauiModelTester/MauiModelTester.csproj to it. The csproj should automatically adjust to use a project
reference to /csharp/src/Microsoft.ML.OnnxRuntime/Microsoft.ML.OnnxRuntime.csproj instead of a package reference to
the Microsoft.ML.OnnxRuntime.Managed nuget package. Note that you must still run build.bat/build.sh with
`--build_csharp` to generate /csharp/Directory.Build.props, but can run with the `--update` parameter so no native build
is done. Most likely you should grab the latest nightly native package from the packaging pipeline so the native code is
compatible.

## Local build setup

The following commands _should_ install the necessary workloads to create the managed package including mobile targets,
build the managed library and create the native (local build only) and managed packages. The packages will be in the
ORT build output directory (e.g. build/Windows/Debug/Debug). The native package will only contain the runtime for the
current platform (e.g. Windows 64-bit if you're building on Windows) so can't be used for testing other platforms. Use
the native package from the nightly feed or a packaging CI.

```
dotnet workload install ios android macos
dotnet workload restore .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=All
msbuild -t:restore .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=All
msbuild -t:build .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=All
msbuild .\OnnxRuntime.CSharp.proj -t:CreatePackage -p:OrtPackageId=Microsoft.ML.OnnxRuntime -p:Configuration=Debug -p:Platform="Any CPU"
```

## Example test data creation

Example commands to create test data

### PyTorch mobilenet V3 with a symbolic dimension called 'batch' that we set to 1.
https://pytorch.org/vision/main/models/mobilenetv3.html exported with torch.onnx.export with the batch dimension
being dynamic. We provide a value of 1 for the batch size to use in our testing.

`create_test_data.py -s batch=1 -m pytorch_mobilenet_v3.onnx`

### SuperResolution with pre-processing in the model.

As the model has pre-processing in it from onnxruntime-extensions, we want to provide the raw bytes from a jpeg or png.

Convert image to protobuf file with name of 'image' to match the model input. The output filename doesn't matter.
`python ..\..\..\tools\python\onnx_test_data_utils.py --action image_to_pb --raw --input lion.png --output superres_input.pb --name image`

Create test data using this input.
`create_test_data.py --input_data superres_input.pb --model_path RealESRGAN_with_pre_post_processing.onnx`
