The MAUI test project can be optionally used with a pre-built ONNX Runtime native nuget package (Microsoft.ML.OnnxRuntime).

To do so, specify the `UsePrebuiltNativePackage` and `CurrentOnnxRuntimeVersion` properties when building the project. These can be set via the command-line or as environment variables.

For example:

```cmd
dotnet build csharp\test\Microsoft.ML.OnnxRuntime.Tests.MAUI\Microsoft.ML.OnnxRuntime.Tests.MAUI.csproj --property:UsePrebuiltNativePackage=true --property:CurrentOnnxRuntimeVersion=1.19.2 --source directory_containing_native_nuget_package --source https://api.nuget.org/v3/index.json
```
