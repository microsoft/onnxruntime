MAUI app that is used directly with the C# bindings project to do some basic local testing of using an InferenceSession
in an app. 

This is **NOT** intended to be a sample for end users. See https://github.com/microsoft/onnxruntime-inference-examples
for those, in particular under the c_sharp, mobile/examples/Xamarin and mobile/examples/MAUI directories.

NOTE: Currently the csproj is not included in \csharp\OnnxRuntime.sln as it requires Visual Studio 2022 Preview. 
Microsoft.ML.OnnxRuntime.InferenceSample.Maui.sln is provided temporarily to enable local build/test, 
however as Microsoft.ML.OnnxRuntime.InferenceSample.Maui.csproj is structured so it is ready for inclusion in 
OnnxRuntime.sln it internally references some other csproj files. Due to that the nuget restore has to be manually run.

From Visual Studio 2022 Preview, go to View -> Terminal.
From the resulting terminal window, execute the command `nuget restore .\Microsoft.ML.OnnxRuntime.InferenceSample.Maui.csproj`

After that a build should work.

The instructions in [../readme.me](../readme.md) regarding making onnxruntime binaries from other platforms available apply here as well.