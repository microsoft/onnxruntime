To test the iOS or Android samples the native build of ONNX Runtime is required and must be in a specific location.

Only the native build for the platform you are testing on is required. 
e.g. if you're testing using an Android device that is arm64, you only need the libonnxruntime.so for arm64-v8a.
The version of the native build should match the checked-out version of the ONNX Runtime repository you're currently using as closely as possible. 
Otherwise mismatches with the native entry points is possible and could cause crashes. 

To acquire the native build you can:
  - build it yourself
    - [Android](https://onnxruntime.ai/docs/build/android.html) build instructions
    - [iOS](https://onnxruntime.ai/docs/build/ios.html) build instructions
  - extract it from the Microsoft.ML.OnnxRuntime nuget package using [NuGetPackageExplorer](https://github.com/NuGetPackageExplorer/NuGetPackageExplorer)
    - release version is [here](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) 
    - integration test version is [here](https://int.nugettest.org/packages/Microsoft.ML.OnnxRuntime/)
      - this is frequently updated and should work if you're currently using the `master` branch of ONNX Runtime
  - or if you have access to the internal packaging pipelines
    - the Zip-Nuget-Java-Nodejs Packaging Pipeline produces the native package as an artifact under `drop-signed-nuget-CPU`
      - run a build for your current branch in the pipeline to ensure the native build matches exactly


For iOS the native build should be at one or more of:

  - <ORT repo root>\build\iOS\iphoneos\Release\Release-iphoneos\onnxruntime.framework for an iOS device
  - <ORT repo root>\build\iOS\iphonesimulator\Release\Release-iphonesimulator\onnxruntime.framework for an iOS simulator

For Android the native build should be at one or more of:

  - <ORT repo root>\build\Android\arm64-v8a\Release\libonnxruntime.so for an 64-bit arm device
  - <ORT repo root>\build\Android\armeabi-v7a\Release\libonnxruntime.so for an 32-bit arm device
  - <ORT repo root>\build\Android\x86\Release\libonnxruntime.so for an x86 Android emulator
  - <ORT repo root>\build\Android\x86_64\Release\libonnxruntime.so for an x86_64 Android emulator