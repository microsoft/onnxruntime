To test you need the libonnxruntime.so for the various Android architectures.

The test project looks for these in '..\..\..\build\Android\<architecture>\Release\libonnxruntime.so'.
e.g. '..\..\..\build\Android\arm64-v8a\Release\libonnxruntime.so'

'..\..\..' is the root directory of the repository.

Build onnxruntime for the required architecture if you're testing changes in the native code.

Alternatively, if you're testing the C# code you can extract the AAR from the nightly nuget Microsoft.ML.OnnxRuntime package.
- Get the nupkg from https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/ORT-Nightly
- Unzip it, and the AAR is in `runtimes/android/native/onnxruntime.aar`.
- Unzip the AAR. The `jni` directory contains a directory for each architecture with the libonnxruntime.so.
- Copy the libonnxruntime.so for the required architectures to /build/Android/<architecture>/Release.
  - e.g. x86_64 for running the emulator on an amd64 machine, and arm64-v8a for running on an arm64 Android device
