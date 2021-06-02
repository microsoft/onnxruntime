---
title: Enabling NNAPI or CoreML Execution Providers
parent: Deploy ONNX Runtime Mobile
grand_parent: How to
nav_order: 6
---

## Using NNAPI and CoreML with ONNX Runtime Mobile

- Usage of NNAPI on Android platforms is via the NNAPI Execution Provider (EP). 
  - See the [NNAPI Execution Provider](../../reference/execution-providers/NNAPI-ExecutionProvider) documentation for more details.
- Usage of CoreML on iOS and macOS platforms is via the CoreML EP. 
  - See the [CoreML Execution Provider](../../reference/execution-providers/CoreML-ExecutionProvider) documentation for more details.

The pre-built ONNX Runtime Mobile package includes the NNAPI EP on Android, and the CoreML EP on iOS.

If performing a custom build of ONNX Runtime, support for the NNAPI EP or CoreML EP must be enabled when building.


### Create a minimal build with NNAPI EP or CoreML EP support

Please see [the instructions](../build/android-ios) for setting up the Android or iOS environment required to build. The Android build can be cross-compiled on Windows or Linux. The iOS/macOS build must be performed on a mac machine.

Once you have all the necessary components setup, follow the instructions to [create the custom build](custom-build), with the following changes:
  - Replace `--minimal_build` with `--minimal_build extended` to enable support for execution providers that dynamically create kernels at runtime, which is required by the NNAPI EP and CoreML EP.
  - Add `--use_nnapi` to include the NNAPI EP in the build
  - Add `--use_coreml` to include the CoreML EP in the build

##### Example build commands with the NNAPI EP enabled:

- Windows example:
  `<ONNX Runtime repository root>.\build.bat --config MinSizeRel --android --android_sdk_path D:\Android --android_ndk_path D:\Android\ndk\21.1.6352462\ --android_abi arm64-v8a --android_api 29 --cmake_generator Ninja --minimal_build extended --use_nnapi --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>`
- Linux example:
  `<ONNX Runtime repository root>./build.sh --config MinSizeRel --android --android_sdk_path /Android --android_ndk_path /Android/ndk/21.1.6352462/ --android_abi arm64-v8a --android_api 29 --minimal_build extended --use_nnapi --disable_ml_ops --disable_exceptions --build_shared_lib --skip_tests --include_ops_by_config <config file from model conversion>`

-------

Next: [Limitations](limitations)