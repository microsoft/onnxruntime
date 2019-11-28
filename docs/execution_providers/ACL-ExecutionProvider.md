## ACL Execution Provider

[Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) is an open source inference engine maintained by Arm and Linaro companies. The integration of ACL as an execution provider (EP) into ONNX Runtime accelerates performance of ONNX model workloads across Armv8 cores.

### Build ACL execution provider
Developers can use ACL library through ONNX Runtime to accelerate inference performance of ONNX models. Instructions for building the ACL execution provider from the source is available below.

### Supported BSP
* i.MX8QM BSP
Install i.MX8QM BSP:
```
source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh
```

Setup build environment:
```
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```

Confiure and build ONNX Runtime with ACL support:
```
export CMAKE_ARGS="-DONNX_CUSTOM_PROTOC_EXECUTABLE=/PROTOC_PATH/protoc"
build.sh --path_to_protoc_exe /PROTOC_PATH/protoc --config RelWithDebInfo --build_shared_lib --use_openmp --update --build
```

--use_acl=ACL_1902

Deploy ONNX runtime on the i.MX 8QM board
```
libonnxruntime.so.0.5.0
onnxruntime_perf_test
onnxruntime_test_all
```

### Supported backend
* i.MX8QM Armv8 CPUs

### Using the ACL execution provider
#### C/C++
To use ACL as execution provider for inferencing, please register it as below.
```
InferenceSession session_object{so};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::ACLExecutionProvider>());
status = session_object.Load(model_file_name);
```
The C API details are [here](../C_API.md#c-api).

### Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../ONNX_Runtime_Perf_Tuning.md)

When/if using [onnxruntime_perf_test](../../onnxruntime/test/perftest), use the flag -e acl
