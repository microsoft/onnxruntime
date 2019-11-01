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

Confiure ONNX Runtime with ACL support:
```
cmake ../onnxruntime-arm-upstream/cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=ON -Donnxruntime_DEV_MODE=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_CUDA=OFF -Donnxruntime_USE_NSYNC=OFF -Donnxruntime_CUDNN_HOME= -Donnxruntime_USE_JEMALLOC=OFF -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_CSHARP=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_USE_EIGEN_FOR_BLAS=ON -Donnxruntime_USE_OPENBLAS=OFF -Donnxruntime_USE_ACL=ON -Donnxruntime_USE_MKLDNN=OFF -Donnxruntime_USE_MKLML=OFF -Donnxruntime_USE_OPENMP=ON -Donnxruntime_USE_TVM=OFF -Donnxruntime_USE_LLVM=OFF -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF -Donnxruntime_USE_BRAINSLICE=OFF -Donnxruntime_USE_NUPHAR=OFF -Donnxruntime_USE_EIGEN_THREADPOOL=OFF -Donnxruntime_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Build ONNX Runtime library, test and performance application:
```
make -j 6
```

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
