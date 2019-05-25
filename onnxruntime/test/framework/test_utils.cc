// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_utils.h"
namespace onnxruntime {
namespace test {
IExecutionProvider* TestCPUExecutionProvider() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return &cpu_provider;
}

#ifdef USE_CUDA
IExecutionProvider* TestCudaExecutionProvider() {
  static CUDAExecutionProviderInfo info;
  static CUDAExecutionProvider cuda_provider(info);
  return &cuda_provider;
}
#endif

#ifdef USE_TENSORRT
IExecutionProvider* TestTensorrtExecutionProvider() {
  static TensorrtExecutionProvider trt_provider;
  return &trt_provider;
}
#endif
}  // namespace test
}  // namespace onnxruntime
