// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "default_providers.h"
#include "providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Mkldnn(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(int device_id, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_BrainSlice(uint32_t ip, int, int, bool, const char*, const char*, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt();

namespace test {

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  return CreateExecutionProviderFactory_CPU(enable_arena)->CreateProvider();
}

std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider() {
#ifdef USE_TENSORRT
  return CreateExecutionProviderFactory_Tensorrt()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider() {
#ifdef USE_CUDA
  return CreateExecutionProviderFactory_CUDA(0)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultMkldnnExecutionProvider(bool enable_arena) {
#ifdef USE_MKLDNN
  return CreateExecutionProviderFactory_Mkldnn(enable_arena ? 1 : 0)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider() {
#ifdef USE_NUPHAR
  return CreateExecutionProviderFactory_Nuphar(0, "")->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultBrainSliceExecutionProvider() {
#ifdef USE_BRAINSLICE
  return CreateExecutionProviderFactory_BrainSlice(0, 1, -1, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin")->CreateProvider();
#else
  return nullptr;
#endif
}

}  // namespace test
}  // namespace onnxruntime
