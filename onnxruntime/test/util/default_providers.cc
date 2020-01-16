// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "default_providers.h"
#include "providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(bool use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(int device_id, bool use_cuda_arena, bool use_cpu_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(bool use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_NGraph(const char* ng_backend_type);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_BrainSlice(uint32_t ip, int, int, bool, const char*, const char*, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id, bool use_cuda_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const char* device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(bool use_arena);

namespace test {

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  return CreateExecutionProviderFactory_CPU(enable_arena)->CreateProvider();
}

std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider(bool enable_cuda_arena) {
#ifdef USE_TENSORRT
  return CreateExecutionProviderFactory_Tensorrt(0, enable_cuda_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_cuda_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider() {
#ifdef USE_OPENVINO
  return CreateExecutionProviderFactory_OpenVINO("CPU")->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider(bool enable_cuda_arena, bool enable_cpu_arena) {
#ifdef USE_CUDA
  return CreateExecutionProviderFactory_CUDA(0, enable_cuda_arena, enable_cpu_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_cuda_arena);
  ORT_UNUSED_PARAMETER(enable_cpu_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider(bool enable_arena) {
#ifdef USE_DNNL
  return CreateExecutionProviderFactory_Dnnl(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultNGraphExecutionProvider() {
#ifdef USE_NGRAPH
  return CreateExecutionProviderFactory_NGraph("CPU")->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider(bool allow_unaligned_buffers) {
#ifdef USE_NUPHAR
  return CreateExecutionProviderFactory_Nuphar(allow_unaligned_buffers, "")->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(allow_unaligned_buffers);
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

std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider() {
#ifdef USE_NNAPI
  return CreateExecutionProviderFactory_Nnapi()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_arena) {
#ifdef USE_ACL
  return CreateExecutionProviderFactory_ACL(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

}  // namespace test
}  // namespace onnxruntime
