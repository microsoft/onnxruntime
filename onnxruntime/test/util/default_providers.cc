// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "default_providers.h"
#include "providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/providers.h"
#include "core/framework/bfc_arena.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CPU(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_NGraph(const char* ng_backend_type);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const char* device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool, const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rknpu();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(int device_id);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ArmNN(int use_arena);

namespace test {

std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena) {
  return CreateExecutionProviderFactory_CPU(enable_arena)->CreateProvider();
}

std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider() {
#ifdef USE_TENSORRT
  if (auto factory = CreateExecutionProviderFactory_Tensorrt(0))
    return factory->CreateProvider();
#endif
  return nullptr;
}

std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider() {
#ifdef USE_MIGRAPHX
  return CreateExecutionProviderFactory_MIGraphX(0)->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider() {
#ifdef USE_OPENVINO
  return CreateExecutionProviderFactory_OpenVINO("")->CreateProvider();
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

std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider(bool enable_arena) {
#ifdef USE_DNNL
  if (auto factory = CreateExecutionProviderFactory_Dnnl(enable_arena ? 1 : 0))
    return factory->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
#endif
  return nullptr;
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

std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider() {
#ifdef USE_NNAPI
  return CreateExecutionProviderFactory_Nnapi()->CreateProvider();
#else
  return nullptr;
#endif
}

std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider() {
#ifdef USE_RKNPU
  return CreateExecutionProviderFactory_Rknpu()->CreateProvider();
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

std::unique_ptr<IExecutionProvider> DefaultArmNNExecutionProvider(bool enable_arena) {
#ifdef USE_ARMNN
  return CreateExecutionProviderFactory_ArmNN(enable_arena)->CreateProvider();
#else
  ORT_UNUSED_PARAMETER(enable_arena);
  return nullptr;
#endif
}

}  // namespace test
}  // namespace onnxruntime
